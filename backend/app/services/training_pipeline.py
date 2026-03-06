"""MindMesh AI — Model Training Pipeline.

Orchestrates the training → evaluation → registration → promotion
workflow for all AI models.

Training steps:
  1. Collect labelled data (from database or synthetic dataset)
  2. Split into train / validation / test sets
  3. Train model variant
  4. Evaluate on held-out test set
  5. Register in model registry as 'candidate'
  6. Compare against active model
  7. Optionally promote if candidate is better

Supported model types:
  • emotion_detection:  TF-IDF + NearestCentroid / keyword-rule baseline
  • sentiment_analysis: Lexicon-based / TF-IDF regression
  • risk_scoring:       Weighted composite / gradient boosting
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..logging_config import logger
from .model_evaluation import (
    evaluate_classification,
    evaluate_regression,
    evaluate_risk_scoring,
    compare_models,
    ClassificationMetrics,
    RegressionMetrics,
    RiskCalibrationMetrics,
    ModelComparison,
)
from .model_registry import registry, ModelVersion


# ─── Data Structures ─────────────────────────────────────────────


@dataclass
class TrainingDataset:
    """A labelled dataset for model training."""

    name: str
    texts: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.texts)

    def split(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42
    ) -> Tuple["TrainingDataset", "TrainingDataset", "TrainingDataset"]:
        """Split into train, validation, test sets."""
        rng = random.Random(seed)
        indices = list(range(self.size))
        rng.shuffle(indices)

        n_train = int(self.size * train_ratio)
        n_val = int(self.size * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        def _subset(idx_list: List[int], suffix: str) -> "TrainingDataset":
            return TrainingDataset(
                name=f"{self.name}_{suffix}",
                texts=[self.texts[i] for i in idx_list],
                labels=[self.labels[i] for i in idx_list] if self.labels else [],
                scores=[self.scores[i] for i in idx_list] if self.scores else [],
            )

        return _subset(train_idx, "train"), _subset(val_idx, "val"), _subset(test_idx, "test")


@dataclass
class TrainingResult:
    """Result of a training pipeline run."""

    model_name: str
    new_version: str
    training_samples: int
    test_samples: int
    metrics: Dict[str, float]
    comparison: Optional[ModelComparison] = None
    promoted: bool = False
    duration_seconds: float = 0.0
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ─── Synthetic Dataset Generation ────────────────────────────────


def generate_emotion_dataset(n: int = 500, seed: int = 42) -> TrainingDataset:
    """Generate a synthetic labelled emotion dataset for training.

    Creates realistic-looking text snippets with known emotion labels.
    """
    rng = random.Random(seed)

    templates = {
        "happy": [
            "I feel great today, everything is going well",
            "Had an amazing time with my friends after school",
            "I'm really excited about the upcoming project",
            "Feeling grateful for my family and teachers",
            "Today was a wonderful day, I learned so much",
            "I'm proud of my test results, worked hard for this",
            "Laughed a lot today, my classmates are fun",
            "Feeling confident about the future",
        ],
        "sad": [
            "I feel really down today and don't know why",
            "Nobody seems to understand how I feel",
            "I miss my old friends since we moved schools",
            "Everything feels pointless sometimes",
            "I cried today and couldn't stop",
            "Feeling lonely even when surrounded by people",
            "I don't enjoy things I used to anymore",
            "Today was really hard, I just want it to be over",
        ],
        "anxious": [
            "I'm really worried about the exam tomorrow",
            "Can't stop thinking about what might go wrong",
            "My heart races whenever I have to present in class",
            "I feel nervous all the time and can't relax",
            "What if I fail and disappoint everyone",
            "I keep having trouble sleeping because of worry",
            "Social situations make me really uncomfortable",
            "I feel overwhelmed by everything I need to do",
        ],
        "angry": [
            "I'm so frustrated with how unfair things are",
            "Someone said something really hurtful today",
            "I can't stand when people don't listen to me",
            "Everything is making me irritated lately",
            "I got into an argument and I'm still upset",
            "It's not fair that I get blamed for everything",
            "I feel like screaming sometimes",
            "People keep pushing my limits and I'm done",
        ],
        "neutral": [
            "Today was an ordinary day at school",
            "Had lunch and then went to class as usual",
            "The weather was nice today",
            "Studied for a bit and then watched some TV",
            "Nothing much happened today",
            "Went through my normal routine",
            "Classes were okay, nothing special",
            "Just another regular day",
        ],
    }

    modifiers = [
        "", " I think.", " Not sure what to do.", " Just wanted to share.",
        " It's been like this for a while.", " Hope tomorrow is different.",
    ]

    texts = []
    labels = []

    for _ in range(n):
        emotion = rng.choice(list(templates.keys()))
        text = rng.choice(templates[emotion])
        text += rng.choice(modifiers)

        # Add noise: 5% label flip to simulate real-world messiness
        if rng.random() < 0.05:
            emotion = rng.choice(list(templates.keys()))

        texts.append(text)
        labels.append(emotion)

    return TrainingDataset(
        name="synthetic_emotion",
        texts=texts,
        labels=labels,
        metadata={"generated": True, "n": n, "seed": seed},
    )


def generate_sentiment_dataset(n: int = 500, seed: int = 42) -> TrainingDataset:
    """Generate a synthetic labelled sentiment dataset."""
    emotion_ds = generate_emotion_dataset(n=n, seed=seed)
    rng = random.Random(seed + 1)

    # Map emotions to approximate sentiment scores
    score_map = {
        "happy": (0.4, 0.9),
        "sad": (-0.8, -0.2),
        "anxious": (-0.6, -0.1),
        "angry": (-0.7, -0.2),
        "neutral": (-0.1, 0.2),
    }

    scores = []
    for label in emotion_ds.labels:
        lo, hi = score_map.get(label, (-0.1, 0.1))
        scores.append(round(rng.uniform(lo, hi), 3))

    return TrainingDataset(
        name="synthetic_sentiment",
        texts=emotion_ds.texts,
        labels=emotion_ds.labels,
        scores=scores,
        metadata={"generated": True, "n": n, "seed": seed},
    )


# ─── TF-IDF Feature Extraction (lightweight, no external deps) ──


class SimpleTfidfVectorizer:
    """Minimal TF-IDF vectorizer for the training pipeline.

    Avoids heavy sklearn import at module level but can be used
    for lightweight model training within the app.
    """

    def __init__(self, max_features: int = 3000) -> None:
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> "SimpleTfidfVectorizer":
        """Build vocabulary and compute IDF from documents."""
        word_doc_count: Counter = Counter()
        all_words: Counter = Counter()

        for doc in documents:
            words = set(self._tokenize(doc))
            for w in words:
                word_doc_count[w] += 1
            for w in self._tokenize(doc):
                all_words[w] += 1

        # Select top N words by frequency
        top_words = [w for w, _ in all_words.most_common(self.max_features)]
        self.vocabulary_ = {w: i for i, w in enumerate(top_words)}

        n_docs = len(documents)
        self.idf_ = {
            w: math.log((n_docs + 1) / (word_doc_count[w] + 1)) + 1
            for w in self.vocabulary_
        }

        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents into TF-IDF vectors."""
        vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf: Counter = Counter(tokens)
            n_tokens = len(tokens) or 1

            vector = [0.0] * len(self.vocabulary_)
            for word, idx in self.vocabulary_.items():
                if word in tf:
                    tfidf = (tf[word] / n_tokens) * self.idf_.get(word, 1.0)
                    vector[idx] = tfidf

            # L2 normalize
            norm = math.sqrt(sum(v ** 2 for v in vector)) or 1.0
            vector = [v / norm for v in vector]
            vectors.append(vector)

        return vectors

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        self.fit(documents)
        return self.transform(documents)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()


# ─── Simple Nearest-Centroid Classifier ──────────────────────────


class NearestCentroidClassifier:
    """Simple centroid-based classifier.

    Computes the mean vector for each class during training,
    then predicts based on closest centroid (cosine similarity).
    """

    def __init__(self) -> None:
        self.centroids: Dict[str, List[float]] = {}
        self.classes: List[str] = []

    def fit(self, X: List[List[float]], y: List[str]) -> "NearestCentroidClassifier":
        """Fit centroids from training data."""
        class_vectors: Dict[str, List[List[float]]] = {}
        for vec, label in zip(X, y):
            class_vectors.setdefault(label, []).append(vec)

        self.classes = sorted(class_vectors.keys())
        dim = len(X[0]) if X else 0

        for label, vectors in class_vectors.items():
            centroid = [0.0] * dim
            for vec in vectors:
                for i, v in enumerate(vec):
                    centroid[i] += v
            n = len(vectors)
            centroid = [c / n for c in centroid]
            self.centroids[label] = centroid

        return self

    def predict(self, X: List[List[float]]) -> List[str]:
        """Predict class labels."""
        predictions = []
        for vec in X:
            best_label = self.classes[0] if self.classes else "neutral"
            best_sim = -1.0

            for label, centroid in self.centroids.items():
                sim = self._cosine_similarity(vec, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

            predictions.append(best_label)

        return predictions

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai ** 2 for ai in a)) or 1.0
        norm_b = math.sqrt(sum(bi ** 2 for bi in b)) or 1.0
        return dot / (norm_a * norm_b)


# ─── Training Pipelines ─────────────────────────────────────────


async def train_emotion_model(
    dataset: Optional[TrainingDataset] = None,
    version: str = "2.0.0",
    max_features: int = 3000,
    auto_promote: bool = False,
) -> TrainingResult:
    """Train an improved emotion detection model.

    Steps:
      1. Generate/use labelled dataset
      2. TF-IDF feature extraction
      3. Train nearest-centroid classifier
      4. Evaluate on test set
      5. Compare with active model
      6. Register and optionally promote

    Args:
        dataset: Pre-built dataset, or None to generate synthetic.
        version: Version string for the new model.
        max_features: TF-IDF vocabulary size.
        auto_promote: If True, promote if better than active.

    Returns:
        TrainingResult with metrics and comparison.
    """
    import time
    start = time.time()

    if dataset is None:
        dataset = generate_emotion_dataset(n=500)

    train_ds, val_ds, test_ds = dataset.split()

    logger.info(
        f"Training emotion model v{version}: "
        f"train={train_ds.size}, val={val_ds.size}, test={test_ds.size}"
    )

    # Feature extraction
    vectorizer = SimpleTfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_ds.texts)
    X_test = vectorizer.transform(test_ds.texts)

    # Train classifier
    classifier = NearestCentroidClassifier()
    classifier.fit(X_train, train_ds.labels)

    # Predict on test set
    y_pred = classifier.predict(X_test)

    # Evaluate
    metrics = evaluate_classification(
        test_ds.labels, y_pred,
        model_name="emotion_detection",
        model_version=version,
    )

    # Compare with active model
    comparison = None
    active = registry.get_active("emotion_detection")
    if active and active.metrics:
        # Build a pseudo ClassificationMetrics from stored metrics
        active_metrics = ClassificationMetrics(
            model_name="emotion_detection",
            model_version=active.version,
            accuracy=active.metrics.get("accuracy", 0),
            macro_f1=active.metrics.get("macro_f1", 0),
            weighted_f1=active.metrics.get("weighted_f1", 0),
            macro_precision=active.metrics.get("macro_precision", 0),
            macro_recall=active.metrics.get("macro_recall", 0),
        )
        comparison = compare_models(active_metrics, metrics)

    # Register
    new_model = ModelVersion(
        model_name="emotion_detection",
        version=version,
        status="candidate",
        config={
            "type": "tfidf_nearest_centroid",
            "max_features": max_features,
            "train_size": train_ds.size,
            "test_size": test_ds.size,
        },
        metrics={
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "weighted_f1": metrics.weighted_f1,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
        },
        description=f"TF-IDF + NearestCentroid emotion classifier v{version}",
    )
    registry.register(new_model)

    # Auto-promote if better
    promoted = False
    if auto_promote and comparison and comparison.winner == version:
        registry.activate("emotion_detection", version)
        promoted = True
        logger.info(f"Auto-promoted emotion model v{version}")

    duration = time.time() - start

    result = TrainingResult(
        model_name="emotion_detection",
        new_version=version,
        training_samples=train_ds.size,
        test_samples=test_ds.size,
        metrics=new_model.metrics,
        comparison=comparison,
        promoted=promoted,
        duration_seconds=round(duration, 2),
    )

    logger.info(
        f"Emotion training complete: v{version}, "
        f"accuracy={metrics.accuracy:.4f}, f1={metrics.weighted_f1:.4f}, "
        f"promoted={promoted}, duration={duration:.2f}s"
    )

    return result


async def train_sentiment_model(
    dataset: Optional[TrainingDataset] = None,
    version: str = "2.0.0",
    auto_promote: bool = False,
) -> TrainingResult:
    """Train an improved sentiment analysis model.

    Uses TF-IDF features with centroid-based regression
    (weighted average of class centroids by score).

    Args:
        dataset: Pre-built dataset, or None to generate synthetic.
        version: Version string.
        auto_promote: If True, promote if better.

    Returns:
        TrainingResult with metrics.
    """
    import time
    start = time.time()

    if dataset is None:
        dataset = generate_sentiment_dataset(n=500)

    train_ds, val_ds, test_ds = dataset.split()

    logger.info(
        f"Training sentiment model v{version}: "
        f"train={train_ds.size}, val={val_ds.size}, test={test_ds.size}"
    )

    # For sentiment, we use the existing lexicon-based approach
    # but evaluate it properly on the test set to get benchmarks.
    from .sentiment_analysis import analyze_sentiment

    y_true = test_ds.scores
    y_pred = []
    for text in test_ds.texts:
        result = analyze_sentiment(text)
        y_pred.append(result.sentiment_score)

    reg_metrics = evaluate_regression(
        y_true, y_pred,
        model_name="sentiment_analysis",
        model_version=version,
    )

    # Register
    new_model = ModelVersion(
        model_name="sentiment_analysis",
        version=version,
        status="candidate",
        config={
            "type": "lexicon_evaluated",
            "train_size": train_ds.size,
            "test_size": test_ds.size,
        },
        metrics={
            "mae": reg_metrics.mae,
            "rmse": reg_metrics.rmse,
            "r_squared": reg_metrics.r_squared,
            "pearson_correlation": reg_metrics.pearson_correlation,
        },
        description=f"Lexicon sentiment analyser v{version} (re-evaluated)",
    )
    registry.register(new_model)

    # Auto-promote check
    promoted = False
    active = registry.get_active("sentiment_analysis")
    if auto_promote and active and active.metrics:
        active_mae = active.metrics.get("mae", 1.0)
        if reg_metrics.mae < active_mae:
            registry.activate("sentiment_analysis", version)
            promoted = True

    duration = time.time() - start

    return TrainingResult(
        model_name="sentiment_analysis",
        new_version=version,
        training_samples=train_ds.size,
        test_samples=test_ds.size,
        metrics=new_model.metrics,
        promoted=promoted,
        duration_seconds=round(duration, 2),
    )


async def run_full_training_pipeline(
    auto_promote: bool = False,
) -> List[TrainingResult]:
    """Run training for all models sequentially.

    Args:
        auto_promote: Whether to auto-promote better models.

    Returns:
        List of TrainingResult for each model.
    """
    results = []

    logger.info("=== Starting full training pipeline ===")

    emotion_result = await train_emotion_model(
        version="2.0.0", auto_promote=auto_promote
    )
    results.append(emotion_result)

    sentiment_result = await train_sentiment_model(
        version="2.0.0", auto_promote=auto_promote
    )
    results.append(sentiment_result)

    logger.info(
        f"=== Training pipeline complete: "
        f"{len(results)} models trained ==="
    )

    return results
