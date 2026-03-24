"""MindMesh AI — Unsupervised Model Training Pipeline.

Orchestrates fitting of all unsupervised models on available data:

  1. Text Embedding Engine  — TF-IDF vectoriser
  2. Emotion Cluster Engine — K-Means on text embeddings
  3. Anomaly Detection      — Isolation Forest on behavioural features
  4. Student Clustering     — GMM on behavioural profiles
  5. Topic Discovery        — NMF on text corpus

Training steps:
  1. Collect text data from BehavioralRecords (no labels needed)
  2. Fit TF-IDF vectoriser on corpus
  3. Fit K-Means emotion clusters
  4. Build behavioural feature vectors per student
  5. Fit Isolation Forest for anomaly detection
  6. Fit GMM for student clustering
  7. Fit NMF for topic discovery
  8. Evaluate with unsupervised metrics (silhouette, CH index)
  9. Register models in registry
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from ..logging_config import logger
from .text_embeddings import get_embedding_engine, reset_embedding_engine
from .emotion_detection import get_emotion_engine
from .anomaly_detection import get_anomaly_engine, BehavioralFeatureVector
from .student_clustering import get_cluster_engine
from .topic_discovery import get_topic_engine
from .model_registry import registry, ModelVersion


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class UnsupervisedMetrics:
    """Evaluation metrics for unsupervised models."""
    silhouette_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    n_clusters: int = 0
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    inertia: float = 0.0            # K-Means inertia
    reconstruction_error: float = 0.0  # NMF reconstruction error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "silhouette_score": round(self.silhouette_score, 4),
            "calinski_harabasz_score": round(self.calinski_harabasz_score, 4),
            "n_clusters": self.n_clusters,
            "cluster_sizes": self.cluster_sizes,
            "inertia": round(self.inertia, 4),
            "reconstruction_error": round(self.reconstruction_error, 4),
        }


@dataclass
class TrainingResult:
    """Result of a training pipeline run."""
    model_name: str
    new_version: str
    training_samples: int
    metrics: Dict[str, Any]
    promoted: bool = False
    duration_seconds: float = 0.0
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ── Synthetic Data Generation ────────────────────────────────────

def generate_synthetic_corpus(n: int = 500, seed: int = 42) -> List[str]:
    """Generate synthetic student text data for model fitting.

    Creates realistic journal entries and check-in texts without
    requiring labels.  This is for bootstrapping models when
    there's not enough real data yet.
    """
    import random
    rng = random.Random(seed)

    templates = [
        # Positive
        "I feel great today, everything is going well",
        "Had an amazing time with my friends after school",
        "I'm really excited about the upcoming project",
        "Feeling grateful for my family and teachers",
        "Today was a wonderful day, I learned so much",
        "I'm proud of my test results, worked hard for this",
        "Laughed a lot today, my classmates are fun",
        "Feeling confident about the future",
        "Made a new friend today, really happy about it",
        "Class was interesting, enjoyed the group project",

        # Negative — sadness
        "I feel really down today and don't know why",
        "Nobody seems to understand how I feel",
        "I miss my old friends since we moved schools",
        "Everything feels pointless sometimes",
        "I cried today and couldn't stop the tears",
        "Feeling lonely even when surrounded by people",
        "I don't enjoy things I used to anymore",
        "Today was really hard, I just want it to be over",

        # Negative — anxiety
        "I'm really worried about the exam tomorrow",
        "Can't stop thinking about what might go wrong",
        "My heart races whenever I have to present in class",
        "I feel nervous all the time and can't relax",
        "What if I fail and disappoint everyone",
        "I keep having trouble sleeping because of worry",
        "Social situations make me really uncomfortable",
        "I feel overwhelmed by everything I need to do",

        # Negative — anger
        "I'm so frustrated with how unfair things are",
        "Someone said something really hurtful today",
        "I can't stand when people don't listen to me",
        "Everything is making me irritated lately",
        "I got into an argument and I'm still upset",

        # Neutral
        "Today was an ordinary day at school",
        "Had lunch and then went to class as usual",
        "The weather was nice today nothing special",
        "Studied for a bit and then watched some TV",
        "Nothing much happened today just another day",
        "Went through my normal routine at school",
        "Classes were okay nothing special happened",
    ]

    modifiers = [
        "", " I think.", " Not sure what to do.", " Just wanted to share.",
        " It's been like this for a while.", " Hope tomorrow is different.",
        " Talking about it helps.", " Just need to get through this week.",
    ]

    texts = []
    for _ in range(n):
        text = rng.choice(templates)
        text += rng.choice(modifiers)
        texts.append(text)

    return texts


def generate_synthetic_features(
    n: int = 100, seed: int = 42
) -> List[BehavioralFeatureVector]:
    """Generate synthetic behavioural feature vectors for model fitting."""
    import random
    rng = random.Random(seed)

    features = []
    for i in range(n):
        # Create realistic distributions
        # Most students are "normal", some are at-risk
        is_at_risk = rng.random() < 0.15

        if is_at_risk:
            fv = BehavioralFeatureVector(
                student_id=f"synthetic_{i}",
                avg_sentiment=rng.uniform(-0.8, -0.1),
                sentiment_std=rng.uniform(0.3, 0.8),
                negative_ratio=rng.uniform(0.4, 0.9),
                dominant_cluster=rng.randint(0, 2),
                emotion_entropy=rng.uniform(0.5, 2.0),
                distress_ratio=rng.uniform(0.3, 0.8),
                entries_per_week=rng.uniform(0.5, 3.0),
                journal_ratio=rng.uniform(0.3, 0.8),
                days_since_last_entry=rng.uniform(0, 14),
                avg_mood=rng.uniform(1.0, 2.5),
                mood_std=rng.uniform(0.8, 2.0),
                high_risk_flags=rng.randint(0, 3),
            )
        else:
            fv = BehavioralFeatureVector(
                student_id=f"synthetic_{i}",
                avg_sentiment=rng.uniform(0.0, 0.6),
                sentiment_std=rng.uniform(0.1, 0.3),
                negative_ratio=rng.uniform(0.05, 0.25),
                dominant_cluster=rng.randint(0, 4),
                emotion_entropy=rng.uniform(1.0, 2.5),
                distress_ratio=rng.uniform(0.0, 0.15),
                entries_per_week=rng.uniform(2.0, 7.0),
                journal_ratio=rng.uniform(0.2, 0.6),
                days_since_last_entry=rng.uniform(0, 3),
                avg_mood=rng.uniform(3.0, 5.0),
                mood_std=rng.uniform(0.2, 0.7),
                high_risk_flags=0,
            )

        features.append(fv)

    return features


# ── Training Pipelines ──────────────────────────────────────────


async def train_text_embeddings(
    corpus: Optional[List[str]] = None,
    version: str = "3.0.0",
) -> TrainingResult:
    """Train the TF-IDF text embedding engine."""
    start = time.time()

    if corpus is None:
        corpus = generate_synthetic_corpus(n=500)

    engine = get_embedding_engine()
    engine.fit(corpus)

    duration = time.time() - start

    # Register in model registry
    model = ModelVersion(
        model_name="text_embeddings",
        version=version,
        status="active",
        config=engine.get_config(),
        metrics={"vocabulary_size": engine.vocabulary_size},
        description=f"TF-IDF text embedding engine v{version}",
        activated_at=datetime.now(timezone.utc),
    )
    registry.register(model)

    logger.info(
        f"Text embeddings trained: v{version}, "
        f"vocab={engine.vocabulary_size}, "
        f"corpus={len(corpus)}, duration={duration:.2f}s"
    )

    return TrainingResult(
        model_name="text_embeddings",
        new_version=version,
        training_samples=len(corpus),
        metrics={"vocabulary_size": engine.vocabulary_size},
        promoted=True,
        duration_seconds=round(duration, 2),
    )


async def train_emotion_clusters(
    corpus: Optional[List[str]] = None,
    version: str = "3.0.0",
    n_clusters: int = 5,
) -> TrainingResult:
    """Train the K-Means emotion clustering engine."""
    start = time.time()

    if corpus is None:
        corpus = generate_synthetic_corpus(n=500)

    # Ensure embeddings are fitted
    engine = get_embedding_engine()
    if not engine.is_fitted:
        engine.fit(corpus)

    # Fit emotion clusters
    emotion_engine = get_emotion_engine()
    emotion_engine.n_clusters = n_clusters
    emotion_engine.fit(corpus)

    # Evaluate: compute silhouette on the clustered data
    X = engine.transform(corpus)
    labels = [emotion_engine.predict(t).predicted_cluster for t in corpus]

    metrics_dict: Dict[str, Any] = {
        "n_clusters": len(emotion_engine.clusters),
    }

    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels, metric="cosine")
        ch = calinski_harabasz_score(X, labels)
        metrics_dict["silhouette_score"] = round(sil, 4)
        metrics_dict["calinski_harabasz_score"] = round(ch, 4)

    # Cluster sizes
    cluster_sizes = Counter(labels)
    metrics_dict["cluster_sizes"] = dict(cluster_sizes)

    duration = time.time() - start

    # Register
    model = ModelVersion(
        model_name="emotion_detection",
        version=version,
        status="active",
        config=emotion_engine.get_config(),
        metrics=metrics_dict,
        description=f"K-Means emotion clustering v{version}",
        activated_at=datetime.now(timezone.utc),
    )
    registry.register(model)

    logger.info(
        f"Emotion clustering trained: v{version}, "
        f"k={len(emotion_engine.clusters)}, "
        f"silhouette={metrics_dict.get('silhouette_score', 'N/A')}"
    )

    return TrainingResult(
        model_name="emotion_detection",
        new_version=version,
        training_samples=len(corpus),
        metrics=metrics_dict,
        promoted=True,
        duration_seconds=round(duration, 2),
    )


async def train_anomaly_detection(
    features: Optional[List[BehavioralFeatureVector]] = None,
    version: str = "1.0.0",
    contamination: float = 0.1,
) -> TrainingResult:
    """Train the Isolation Forest anomaly detection engine."""
    start = time.time()

    if features is None:
        features = generate_synthetic_features(n=100)

    anomaly_engine = get_anomaly_engine()
    anomaly_engine.contamination = contamination
    anomaly_engine.fit(features)

    # Evaluate
    report = anomaly_engine.predict_batch(features)

    metrics_dict = {
        "total_students": report.total_students,
        "anomalies_detected": report.anomalies_detected,
        "anomaly_rate": report.anomaly_rate,
        "contamination": contamination,
    }

    duration = time.time() - start

    model = ModelVersion(
        model_name="anomaly_detection",
        version=version,
        status="active",
        config=anomaly_engine.get_config(),
        metrics=metrics_dict,
        description=f"Isolation Forest anomaly detection v{version}",
        activated_at=datetime.now(timezone.utc),
    )
    registry.register(model)

    logger.info(
        f"Anomaly detection trained: v{version}, "
        f"samples={len(features)}, "
        f"anomaly_rate={report.anomaly_rate:.4f}"
    )

    return TrainingResult(
        model_name="anomaly_detection",
        new_version=version,
        training_samples=len(features),
        metrics=metrics_dict,
        promoted=True,
        duration_seconds=round(duration, 2),
    )


async def train_student_clustering(
    features: Optional[List[BehavioralFeatureVector]] = None,
    version: str = "1.0.0",
) -> TrainingResult:
    """Train the GMM student clustering engine."""
    start = time.time()

    if features is None:
        features = generate_synthetic_features(n=100)

    cluster_engine = get_cluster_engine()
    cluster_engine.fit(features)

    report = cluster_engine.generate_report(features)

    metrics_dict = {
        "n_clusters": report.n_clusters,
        "silhouette_score": report.silhouette_score,
        "calinski_harabasz_score": report.calinski_harabasz_score,
        "total_students": report.total_students,
    }

    duration = time.time() - start

    model = ModelVersion(
        model_name="student_clustering",
        version=version,
        status="active",
        config=cluster_engine.get_config(),
        metrics=metrics_dict,
        description=f"GMM student clustering v{version}",
        activated_at=datetime.now(timezone.utc),
    )
    registry.register(model)

    logger.info(
        f"Student clustering trained: v{version}, "
        f"k={report.n_clusters}, "
        f"silhouette={report.silhouette_score:.4f}"
    )

    return TrainingResult(
        model_name="student_clustering",
        new_version=version,
        training_samples=len(features),
        metrics=metrics_dict,
        promoted=True,
        duration_seconds=round(duration, 2),
    )


async def train_topic_discovery(
    corpus: Optional[List[str]] = None,
    version: str = "1.0.0",
    n_topics: int = 8,
) -> TrainingResult:
    """Train the NMF topic discovery engine."""
    start = time.time()

    if corpus is None:
        corpus = generate_synthetic_corpus(n=500)

    topic_engine = get_topic_engine()
    topic_engine.n_topics = n_topics
    topic_engine.fit(corpus)

    report = topic_engine.get_report(total_documents=len(corpus))

    metrics_dict = {
        "n_topics": report.n_topics,
        "reconstruction_error": report.reconstruction_error,
        "total_documents": report.total_documents,
        "topic_labels": [t.label for t in report.topics],
    }

    duration = time.time() - start

    model = ModelVersion(
        model_name="topic_discovery",
        version=version,
        status="active",
        config=topic_engine.get_config(),
        metrics=metrics_dict,
        description=f"NMF topic discovery v{version}",
        activated_at=datetime.now(timezone.utc),
    )
    registry.register(model)

    logger.info(
        f"Topic discovery trained: v{version}, "
        f"topics={report.n_topics}, "
        f"recon_error={report.reconstruction_error:.4f}"
    )

    return TrainingResult(
        model_name="topic_discovery",
        new_version=version,
        training_samples=len(corpus),
        metrics=metrics_dict,
        promoted=True,
        duration_seconds=round(duration, 2),
    )


# ── Full Pipeline ────────────────────────────────────────────────


async def run_full_training_pipeline(
    corpus: Optional[List[str]] = None,
    features: Optional[List[BehavioralFeatureVector]] = None,
) -> List[TrainingResult]:
    """Run the complete unsupervised training pipeline.

    Fits all models sequentially in the correct dependency order:
      1. Text embeddings (foundation)
      2. Emotion clusters (depends on embeddings)
      3. Topic discovery (depends on embeddings)
      4. Anomaly detection (depends on feature vectors)
      5. Student clustering (depends on feature vectors)

    Args:
        corpus: Optional text corpus. Uses synthetic if not provided.
        features: Optional feature vectors. Uses synthetic if not provided.

    Returns:
        List of TrainingResult for each model.
    """
    logger.info("=== Starting full unsupervised training pipeline ===")
    start = time.time()
    results = []

    # Generate data if not provided
    if corpus is None:
        corpus = generate_synthetic_corpus(n=500)
    if features is None:
        features = generate_synthetic_features(n=100)

    # 1. Text embeddings
    result = await train_text_embeddings(corpus=corpus, version="3.0.0")
    results.append(result)

    # 2. Emotion clusters
    result = await train_emotion_clusters(corpus=corpus, version="3.0.0")
    results.append(result)

    # 3. Topic discovery
    result = await train_topic_discovery(corpus=corpus, version="1.0.0")
    results.append(result)

    # 4. Anomaly detection
    result = await train_anomaly_detection(features=features, version="1.0.0")
    results.append(result)

    # 5. Student clustering
    result = await train_student_clustering(features=features, version="1.0.0")
    results.append(result)

    total_time = time.time() - start
    logger.info(
        f"=== Training pipeline complete: "
        f"{len(results)} models trained in {total_time:.2f}s ==="
    )

    return results
