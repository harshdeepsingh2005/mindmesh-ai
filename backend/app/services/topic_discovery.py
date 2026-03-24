"""MindMesh AI — Topic Discovery Service (Unsupervised).

Discovers latent themes/topics in student text data using
Non-negative Matrix Factorization (NMF) on TF-IDF matrices.

NMF decomposes the document-term matrix into:
  W (document-topic) × H (topic-term)

This reveals what students are writing ABOUT — not just how they
feel — enabling counselors to understand common concerns:
  • academic pressure
  • social conflicts
  • family issues
  • self-image concerns
  • future anxiety

All topics are discovered automatically from the data — no
pre-defined categories are needed.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.decomposition import NMF

from ..config import settings
from ..logging_config import logger
from .text_embeddings import get_embedding_engine, preprocess_text


# ── Constants ────────────────────────────────────────────────────

DEFAULT_N_TOPICS = 8
TOP_WORDS_PER_TOPIC = 10
MIN_DOCUMENTS_FOR_NMF = 15


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class Topic:
    """A discovered topic with its top terms."""
    topic_id: int
    top_terms: List[str]
    term_weights: List[float]
    label: str               # auto-generated descriptive label
    document_count: int       # how many documents are primarily about this topic
    prevalence: float         # proportion of corpus about this topic


@dataclass
class TopicAssignment:
    """Topic analysis result for a single document."""
    dominant_topic: int
    topic_label: str
    topic_distribution: Dict[int, float]  # topic_id → weight
    confidence: float


@dataclass
class TopicModelReport:
    """Full report from topic discovery."""
    n_topics: int
    topics: List[Topic]
    reconstruction_error: float
    total_documents: int
    model_version: str = "topics-v1.0.0-nmf"


# ── Topic Discovery Engine ──────────────────────────────────────


class TopicDiscoveryEngine:
    """NMF-based topic discovery for student text corpora.

    Phases:
      1. fit():      Discover topics from a corpus
      2. predict():  Assign topic distribution to new text
    """

    def __init__(self, n_topics: int = DEFAULT_N_TOPICS) -> None:
        self.n_topics = n_topics
        self._nmf: Optional[NMF] = None
        self._topics: List[Topic] = []
        self._is_fitted = False
        self._reconstruction_error = 0.0
        self._version = "topics-v1.0.0-nmf"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def topics(self) -> List[Topic]:
        return self._topics

    def fit(self, texts: List[str]) -> "TopicDiscoveryEngine":
        """Discover topics from a corpus of student texts.

        Args:
            texts: Raw text documents.

        Returns:
            Self (for chaining).
        """
        if len(texts) < MIN_DOCUMENTS_FOR_NMF:
            logger.warning(
                f"Not enough documents for topic discovery "
                f"({len(texts)} < {MIN_DOCUMENTS_FOR_NMF})"
            )
            self._setup_fallback()
            return self

        engine = get_embedding_engine()
        if not engine.is_fitted:
            engine.fit(texts)

        # NMF requires non-negative input — use raw TF-IDF (not L2-normalised)
        from sklearn.feature_extraction.text import TfidfVectorizer

        cleaned = [preprocess_text(t) or "empty" for t in texts]

        # Fit a separate vectoriser for NMF (no L2 normalisation)
        self._tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,
        )
        X = self._tfidf.fit_transform(cleaned)

        # Fit NMF
        actual_topics = min(self.n_topics, len(texts) - 1, X.shape[1])
        actual_topics = max(2, actual_topics)

        self._nmf = NMF(
            n_components=actual_topics,
            init="nndsvd",
            random_state=42,
            max_iter=500,
        )
        W = self._nmf.fit_transform(X)  # document-topic matrix
        self._reconstruction_error = float(self._nmf.reconstruction_err_)

        # Extract topics
        feature_names = self._tfidf.get_feature_names_out()
        self._topics = []

        for topic_idx, topic_vec in enumerate(self._nmf.components_):
            top_indices = topic_vec.argsort()[-TOP_WORDS_PER_TOPIC:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_weights = [float(topic_vec[i]) for i in top_indices]

            # Count documents primarily about this topic
            dominant_docs = int((W.argmax(axis=1) == topic_idx).sum())

            # Auto-label from top 3 terms
            label = "_".join(top_terms[:3])

            self._topics.append(Topic(
                topic_id=topic_idx,
                top_terms=top_terms,
                term_weights=[round(w, 4) for w in top_weights],
                label=label,
                document_count=dominant_docs,
                prevalence=round(dominant_docs / len(texts), 4),
            ))

        self._is_fitted = True

        logger.info(
            f"TopicDiscoveryEngine fitted: n_topics={actual_topics}, "
            f"documents={len(texts)}, "
            f"reconstruction_error={self._reconstruction_error:.4f}"
        )
        self.save()

        return self

    def predict(self, text: str) -> TopicAssignment:
        """Assign topic distribution to a single text.

        Args:
            text: Raw text input.

        Returns:
            TopicAssignment with dominant topic and distribution.
        """
        if not self._is_fitted or self._nmf is None:
            return TopicAssignment(
                dominant_topic=0,
                topic_label="unknown",
                topic_distribution={},
                confidence=0.0,
            )

        cleaned = preprocess_text(text) or "empty"
        X = self._tfidf.transform([cleaned])
        W = self._nmf.transform(X)[0]

        # Normalise to probability distribution
        total = W.sum()
        if total > 0:
            distribution = {i: round(float(w / total), 4) for i, w in enumerate(W)}
        else:
            distribution = {i: round(1.0 / len(W), 4) for i in range(len(W))}

        dominant = int(np.argmax(W))
        confidence = distribution.get(dominant, 0.0)

        topic_label = (
            self._topics[dominant].label
            if dominant < len(self._topics)
            else f"topic_{dominant}"
        )

        return TopicAssignment(
            dominant_topic=dominant,
            topic_label=topic_label,
            topic_distribution=distribution,
            confidence=confidence,
        )

    def get_report(self, total_documents: int = 0) -> TopicModelReport:
        """Generate a full report of discovered topics."""
        return TopicModelReport(
            n_topics=len(self._topics),
            topics=self._topics,
            reconstruction_error=self._reconstruction_error,
            total_documents=total_documents,
            model_version=self._version,
        )

    def _setup_fallback(self) -> None:
        """Minimal fallback when corpus is too small."""
        self._is_fitted = True
        self._topics = [
            Topic(
                topic_id=0,
                top_terms=["school", "day", "today"],
                term_weights=[1.0, 0.8, 0.7],
                label="general_school_day",
                document_count=0,
                prevalence=1.0,
            )
        ]
        self._version = "topics-v1.0.0-fallback"

    def get_config(self) -> Dict:
        """Return engine config for registry."""
        return {
            "type": "nmf_topic_model",
            "n_topics": self.n_topics,
            "n_topics_actual": len(self._topics),
            "reconstruction_error": self._reconstruction_error,
            "is_fitted": self._is_fitted,
            "version": self._version,
        }

    def save(self) -> None:
        """Save the fitted model to disk."""
        if not self._is_fitted:
            return
        os.makedirs(settings.MODEL_SAVE_DIR, exist_ok=True)
        path = os.path.join(settings.MODEL_SAVE_DIR, "topic_discovery.joblib")
        joblib.dump(self, path)
        logger.info(f"TopicDiscoveryEngine saved to {path}")

    @classmethod
    def load(cls) -> Optional["TopicDiscoveryEngine"]:
        """Load the fitted model from disk."""
        path = os.path.join(settings.MODEL_SAVE_DIR, "topic_discovery.joblib")
        if os.path.exists(path):
            try:
                engine = joblib.load(path)
                logger.info(f"TopicDiscoveryEngine loaded from {path}")
                return engine
            except Exception as e:
                logger.error(f"Failed to load TopicDiscoveryEngine: {e}")
        return None


# ── Singleton ────────────────────────────────────────────────────

_engine: Optional[TopicDiscoveryEngine] = None


def get_topic_engine() -> TopicDiscoveryEngine:
    """Get or create the global TopicDiscoveryEngine."""
    global _engine
    if _engine is None:
        _engine = TopicDiscoveryEngine.load()
        if _engine is None:
            _engine = TopicDiscoveryEngine()
    return _engine
