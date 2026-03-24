"""MindMesh AI — Text Embedding Service (Unsupervised).

Converts raw student text (journals, check-ins, survey responses)
into numerical feature vectors using TF-IDF.  These vectors are the
foundation for all downstream unsupervised tasks:

  • K-Means emotion clustering
  • NMF topic discovery
  • Isolation Forest anomaly detection
  • Student behavioral profiling

The module maintains a single fitted vectorizer that is retrained
whenever new data is ingested via `fit()` or `fit_transform()`.

NOTE: This intentionally avoids pre-trained transformer models
(BERT, sentence-transformers) to keep the project purely
unsupervised with no dependency on external labeled pre-training.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..config import settings
from ..logging_config import logger


# ── Text Preprocessing ───────────────────────────────────────────


def preprocess_text(text: Optional[str]) -> str:
    """Clean and normalise text for feature extraction.

    Steps:
        1. Lowercase
        2. Remove URLs, emails, special chars
        3. Collapse whitespace
        4. Strip leading/trailing whitespace

    Args:
        text: Raw text input.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)           # URLs
    text = re.sub(r"\S+@\S+", "", text)                     # emails
    text = re.sub(r"[^a-z\s']", " ", text)                  # keep letters + apostrophes
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace

    return text


# ── TF-IDF Embedding Engine ─────────────────────────────────────


class TextEmbeddingEngine:
    """TF-IDF based text vectoriser for unsupervised analysis.

    Fits a TF-IDF model on the corpus of student texts, then
    transforms individual texts into sparse → dense feature vectors.

    Configuration:
        max_features: Vocabulary size cap (default 5000).
        ngram_range:  Uni- and bigrams (1, 2) to capture phrases.
        min_df:       Minimum document frequency.
        max_df:       Maximum document frequency (filter stopwords).
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
            sublinear_tf=True,        # apply log(1 + tf) scaling
        )
        self._is_fitted = False
        self._vocabulary_size = 0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def fit(self, texts: List[str]) -> "TextEmbeddingEngine":
        """Fit the TF-IDF model on a corpus of texts.

        Args:
            texts: List of raw text documents.

        Returns:
            Self (for chaining).
        """
        cleaned = [preprocess_text(t) for t in texts]
        # Filter empty strings
        cleaned = [t if t else "empty" for t in cleaned]

        self._vectorizer.fit(cleaned)
        self._is_fitted = True
        self._vocabulary_size = len(self._vectorizer.vocabulary_)

        logger.info(
            f"TextEmbeddingEngine fitted: vocab_size={self._vocabulary_size}, "
            f"documents={len(cleaned)}"
        )
        self.save()
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into TF-IDF feature vectors.

        Args:
            texts: List of raw text documents.

        Returns:
            Dense numpy array of shape (n_documents, n_features).

        Raises:
            RuntimeError: If the engine has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TextEmbeddingEngine must be fitted before transform. "
                "Call fit() or fit_transform() first."
            )

        cleaned = [preprocess_text(t) for t in texts]
        cleaned = [t if t else "empty" for t in cleaned]

        sparse_matrix = self._vectorizer.transform(cleaned)
        dense = sparse_matrix.toarray()

        # L2 normalise so cosine similarity = dot product
        dense = normalize(dense, norm="l2")

        return dense

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            texts: List of raw text documents.

        Returns:
            Dense numpy array of shape (n_documents, n_features).
        """
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return the vocabulary terms from the fitted model."""
        if not self._is_fitted:
            return []
        return list(self._vectorizer.get_feature_names_out())

    def get_config(self) -> Dict:
        """Return the engine configuration for registry storage."""
        return {
            "type": "tfidf",
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "vocabulary_size": self._vocabulary_size,
            "is_fitted": self._is_fitted,
        }

    def save(self) -> None:
        """Save the fitted model to disk."""
        if not self._is_fitted:
            return
        os.makedirs(settings.MODEL_SAVE_DIR, exist_ok=True)
        path = os.path.join(settings.MODEL_SAVE_DIR, "text_embeddings.joblib")
        joblib.dump(self, path)
        logger.info(f"TextEmbeddingEngine saved to {path}")

    @classmethod
    def load(cls) -> Optional["TextEmbeddingEngine"]:
        """Load the fitted model from disk."""
        path = os.path.join(settings.MODEL_SAVE_DIR, "text_embeddings.joblib")
        if os.path.exists(path):
            try:
                engine = joblib.load(path)
                logger.info(f"TextEmbeddingEngine loaded from {path}")
                return engine
            except Exception as e:
                logger.error(f"Failed to load TextEmbeddingEngine: {e}")
        return None


# ── Singleton (lazy-initialised on first use) ────────────────────

_engine: Optional[TextEmbeddingEngine] = None


def get_embedding_engine() -> TextEmbeddingEngine:
    """Get or create the global TextEmbeddingEngine instance."""
    global _engine
    if _engine is None:
        _engine = TextEmbeddingEngine.load()
        if _engine is None:
            _engine = TextEmbeddingEngine()
    return _engine


def reset_embedding_engine() -> None:
    """Reset the global engine (used during retraining)."""
    global _engine
    _engine = TextEmbeddingEngine()
    logger.info("TextEmbeddingEngine reset")
