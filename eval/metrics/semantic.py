"""
Semantic Similarity Metrics
===========================
Computes semantic similarity between generated answers and gold answers
using sentence embeddings.

Uses sentence-transformers for efficient embedding computation.
Falls back to token overlap if sentence-transformers is not available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model for semantic similarity
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache for model instance
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer | None:
    """
    Get or create a cached SentenceTransformer model.

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        SentenceTransformer model instance or None if not available
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading sentence-transformers model: {model_name}")
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


class SemanticSimilarity:
    """
    Computes semantic similarity between texts using sentence embeddings.

    Uses sentence-transformers for embedding computation with cosine similarity.
    Falls back to simple token overlap if sentence-transformers is unavailable.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize SemanticSimilarity calculator.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model = _get_model(model_name)
        self._available = self._model is not None

    @property
    def is_available(self) -> bool:
        """Check if semantic similarity is available."""
        return self._available

    def compute(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text (e.g., generated answer)
            text2: Second text (e.g., gold answer)

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0

        if not self._available:
            # Fallback to token overlap
            return self._token_overlap_similarity(text1, text2)

        try:
            # Encode both texts
            embeddings = self._model.encode([text1, text2], convert_to_numpy=True)
            return cosine_similarity(embeddings[0], embeddings[1])
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return self._token_overlap_similarity(text1, text2)

    def compute_batch(self, texts1: list[str], texts2: list[str]) -> list[float]:
        """
        Compute semantic similarity for multiple text pairs.

        Args:
            texts1: List of first texts
            texts2: List of second texts

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("Input lists must have the same length")

        if not self._available:
            return [
                self._token_overlap_similarity(t1, t2)
                for t1, t2 in zip(texts1, texts2, strict=False)
            ]

        try:
            # Encode all texts at once for efficiency
            all_texts = texts1 + texts2
            all_embeddings = self._model.encode(all_texts, convert_to_numpy=True)

            n = len(texts1)
            embeddings1 = all_embeddings[:n]
            embeddings2 = all_embeddings[n:]

            return [
                cosine_similarity(e1, e2) for e1, e2 in zip(embeddings1, embeddings2, strict=False)
            ]
        except Exception as e:
            logger.error(f"Error in batch semantic similarity: {e}")
            return [
                self._token_overlap_similarity(t1, t2)
                for t1, t2 in zip(texts1, texts2, strict=False)
            ]

    @staticmethod
    def _token_overlap_similarity(text1: str, text2: str) -> float:
        """
        Fallback: Compute Jaccard similarity based on token overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0

        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)


# Convenience function
def semantic_similarity(
    text1: str,
    text2: str,
    model_name: str = DEFAULT_MODEL,
) -> float:
    """
    Compute semantic similarity between two texts.

    Convenience function that creates a SemanticSimilarity instance.

    Args:
        text1: First text (e.g., generated answer)
        text2: Second text (e.g., gold answer)
        model_name: sentence-transformers model name

    Returns:
        Similarity score (0.0 to 1.0)
    """
    calculator = SemanticSimilarity(model_name)
    return calculator.compute(text1, text2)


def answer_semantic_similarity(
    answer: str,
    gold_answer: str | None,
    model_name: str = DEFAULT_MODEL,
) -> float | None:
    """
    Compute semantic similarity between generated and gold answers.

    Args:
        answer: Generated answer
        gold_answer: Gold standard answer (optional)
        model_name: sentence-transformers model name

    Returns:
        Similarity score (0.0 to 1.0) or None if no gold answer
    """
    if not gold_answer:
        return None

    return semantic_similarity(answer, gold_answer, model_name)
