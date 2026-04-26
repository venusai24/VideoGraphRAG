"""
Similarity utilities for embedding comparison.

Provides efficient cosine similarity computation used across the pipeline
(scene detection, grouping, selection).
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Both vectors are assumed to be 1‑D.  If either has zero norm the
    similarity is defined as 0.0 to avoid division‑by‑zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for a matrix of row‑vectors.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n, d)`` where *n* is the number of vectors.

    Returns
    -------
    np.ndarray
        Shape ``(n, n)`` symmetric similarity matrix.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)  # prevent div‑by‑zero
    normed = embeddings / norms
    return normed @ normed.T
