"""
Frame selector – picks the single best representative from each group.

Scoring formula
---------------
score = w_blur × normalised_blur + w_centrality × centrality

Where:
    • normalised_blur  = blur_score / max_blur_in_group (so it ∈ [0, 1])
    • centrality       = mean cosine similarity to every other frame in the
                         group (already ∈ [0, 1] for normalised embeddings)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from video_rag_preprocessing.config import PipelineConfig
from video_rag_preprocessing.utils.similarity import cosine_similarity_matrix

logger = logging.getLogger(__name__)


def select_best_frames(
    groups: List[List[Dict[str, Any]]],
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    """Select one best frame per group.

    Parameters
    ----------
    groups : list[list[dict]]
        Each inner list is a group of frame records containing at minimum
        ``"blur_score"`` (float) and ``"embedding_dino"`` (np.ndarray).
    config : PipelineConfig
        Uses ``w_blur`` and ``w_centrality``.

    Returns
    -------
    list[dict]
        One record per group — the highest‑scoring frame.
    """
    w_blur = config.w_blur
    w_cent = config.w_centrality
    selected: List[Dict[str, Any]] = []

    for group in groups:
        if len(group) == 1:
            selected.append(group[0])
            continue

        # ── Blur scores (normalised to [0, 1]) ────────────────────
        blur_scores = np.array([r["blur_score"] for r in group], dtype=np.float64)
        max_blur = blur_scores.max()
        if max_blur > 0:
            norm_blur = blur_scores / max_blur
        else:
            norm_blur = np.zeros_like(blur_scores)

        # ── Centrality (mean cosine sim to all others) ─────────────
        embs = np.stack([r["embedding_dino"] for r in group])
        sim_matrix = cosine_similarity_matrix(embs)  # (n, n)
        # Exclude self‑similarity on the diagonal
        n = len(group)
        np.fill_diagonal(sim_matrix, 0.0)
        centrality = sim_matrix.sum(axis=1) / max(n - 1, 1)

        # ── Combined score ─────────────────────────────────────────
        scores = w_blur * norm_blur + w_cent * centrality
        best_idx = int(np.argmax(scores))
        selected.append(group[best_idx])

    logger.info(
        "Selection: %d keyframes chosen from %d groups.", len(selected), len(groups)
    )
    return selected
