"""
Temporal‑aware similarity grouping.

Groups frames within a scene so that each group contains visually
similar, temporally contiguous content.  This avoids O(n²) all‑pairs
comparisons by maintaining running references to the *first* frame and
*last* frame of the current group.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from video_rag_preprocessing.config import PipelineConfig
from video_rag_preprocessing.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


def group_frames(
    scene_frames: List[Dict[str, Any]],
    config: PipelineConfig,
) -> List[List[Dict[str, Any]]]:
    """Group *scene_frames* by temporal‑aware embedding similarity.

    Algorithm
    ---------
    1.  Start a new group with the first frame.
    2.  For each subsequent frame:
        - If ``cos(frame, last_in_group) ≥ t1`` **and**
          ``cos(frame, first_in_group) ≥ t2`` → add to group.
        - Otherwise → finalise the current group and start a new one.

    Parameters
    ----------
    scene_frames : list[dict]
        Must contain ``"embedding_dino"`` (np.ndarray) per frame.
    config : PipelineConfig
        Uses ``sim_t1`` (local continuity) and ``sim_t2`` (anchor).

    Returns
    -------
    list[list[dict]]
        Groups of semantically similar, contiguous frames.
    """
    if not scene_frames:
        return []

    t1 = config.sim_t1
    t2 = config.sim_t2

    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = [scene_frames[0]]

    for rec in scene_frames[1:]:
        emb = rec["embedding_dino"]
        first_emb: np.ndarray = current_group[0]["embedding_dino"]
        last_emb: np.ndarray = current_group[-1]["embedding_dino"]

        sim_local = cosine_similarity(emb, last_emb)
        sim_anchor = cosine_similarity(emb, first_emb)

        if sim_local >= t1 and sim_anchor >= t2:
            current_group.append(rec)
        else:
            groups.append(current_group)
            current_group = [rec]

    if current_group:
        groups.append(current_group)

    logger.info(
        "Grouping: %d groups from %d frames (t1=%.2f, t2=%.2f).",
        len(groups),
        len(scene_frames),
        t1,
        t2,
    )
    return groups
