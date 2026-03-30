"""
Lightweight scene boundary detector.

Splits a sequence of frames into *scenes* by measuring cosine similarity
of consecutive DINOv2 embeddings.  A drop below the configured threshold
signals a new scene.

No heavy external libraries are used — only NumPy and our own similarity
utilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from video_rag_preprocessing.config import PipelineConfig
from video_rag_preprocessing.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


def detect_scenes(
    frame_records: List[Dict[str, Any]],
    config: PipelineConfig,
) -> List[List[Dict[str, Any]]]:
    """Segment *frame_records* into scenes using embedding similarity.

    Parameters
    ----------
    frame_records : list[dict]
        Each dict must contain at least ``"embedding_dino"`` (np.ndarray).
        Other keys are passed through untouched.
    config : PipelineConfig
        Uses ``scene_sim_threshold``.

    Returns
    -------
    list[list[dict]]
        Outer list = scenes; inner list = ordered frames in that scene.
    """
    if not frame_records:
        return []

    threshold = config.scene_sim_threshold
    scenes: List[List[Dict[str, Any]]] = []
    current_scene: List[Dict[str, Any]] = [frame_records[0]]

    for i in range(1, len(frame_records)):
        prev_emb: np.ndarray = frame_records[i - 1]["embedding_dino"]
        curr_emb: np.ndarray = frame_records[i]["embedding_dino"]

        sim = cosine_similarity(prev_emb, curr_emb)

        if sim < threshold:
            # Scene boundary detected
            scenes.append(current_scene)
            current_scene = [frame_records[i]]
        else:
            current_scene.append(frame_records[i])

    # Don't forget the last scene
    if current_scene:
        scenes.append(current_scene)

    logger.info(
        "Scene detection: %d scenes from %d frames (threshold=%.2f).",
        len(scenes),
        len(frame_records),
        threshold,
    )
    return scenes
