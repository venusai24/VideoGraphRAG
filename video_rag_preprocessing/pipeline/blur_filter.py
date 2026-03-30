"""
Blur filter – discards blurry frames using Variance of Laplacian.

The Laplacian highlights edges; its variance is high for sharp images
and low for blurry ones.  We convert to grayscale *once* and cache the
score for downstream use (frame selection).
"""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from video_rag_preprocessing.config import PipelineConfig
from video_rag_preprocessing.utils.image_utils import to_grayscale

logger = logging.getLogger(__name__)


class ScoredFrame(NamedTuple):
    """A frame annotated with its blur score and timestamp."""

    frame: np.ndarray       # BGR image
    timestamp: float        # seconds
    blur_score: float       # Laplacian variance


def compute_blur_score(frame: np.ndarray) -> float:
    """Return the Variance of Laplacian for a BGR frame."""
    gray = to_grayscale(frame)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_blurry_frames(
    frames: List[Tuple[np.ndarray, float]],
    config: PipelineConfig,
) -> List[ScoredFrame]:
    """Remove blurry frames below ``config.blur_threshold``.

    Parameters
    ----------
    frames : list[tuple[np.ndarray, float]]
        ``(frame_bgr, timestamp)`` pairs from the sampler.
    config : PipelineConfig
        Pipeline configuration (``blur_threshold``).

    Returns
    -------
    list[ScoredFrame]
        Non‑blurry frames with their blur scores attached.
    """
    kept: List[ScoredFrame] = []
    discarded = 0

    for frame, ts in tqdm(frames, desc="Blur filtering", unit="frame"):
        score = compute_blur_score(frame)
        if score >= config.blur_threshold:
            kept.append(ScoredFrame(frame=frame, timestamp=ts, blur_score=score))
        else:
            discarded += 1

    logger.info(
        "Blur filter: kept %d, discarded %d (threshold=%.1f).",
        len(kept),
        discarded,
        config.blur_threshold,
    )
    return kept
