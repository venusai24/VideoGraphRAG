"""
Frame sampler – extracts frames at a uniform target FPS from any video.

Timestamps are computed analytically from the source FPS so that the
sampling is deterministic and frame‑skip errors are avoided.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from video_rag_preprocessing.config import PipelineConfig

logger = logging.getLogger(__name__)

FrameWithTimestamp = Tuple[np.ndarray, float]


def sample_frames(
    video_path: str | Path,
    config: PipelineConfig,
) -> List[FrameWithTimestamp]:
    """Sample frames from *video_path* at ``config.fps_sample`` FPS.

    Parameters
    ----------
    video_path : str | Path
        Path to the input video file.
    config : PipelineConfig
        Pipeline configuration (only ``fps_sample`` is used).

    Returns
    -------
    list[tuple[np.ndarray, float]]
        Each element is ``(frame_bgr, timestamp_seconds)``.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If the video cannot be opened by OpenCV.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        source_fps: float = cap.get(cv2.CAP_PROP_FPS)
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps <= 0:
            raise RuntimeError(
                f"Invalid source FPS ({source_fps}) reported by OpenCV."
            )

        duration_sec = total_frames / source_fps
        target_fps = config.fps_sample

        # Compute the exact timestamps we want to sample
        sample_interval = 1.0 / target_fps
        timestamps = np.arange(0.0, duration_sec, sample_interval)

        logger.info(
            "Video: %.2f s @ %.1f FPS (%d frames). "
            "Sampling %d frames @ %.1f FPS.",
            duration_sec,
            source_fps,
            total_frames,
            len(timestamps),
            target_fps,
        )

        frames: List[FrameWithTimestamp] = []

        for ts in tqdm(timestamps, desc="Sampling frames", unit="frame"):
            # Seek to the target timestamp (milliseconds)
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.debug("Could not read frame at %.3f s – skipping.", ts)
                continue
            frames.append((frame, float(ts)))

        logger.info("Sampled %d frames from video.", len(frames))
        return frames

    finally:
        cap.release()
