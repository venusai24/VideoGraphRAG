"""
Image utility functions shared across pipeline stages.
"""

import cv2
import numpy as np


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to single‑channel grayscale."""
    if len(frame.shape) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame (OpenCV default) to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_for_model(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize a frame to ``(size, size)`` for model input.

    Uses ``INTER_AREA`` when downscaling and ``INTER_LANCZOS4`` when
    upscaling for the best visual quality.
    """
    h, w = frame.shape[:2]
    interp = cv2.INTER_AREA if (h > size and w > size) else cv2.INTER_LANCZOS4
    return cv2.resize(frame, (size, size), interpolation=interp)
