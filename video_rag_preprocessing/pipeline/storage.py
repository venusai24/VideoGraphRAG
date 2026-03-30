"""
Storage – persists selected keyframes and their metadata to disk.

Directory layout
----------------
<output_dir>/
    frames/
        frame_000000.jpg
        frame_000001.jpg
        ...
    metadata.json            # list of per‑frame dicts
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from video_rag_preprocessing.config import PipelineConfig

logger = logging.getLogger(__name__)


def save_results(
    selected_frames: List[Dict[str, Any]],
    config: PipelineConfig,
) -> Path:
    """Write keyframes to disk and save metadata as JSON.

    Parameters
    ----------
    selected_frames : list[dict]
        Each dict must contain:
        - ``"frame"``           : np.ndarray (BGR image)
        - ``"timestamp"``       : float
        - ``"scene_id"``        : int
        - ``"blur_score"``      : float
        - ``"embedding_clip"``  : np.ndarray
        - ``"embedding_dino"``  : np.ndarray
    config : PipelineConfig
        Uses ``output_dir``, ``save_images``, ``image_format``, ``image_quality``.

    Returns
    -------
    Path
        The output directory where everything was saved.
    """
    output_dir = config.output_dir
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    metadata_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(selected_frames):
        # ── Save image ─────────────────────────────────────────────
        filename = f"frame_{idx:06d}.{config.image_format}"
        image_path = frames_dir / filename

        if config.save_images:
            params: List[int] = []
            if config.image_format in ("jpg", "jpeg"):
                params = [cv2.IMWRITE_JPEG_QUALITY, config.image_quality]
            elif config.image_format == "png":
                params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

            cv2.imwrite(str(image_path), rec["frame"], params)

        # ── Build metadata record ──────────────────────────────────
        meta: Dict[str, Any] = {
            "frame_id": idx,
            "timestamp": round(rec["timestamp"], 4),
            "scene_id": rec["scene_id"],
            "blur_score": round(rec["blur_score"], 2),
            "embedding_clip": _to_list(rec["embedding_clip"]),
            "embedding_dino": _to_list(rec["embedding_dino"]),
            "image_path": str(image_path.relative_to(output_dir)),
        }
        metadata_records.append(meta)

    # ── Write metadata JSON ────────────────────────────────────────
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata_records, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved %d keyframes and metadata to %s.",
        len(selected_frames),
        output_dir,
    )
    return output_dir


# -------------------------------------------------------------------- #
#  Helpers                                                               #
# -------------------------------------------------------------------- #

def _to_list(arr: np.ndarray) -> List[float]:
    """Convert a NumPy array to a JSON‑serialisable list of floats."""
    return [round(float(v), 6) for v in arr]
