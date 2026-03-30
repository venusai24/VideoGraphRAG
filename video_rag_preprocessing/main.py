"""
Main entry‑point for the Video RAG preprocessing pipeline.

Usage
-----
    python -m video_rag_preprocessing.main --video_path input.mp4 --output_dir out/

The pipeline:
    video → sampling → blur filtering → embedding →
    scene detection → grouping → selection → storage
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from video_rag_preprocessing.config import PipelineConfig
from video_rag_preprocessing.pipeline.blur_filter import filter_blurry_frames
from video_rag_preprocessing.pipeline.embedding import EmbeddingExtractor
from video_rag_preprocessing.pipeline.grouping import group_frames
from video_rag_preprocessing.pipeline.sampler import sample_frames
from video_rag_preprocessing.pipeline.scene_detector import detect_scenes
from video_rag_preprocessing.pipeline.selector import select_best_frames
from video_rag_preprocessing.pipeline.storage import save_results

logger = logging.getLogger("video_rag_preprocessing")


# ──────────────────────────────────────────────────────────────────── #
#  CLI                                                                  #
# ──────────────────────────────────────────────────────────────────── #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video RAG Preprocessing Pipeline – extract high‑quality "
        "keyframes from a video for retrieval‑augmented generation.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory where keyframes and metadata will be saved (default: output/).",
    )

    # Optional overrides for key config parameters
    parser.add_argument("--fps", type=float, default=None, help="Sampling FPS (default: 12).")
    parser.add_argument("--blur_threshold", type=float, default=None, help="Blur threshold (default: 120).")
    parser.add_argument("--batch_size", type=int, default=None, help="Embedding batch size (default: 32).")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device (cuda/cpu, default: cuda).")
    parser.add_argument("--scene_threshold", type=float, default=None, help="Scene similarity threshold (default: 0.80).")
    parser.add_argument("--sim_t1", type=float, default=None, help="Grouping local threshold (default: 0.92).")
    parser.add_argument("--sim_t2", type=float, default=None, help="Grouping anchor threshold (default: 0.88).")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a ``PipelineConfig`` from CLI arguments, applying overrides."""
    config = PipelineConfig(output_dir=Path(args.output_dir))

    if args.fps is not None:
        config.fps_sample = args.fps
    if args.blur_threshold is not None:
        config.blur_threshold = args.blur_threshold
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.scene_threshold is not None:
        config.scene_sim_threshold = args.scene_threshold
    if args.sim_t1 is not None:
        config.sim_t1 = args.sim_t1
    if args.sim_t2 is not None:
        config.sim_t2 = args.sim_t2

    return config


# ──────────────────────────────────────────────────────────────────── #
#  Pipeline orchestration                                               #
# ──────────────────────────────────────────────────────────────────── #

def run_pipeline(video_path: str, config: PipelineConfig) -> Path:
    """Execute the full preprocessing pipeline and return the output dir.

    Stages
    ------
    1. Frame sampling
    2. Blur filtering
    3. Embedding extraction (CLIP + DINOv2)
    4. Scene detection
    5. Similarity grouping (per scene)
    6. Frame selection (per group)
    7. Storage
    """
    t0 = time.perf_counter()

    # ── 1. Sample frames ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 / 7 — Frame Sampling")
    logger.info("=" * 60)
    raw_frames = sample_frames(video_path, config)
    logger.info("  → Sampled frames: %d", len(raw_frames))

    if not raw_frames:
        logger.error("No frames were sampled – aborting.")
        sys.exit(1)

    # ── 2. Blur filtering ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 / 7 — Blur Filtering")
    logger.info("=" * 60)
    scored_frames = filter_blurry_frames(raw_frames, config)
    logger.info("  → Frames after blur filter: %d", len(scored_frames))

    # Free raw frame memory
    del raw_frames

    if not scored_frames:
        logger.error("All frames were filtered as blurry – aborting.")
        sys.exit(1)

    # ── 3. Embedding extraction ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 / 7 — Embedding Extraction")
    logger.info("=" * 60)
    extractor = EmbeddingExtractor(config)
    bgr_frames = [sf.frame for sf in scored_frames]
    embeddings = extractor.extract(bgr_frames)
    logger.info("  → Embeddings computed for %d frames.", len(embeddings))

    # Free model memory
    del extractor

    # ── Build unified frame records ────────────────────────────────
    frame_records: List[Dict[str, Any]] = []
    for sf, emb_dict in zip(scored_frames, embeddings):
        frame_records.append({
            "frame": sf.frame,
            "timestamp": sf.timestamp,
            "blur_score": sf.blur_score,
            "embedding_clip": emb_dict["embedding_clip"],
            "embedding_dino": emb_dict["embedding_dino"],
        })

    del scored_frames, embeddings

    # ── 4. Scene detection ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 / 7 — Scene Detection")
    logger.info("=" * 60)
    scenes = detect_scenes(frame_records, config)
    logger.info("  → Scenes detected: %d", len(scenes))

    del frame_records  # data is now inside `scenes`

    # ── 5. Similarity grouping (within each scene) ─────────────────
    logger.info("=" * 60)
    logger.info("STAGE 5 / 7 — Similarity Grouping")
    logger.info("=" * 60)
    all_groups: List[List[Dict[str, Any]]] = []
    for scene_id, scene in enumerate(
        tqdm(scenes, desc="Grouping per scene", unit="scene")
    ):
        groups = group_frames(scene, config)
        # Tag every frame with its scene id
        for grp in groups:
            for rec in grp:
                rec["scene_id"] = scene_id
        all_groups.extend(groups)

    total_groups = len(all_groups)
    logger.info("  → Total groups: %d", total_groups)

    del scenes

    # ── 6. Frame selection ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 6 / 7 — Frame Selection")
    logger.info("=" * 60)
    selected = select_best_frames(all_groups, config)
    logger.info("  → Selected keyframes: %d", len(selected))

    del all_groups

    # ── 7. Storage ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 7 / 7 — Storage")
    logger.info("=" * 60)
    output_path = save_results(selected, config)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f s.", elapsed)
    logger.info("Output: %s", output_path)
    logger.info("=" * 60)

    return output_path


# ──────────────────────────────────────────────────────────────────── #
#  Entry point                                                          #
# ──────────────────────────────────────────────────────────────────── #

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    config = build_config(args)

    logger.info("Video RAG Preprocessing Pipeline")
    logger.info("  video_path   : %s", args.video_path)
    logger.info("  output_dir   : %s", config.output_dir)
    logger.info("  fps_sample   : %.1f", config.fps_sample)
    logger.info("  blur_thresh  : %.1f", config.blur_threshold)
    logger.info("  scene_thresh : %.2f", config.scene_sim_threshold)
    logger.info("  sim_t1       : %.2f", config.sim_t1)
    logger.info("  sim_t2       : %.2f", config.sim_t2)
    logger.info("  batch_size   : %d", config.batch_size)
    logger.info("  device       : %s", config.device)

    run_pipeline(args.video_path, config)


if __name__ == "__main__":
    main()
