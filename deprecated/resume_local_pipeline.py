# Deprecated – not used in final pipeline
"""
resume_local_pipeline.py — Resume Local Pipeline
==================================================
For videos that have optimal_audio.wav extracted but the local pipeline
(run_pipeline.py) failed before producing reconstructed.mp4.

This script re-runs the local pipeline using the original video source,
then optionally triggers graph building.

Usage:
    # Resume a single video (auto-detects source from input/)
    python resume_local_pipeline.py outputs/39fbc2c8

    # Resume with explicit source video path
    python resume_local_pipeline.py outputs/39fbc2c8 --source input/The_International_Airport_That_Can_Only_Send_Flights_to_Turkey_enc.mp4

    # Resume and also rebuild graph afterwards
    python resume_local_pipeline.py outputs/39fbc2c8 --with-graph

    # Resume multiple videos
    python resume_local_pipeline.py outputs/39fbc2c8 outputs/ac1682fb --with-graph
"""

import os
import sys
import json
import hashlib
import argparse
import logging
import subprocess
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("resume_local.log", mode="a")])
logger = logging.getLogger("ResumeLocal")

PREPROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_rag_preprocessing")
sys.path.insert(0, PREPROC_DIR)


def video_id_from_filename(filename: str) -> str:
    """Reproduce the orchestrator's video_id hashing: MD5 of filename, first 8 chars."""
    return hashlib.md5(filename.encode()).hexdigest()[:8]


def find_source_video(video_id: str, search_dirs: list = None) -> str:
    """
    Find the original source video for a given video_id.
    1. Checks video_id_mapping.json (extracted from logs)
    2. Falls back to MD5 hash matching of filenames in search_dirs.
    """
    if search_dirs is None:
        search_dirs = ["input", "input/processed", "."]

    # 1. Try JSON mapping first (for UUID-based folders)
    mapping_path = Path("video_id_mapping.json")
    if mapping_path.exists():
        try:
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
                if video_id in mapping:
                    filename = mapping[video_id]
                    for search_dir in search_dirs:
                        potential_path = Path(search_dir) / filename
                        if potential_path.exists():
                            return str(potential_path)
        except Exception as e:
            logger.warning(f"Error reading video_id_mapping.json: {e}")

    # 2. Fallback to MD5 hashing (for newer deterministic folders)
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue

        for mp4 in search_path.glob("*.mp4"):
            computed_id = video_id_from_filename(mp4.name)
            if computed_id == video_id:
                return str(mp4)

    return None


def run_local_pipeline(video_path: str, output_dir: str, env: dict = None) -> bool:
    """Run the local pipeline (run_pipeline.py) as a subprocess."""
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)

    script = os.path.join(PREPROC_DIR, "run_pipeline.py")
    cmd = [
        sys.executable, script,
        "--video_path", video_path,
        "--output_dir", output_dir,
    ]

    logger.info(f"Running local pipeline: {Path(video_path).name} → {output_dir}")
    logger.info(f"Command: {' '.join(cmd)}")

    if env is None:
        env = os.environ.copy()

    result = subprocess.run(
        cmd,
        cwd=PREPROC_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    if result.returncode != 0:
        logger.error(f"Local pipeline failed with exit code {result.returncode}")
        return False

    # Verify output
    reconstructed = os.path.join(output_dir, "reconstructed.mp4")
    if not os.path.exists(reconstructed):
        logger.error(f"Pipeline completed but reconstructed.mp4 not found at {reconstructed}")
        return False

    # Generate video metadata
    meta = _get_video_metadata(reconstructed)
    meta_path = os.path.join(output_dir, "video_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"✓ Local pipeline complete. reconstructed.mp4 created ({meta})")
    return True


def _get_video_metadata(video_path: str) -> dict:
    """Extract fps and duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration,width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(res.stdout)
        stream = info.get("streams", [{}])[0]
        fmt = info.get("format", {})

        fps_str = stream.get("r_frame_rate", "30/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
        duration = float(stream.get("duration", 0) or fmt.get("duration", 0))

        return {
            "fps": round(fps, 2),
            "duration": round(duration, 3),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
        }
    except Exception as e:
        logger.warning(f"ffprobe metadata extraction failed: {e}")
        return {"fps": 24.0, "duration": 0.0, "width": 0, "height": 0}


def run_graph_pipeline(output_dir: str, env: dict = None) -> bool:
    """Run the graph pipeline as a subprocess."""
    output_dir = os.path.abspath(output_dir)
    script = os.path.join(PREPROC_DIR, "run_neo4j_pipeline.py")
    cmd = [sys.executable, script, output_dir]

    logger.info(f"Running graph pipeline for: {output_dir}")
    
    if env is None:
        env = os.environ.copy()

    result = subprocess.run(cmd, cwd=PREPROC_DIR, stdout=sys.stdout, stderr=sys.stderr, env=env)

    if result.returncode != 0:
        logger.error(f"Graph pipeline failed with exit code {result.returncode}")
        return False

    logger.info(f"✓ Graph pipeline complete for {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Resume local pipeline for partially processed videos")
    parser.add_argument("output_dirs", nargs="+", help="Output directories (e.g. outputs/39fbc2c8)")
    parser.add_argument("--source", type=str, default=None,
                        help="Explicit source video path (only for single video)")
    parser.add_argument("--with-graph", action="store_true",
                        help="Also run graph pipeline after local pipeline completes")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if reconstructed.mp4 already exists")
    args = parser.parse_args()

    results = {"success": [], "failed": [], "skipped": []}

    for output_dir in args.output_dirs:
        video_id = Path(output_dir).name
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {video_id}")

        # Check if already done
        reconstructed = os.path.join(output_dir, "reconstructed.mp4")
        if args.skip_existing and os.path.exists(reconstructed):
            logger.info(f"  ✓ reconstructed.mp4 already exists. Skipping local pipeline.")
            if args.with_graph:
                run_graph_pipeline(output_dir)
            results["skipped"].append(video_id)
            continue

        # Find source video
        if args.source and len(args.output_dirs) == 1:
            source_video = args.source
        else:
            source_video = find_source_video(video_id)

        if not source_video:
            logger.error(f"  ✗ Cannot find source video for {video_id}. "
                         f"Use --source to specify manually.")
            results["failed"].append(video_id)
            continue

        logger.info(f"  Source video: {source_video}")

        # Check if JSONs exist (needed for graph building)
        has_jsons = os.path.exists(os.path.join(output_dir, "scenes.json"))
        if not has_jsons:
            logger.warning(f"  ⚠ No scenes.json found — graph building will fail without AVI data.")
            logger.warning(f"  Run fetch_avi_results.py first, or use --with-graph after fetching.")

        # Run local pipeline
        ok = run_local_pipeline(source_video, output_dir)
        if not ok:
            results["failed"].append(video_id)
            continue

        # Optionally run graph pipeline
        if args.with_graph:
            if not has_jsons:
                logger.warning(f"  ⚠ Skipping graph build — no AVI JSONs present.")
            else:
                graph_ok = run_graph_pipeline(output_dir)
                if not graph_ok:
                    logger.error(f"  ✗ Graph pipeline failed for {video_id}")
                    results["failed"].append(video_id)
                    continue

        results["success"].append(video_id)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Success: {len(results['success'])}  {results['success']}")
    logger.info(f"  Failed:  {len(results['failed'])}  {results['failed']}")
    logger.info(f"  Skipped: {len(results['skipped'])}  {results['skipped']}")


if __name__ == "__main__":
    main()
