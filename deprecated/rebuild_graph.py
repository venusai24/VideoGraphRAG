# Deprecated – not used in final pipeline
"""
rebuild_graph.py — Graph-Only Rebuild
======================================
For videos that already have all data (reconstructed.mp4, JSONs, clips)
but need their Neo4j graph rebuilt.

Usage:
    # Single video
    python rebuild_graph.py outputs/b120483a

    # Multiple videos
    python rebuild_graph.py outputs/b120483a outputs/ac1682fb

    # All output directories that are ready for graph building
    python rebuild_graph.py --all

    # Dry run — just validate
    python rebuild_graph.py --all --dry-run
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("rebuild_graph.log", mode="a")])
logger = logging.getLogger("RebuildGraph")

PREPROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_rag_preprocessing")

# Minimum files needed for graph building
REQUIRED_FILES = ["scenes.json"]
RECOMMENDED_FILES = ["raw_insights.json", "transcript.json", "ocr.json", "keywords.json", "rag_chunks.json"]


def validate_output_dir(output_dir: str) -> dict:
    """
    Check what files exist in an output directory.
    Returns a status dict with details.
    """
    path = Path(output_dir)
    status = {
        "path": str(path),
        "video_id": path.name,
        "exists": path.exists(),
        "has_reconstructed": (path / "reconstructed.mp4").exists(),
        "has_clips": (path / "clips").is_dir() and len(list((path / "clips").glob("*.mp4"))) > 0,
        "missing_required": [],
        "missing_recommended": [],
        "ready": False,
    }

    if not status["exists"]:
        return status

    for f in REQUIRED_FILES:
        if not (path / f).exists():
            status["missing_required"].append(f)

    for f in RECOMMENDED_FILES:
        if not (path / f).exists():
            status["missing_recommended"].append(f)

    status["ready"] = len(status["missing_required"]) == 0
    return status


def rebuild_graph_for_video(output_dir: str, dry_run: bool = False):
    """Run the graph pipeline for a single video output directory."""
    status = validate_output_dir(output_dir)

    logger.info(f"{'=' * 60}")
    logger.info(f"Video: {status['video_id']}")
    logger.info(f"  Path:            {status['path']}")
    logger.info(f"  reconstructed:   {'✓' if status['has_reconstructed'] else '✗'}")
    logger.info(f"  clips/:          {'✓' if status['has_clips'] else '✗'}")
    logger.info(f"  missing required:    {status['missing_required'] or 'none'}")
    logger.info(f"  missing recommended: {status['missing_recommended'] or 'none'}")

    if not status["ready"]:
        logger.error(f"  ✗ SKIPPED — missing required files: {status['missing_required']}")
        return False

    if status["missing_recommended"]:
        logger.warning(f"  ⚠ Proceeding with missing recommended files (graph may be sparse)")

    if dry_run:
        logger.info(f"  [DRY RUN] Would rebuild graph for {status['video_id']}")
        return True

    try:
        logger.info(f"  → Running graph pipeline...")
        output_dir_abs = os.path.abspath(output_dir)
        script = os.path.join(PREPROC_DIR, "run_neo4j_pipeline.py")
        cmd = [sys.executable, script, output_dir_abs]
        
        result = subprocess.run(
            cmd,
            cwd=PREPROC_DIR,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Graph pipeline exited with code {result.returncode}")
        
        logger.info(f"  ✓ Graph rebuild COMPLETED for {status['video_id']}")
        return True
    except Exception as e:
        logger.error(f"  ✗ Graph rebuild FAILED for {status['video_id']}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Rebuild Neo4j graph for processed videos")
    parser.add_argument("output_dirs", nargs="*", help="One or more outputs/<video_id> directories")
    parser.add_argument("--all", action="store_true", help="Process all directories under outputs/")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't actually rebuild")
    args = parser.parse_args()

    if args.all:
        outputs_root = Path("outputs")
        if not outputs_root.exists():
            logger.error("No outputs/ directory found")
            sys.exit(1)
        dirs = sorted([str(d) for d in outputs_root.iterdir() if d.is_dir()])
    elif args.output_dirs:
        dirs = args.output_dirs
    else:
        parser.print_help()
        sys.exit(1)

    logger.info(f"Processing {len(dirs)} output directories...")

    results = {"success": [], "failed": [], "skipped": []}

    for d in dirs:
        status = validate_output_dir(d)
        if not status["exists"]:
            logger.warning(f"Directory does not exist: {d}")
            results["skipped"].append(d)
            continue
        if not status["ready"]:
            logger.warning(f"Not ready (missing {status['missing_required']}): {d}")
            results["skipped"].append(d)
            continue

        ok = rebuild_graph_for_video(d, dry_run=args.dry_run)
        if ok:
            results["success"].append(d)
        else:
            results["failed"].append(d)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY")
    logger.info(f"  Success: {len(results['success'])}")
    logger.info(f"  Failed:  {len(results['failed'])}")
    logger.info(f"  Skipped: {len(results['skipped'])}")

    if results["failed"]:
        logger.error(f"  Failed directories: {results['failed']}")
    if results["skipped"]:
        logger.info(f"  Skipped directories: {results['skipped']}")


if __name__ == "__main__":
    main()
