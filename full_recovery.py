"""
full_recovery.py — Combined Recovery Orchestrator
===================================================
Automatically diagnoses what's missing for each video and runs
the appropriate recovery steps in the right order.

Usage:
    # Auto-diagnose and recover specific videos
    python full_recovery.py 39fbc2c8 ac1682fb b120483a

    # Auto-diagnose and recover ALL videos with issues
    python full_recovery.py --all

    # Dry run — just show what would be done
    python full_recovery.py --all --dry-run

    # Only do specific steps
    python full_recovery.py 39fbc2c8 --steps fetch-avi,graph
    python full_recovery.py b120483a --steps graph
"""

import os
import sys
import hashlib
import argparse
import logging
import threading
import queue
import concurrent.futures
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("full_recovery.log", mode="a")])
logger = logging.getLogger("FullRecovery")

# ── Import recovery modules ─────────────────────────────────────────
# We import from the sibling scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

PREPROC_DIR = os.path.join(SCRIPT_DIR, "video_rag_preprocessing")
sys.path.insert(0, PREPROC_DIR)


REQUIRED_JSON_FILES = ["scenes.json"]  # Minimum for graph building
CORE_JSON_FILES = ["scenes.json", "transcript.json", "ocr.json", "keywords.json", "rag_chunks.json"]
ALL_JSON_FILES = ["scenes.json", "raw_insights.json", "transcript.json",
                  "ocr.json", "keywords.json", "rag_chunks.json"]


class GpuManager:
    """Manages available GPU IDs for parallel tasks."""
    def __init__(self, gpu_ids: list):
        self.gpu_queue = queue.Queue()
        for gid in gpu_ids:
            self.gpu_queue.put(gid)
        self.num_gpus = len(gpu_ids)

    def acquire(self):
        return self.gpu_queue.get()

    def release(self, gid):
        self.gpu_queue.put(gid)


def diagnose_video(output_dir: str) -> dict:
    """
    Diagnose what's present and what's missing for a video.
    Returns a diagnosis dict with recommended recovery steps.
    """
    path = Path(output_dir)
    video_id = path.name

    diagnosis = {
        "video_id": video_id,
        "path": str(path),
        "exists": path.exists(),
        "has_audio": (path / "optimal_audio.wav").exists(),
        "has_reconstructed": (path / "reconstructed.mp4").exists(),
        "has_video_metadata": (path / "video_metadata.json").exists(),
        "has_scores": (path / "scores.json").exists(),
        "has_clips": (path / "clips").is_dir() and len(list((path / "clips").glob("*.mp4"))) > 0,
        "json_files": {},
        "steps_needed": [],
        "scenario": "unknown",
    }

    for f in ALL_JSON_FILES:
        diagnosis["json_files"][f] = (path / f).exists()

    has_all_jsons = all(diagnosis["json_files"].values())
    has_core_jsons = all((path / f).exists() for f in CORE_JSON_FILES)
    has_scenes = diagnosis["json_files"].get("scenes.json", False)
    has_raw_insights = diagnosis["json_files"].get("raw_insights.json", False)
    has_any_jsons = any(diagnosis["json_files"].values())

    # Determine scenario based on what's present
    if diagnosis["has_reconstructed"] and has_all_jsons and diagnosis["has_clips"]:
        # Everything present — only graph rebuild needed
        diagnosis["scenario"] = "complete_needs_graph"
        diagnosis["steps_needed"] = ["graph"]
    elif diagnosis["has_reconstructed"] and has_core_jsons and not has_raw_insights:
        # Has reconstruction + core JSONs but missing raw_insights
        # Can still build graph (entities will be sparse without raw_insights)
        diagnosis["scenario"] = "needs_avi_fetch_and_graph"
        diagnosis["steps_needed"] = ["fetch-avi", "graph"]
    elif diagnosis["has_reconstructed"] and has_all_jsons:
        # Has everything except clips — graph will create them
        diagnosis["scenario"] = "needs_graph"
        diagnosis["steps_needed"] = ["graph"]
    elif diagnosis["has_reconstructed"] and not has_core_jsons:
        # Has reconstruction but missing core JSONs
        diagnosis["scenario"] = "needs_avi_and_graph"
        diagnosis["steps_needed"] = ["fetch-avi", "graph"]
    elif not diagnosis["has_reconstructed"] and has_core_jsons and not has_raw_insights:
        # Missing reconstruction, has core JSONs but not raw_insights
        diagnosis["scenario"] = "needs_local_avi_graph"
        diagnosis["steps_needed"] = ["local-pipeline", "fetch-avi", "graph"]
    elif not diagnosis["has_reconstructed"] and has_all_jsons:
        # Missing only reconstruction
        diagnosis["scenario"] = "needs_local_and_graph"
        diagnosis["steps_needed"] = ["local-pipeline", "graph"]
    elif not diagnosis["has_reconstructed"] and not has_any_jsons:
        # Missing everything — audio only
        diagnosis["scenario"] = "needs_everything"
        diagnosis["steps_needed"] = ["local-pipeline", "fetch-avi", "graph"]
    else:
        # Partial state — needs investigation
        diagnosis["scenario"] = "partial"
        steps = []
        if not diagnosis["has_reconstructed"]:
            steps.append("local-pipeline")
        if not has_core_jsons:
            steps.append("fetch-avi")
        steps.append("graph")
        diagnosis["steps_needed"] = steps

    return diagnosis


def print_diagnosis(diagnosis: dict):
    """Pretty-print a diagnosis."""
    d = diagnosis
    logger.info(f"  Video ID:         {d['video_id']}")
    logger.info(f"  Scenario:         {d['scenario']}")
    logger.info(f"  optimal_audio:    {'✓' if d['has_audio'] else '✗'}")
    logger.info(f"  reconstructed:    {'✓' if d['has_reconstructed'] else '✗'}")
    logger.info(f"  video_metadata:   {'✓' if d['has_video_metadata'] else '✗'}")
    logger.info(f"  clips/:           {'✓' if d['has_clips'] else '✗'}")

    for f, exists in d["json_files"].items():
        logger.info(f"  {f:20s} {'✓' if exists else '✗'}")

    logger.info(f"  Steps needed:     {d['steps_needed']}")


def recover_video(diagnosis: dict, steps_filter: list = None, dry_run: bool = False, gpu_manager: GpuManager = None) -> bool:
    """Execute recovery steps for a diagnosed video."""
    video_id = diagnosis["video_id"]
    output_dir = diagnosis["path"]
    steps = diagnosis["steps_needed"]

    if steps_filter:
        steps = [s for s in steps if s in steps_filter]

    if not steps:
        logger.info(f"  No recovery steps needed for {video_id}")
        return True

    logger.info(f"[{video_id}] Recovery plan: {' → '.join(steps)}")

    if dry_run:
        logger.info(f"[{video_id}] [DRY RUN] Would execute: {steps}")
        return True

    # Setup per-video logging
    log_path = os.path.join(output_dir, "recovery.log")
    os.makedirs(output_dir, exist_ok=True)
    
    gpu_id = None
    if gpu_manager:
        logger.info(f"[{video_id}] Waiting for available GPU...")
        gpu_id = gpu_manager.acquire()
        logger.info(f"[{video_id}] Acquired GPU: {gpu_id}")

    try:
        with open(log_path, "a") as log_f:
            log_f.write(f"\n{'='*60}\nStarting recovery: {video_id}\nPlan: {steps}\nGPU: {gpu_id}\n{'='*60}\n")
            log_f.flush()

            for step in steps:
                logger.info(f"[{video_id}] Step: {step}")
                log_f.write(f"\n── Step: {step} ──\n")
                log_f.flush()

                if step == "fetch-avi":
                    ok = _step_fetch_avi(video_id, output_dir, log_f)
                elif step == "local-pipeline":
                    ok = _step_local_pipeline(video_id, output_dir, log_f, gpu_id)
                elif step == "graph":
                    ok = _step_graph_build(output_dir, log_f, gpu_id)
                else:
                    logger.error(f"[{video_id}] Unknown step: {step}")
                    ok = False

                if not ok:
                    logger.error(f"[{video_id}] ✗ Step '{step}' failed. See {log_path} for details.")
                    return False

        logger.info(f"[{video_id}] ✓ Recovery successful.")
        return True
    finally:
        if gpu_manager and gpu_id is not None:
            gpu_manager.release(gpu_id)


def _step_fetch_avi(video_id: str, output_dir: str, log_f) -> bool:
    """Run the AVI fetch step."""
    # We temporarily redirect stdout to the log file for this step
    # since it uses a library-level logger. 
    # NOTE: This only works if the logger uses sys.stdout/err.
    from fetch_avi_results import create_avi_client, list_avi_videos, save_avi_results

    log_f.write(f"Fetching AVI results for {video_id}...\n")
    try:
        client = create_avi_client()
        videos = list_avi_videos(client)

        # Find matching AVI video
        avi_video_id = None
        for v in videos:
            name = v.get("name", "")
            if name == f"video_{video_id}" and v.get("state") == "Processed":
                avi_video_id = v.get("id")
                break

        if not avi_video_id:
            log_f.write(f"  ✗ No processed AVI video found for video_{video_id}\n")
            return False

        log_f.write(f"  Found AVI video: {avi_video_id}\n")
        return save_avi_results(client, avi_video_id, output_dir, video_id)
    except Exception as e:
        log_f.write(f"  ✗ Error in fetch-avi: {e}\n")
        import traceback
        traceback.print_exc(file=log_f)
        return False


def _step_local_pipeline(video_id: str, output_dir: str, log_f, gpu_id: int = None) -> bool:
    """Run the local pipeline step."""
    from resume_local_pipeline import find_source_video, run_local_pipeline

    source = find_source_video(video_id)
    if not source:
        log_f.write(f"  ✗ Cannot find source video for {video_id}\n")
        return False

    log_f.write(f"  Source video: {source}\n")
    
    # Override sys.stdout and sys.stderr for the duration of run_local_pipeline
    # This is not perfectly thread-safe if other threads use sys.stdout directly,
    # but run_local_pipeline calls subprocess.run with stdout=sys.stdout.
    # We will pass the log_f to a custom env for the subprocess instead.
    
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
    # We need to capture the output of run_local_pipeline. 
    # Since we modified run_local_pipeline to take an env, but it still uses sys.stdout,
    # we'll monkey-patch sys.stdout/err if we're in a single thread, OR
    # better yet, we should have modified run_local_pipeline to take stdout/stderr handles.
    
    # For now, let's just use subprocess directly here to be sure.
    import subprocess
    script = os.path.join(PREPROC_DIR, "run_pipeline.py")
    cmd = [sys.executable, script, "--video_path", os.path.abspath(source), "--output_dir", os.path.abspath(output_dir)]
    
    result = subprocess.run(
        cmd,
        cwd=PREPROC_DIR,
        stdout=log_f,
        stderr=log_f,
        env=env
    )
    return result.returncode == 0


def _step_graph_build(output_dir: str, log_f, gpu_id: int = None) -> bool:
    """Run the graph build step as a subprocess."""
    import subprocess

    # Verify minimum requirements
    if not os.path.exists(os.path.join(output_dir, "scenes.json")):
        log_f.write(f"  ✗ Cannot build graph — scenes.json missing\n")
        return False

    try:
        output_dir_abs = os.path.abspath(output_dir)
        script = os.path.join(PREPROC_DIR, "run_neo4j_pipeline.py")
        cmd = [sys.executable, script, output_dir_abs]
        
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        result = subprocess.run(
            cmd,
            cwd=PREPROC_DIR,
            stdout=log_f,
            stderr=log_f,
            env=env
        )
        
        if result.returncode != 0:
            log_f.write(f"  ✗ Graph pipeline exited with code {result.returncode}\n")
            return False
        return True
    except Exception as e:
        log_f.write(f"  ✗ Graph build failed: {e}\n")
        import traceback
        traceback.print_exc(file=log_f)
        return False


def main():
    parser = argparse.ArgumentParser(description="Full recovery orchestrator for failed pipeline videos")
    parser.add_argument("video_ids", nargs="*", help="Video IDs to recover (folder names under outputs/)")
    parser.add_argument("--all", action="store_true",
                        help="Auto-discover all videos that need recovery")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only diagnose, don't execute recovery")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated list of steps to run: local-pipeline,fetch-avi,graph")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Only show diagnosis for all videos")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    steps_filter = args.steps.split(",") if args.steps else None

    # Collect video directories
    if args.all or args.diagnose_only:
        outputs_root = Path("outputs")
        if not outputs_root.exists():
            logger.error("No outputs/ directory found")
            sys.exit(1)
        video_dirs = sorted([str(d) for d in outputs_root.iterdir() if d.is_dir()])
    elif args.video_ids:
        video_dirs = [f"outputs/{vid}" if not vid.startswith("outputs/") else vid
                      for vid in args.video_ids]
    else:
        parser.print_help()
        sys.exit(1)

    # Diagnose all
    diagnoses = []
    for d in video_dirs:
        diagnosis = diagnose_video(d)
        diagnoses.append(diagnosis)

    # Show diagnosis
    needs_recovery = [d for d in diagnoses if d["steps_needed"]]
    complete = [d for d in diagnoses if not d["steps_needed"]]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"DIAGNOSIS REPORT")
    logger.info(f"  Total videos:      {len(diagnoses)}")
    logger.info(f"  Fully complete:    {len(complete)}")
    logger.info(f"  Need recovery:     {len(needs_recovery)}")

    if needs_recovery:
        logger.info(f"\n  Videos needing recovery:")
        for d in needs_recovery:
            logger.info(f"\n  {'─' * 50}")
            print_diagnosis(d)

    if args.diagnose_only:
        return

    if not needs_recovery:
        logger.info("  All videos are complete. Nothing to recover.")
        return

    # Execute recovery
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EXECUTING RECOVERY (Parallel Workers: {args.workers})")

    results = {"success": [], "failed": [], "skipped": []}
    gpu_manager = GpuManager(gpu_ids=[0, 1, 2, 3])  # Fixed list of 4 GPUs

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Create a mapping of future to video_id
        future_to_vid = {
            executor.submit(recover_video, d, steps_filter, args.dry_run, gpu_manager): d["video_id"]
            for d in needs_recovery
        }

        for future in concurrent.futures.as_completed(future_to_vid):
            vid = future_to_vid[future]
            try:
                ok = future.result()
                if ok:
                    results["success"].append(vid)
                else:
                    results["failed"].append(vid)
            except Exception as exc:
                logger.error(f"[{vid}] Recovery generated an exception: {exc}")
                import traceback
                logger.error(traceback.format_exc())
                results["failed"].append(vid)

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"  Success: {len(results['success'])}  {results['success']}")
    logger.info(f"  Failed:  {len(results['failed'])}  {results['failed']}")


if __name__ == "__main__":
    main()
