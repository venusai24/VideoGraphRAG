"""
VideoGraphRAG End-to-End Pipeline Orchestrator
================================================
Manages the lifecycle of videos from raw input through to Neo4j ingestion.

Responsibilities:
  - InputWatcher: scans input/ for new .mp4 files
  - LocalPipelineWorker: semantic video reduction (run_pipeline.run)
  - AzurePipelineWorker: scene + semantic extraction via AVI
  - OutputSynchronizer: align & consolidate into outputs/<video_id>/
  - GraphIngestionWorker: invoke graph builders for clipping + Neo4j push

State machine:
  PENDING → LOCAL_RUNNING / AZURE_RUNNING (parallel) → SYNCING → GRAPH_BUILDING → COMPLETED | FAILED
"""

import os
import sys
import uuid
import json
import shutil
import asyncio
import logging
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────
# Logging — configure BEFORE any other module touches logging
# ──────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file_handler = logging.FileHandler("orchestrator.log", mode="a")
log_file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
log_stream_handler = logging.StreamHandler()
log_stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(log_file_handler)
root_logger.addHandler(log_stream_handler)

logger = logging.getLogger("Orchestrator")

# ──────────────────────────────────────────────────────────────────────
# Resolve imports from video_rag_preprocessing
# ──────────────────────────────────────────────────────────────────────
PREPROC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_rag_preprocessing")
sys.path.insert(0, PREPROC_DIR)

from avi_client import AVIClient


# ──────────────────────────────────────────────────────────────────────
# Local Pipeline: invoke run_pipeline.run() as a subprocess
# ──────────────────────────────────────────────────────────────────────
# We run it as a subprocess rather than in-process because:
#   1. It loads torch/CLIP/DINO/YOLO — these cannot be pickled for ProcessPoolExecutor
#   2. It needs CWD to be video_rag_preprocessing/ for relative imports to resolve
#   3. Subprocess isolation prevents model memory from leaking into the orchestrator

async def _stream_subprocess_output(process, stdout_prefix, stderr_prefix):
    """Read subprocess output line by line and log it in real-time."""
    async def _read(stream, prefix, is_stderr):
        while True:
            line = await stream.readline()
            if not line:
                break
            line_str = line.decode().strip()
            if line_str:
                if is_stderr:
                    logger.warning(f"{prefix} {line_str}")
                else:
                    logger.info(f"{prefix} {line_str}")

    await asyncio.gather(
        _read(process.stdout, stdout_prefix, False),
        _read(process.stderr, stderr_prefix, True)
    )

async def run_local_pipeline_subprocess(video_path: str, output_dir: str) -> dict:
    """
    Invoke the existing semantic video reduction pipeline as an async subprocess.
    Returns video metadata (fps, duration) from the produced output.
    """
    # Convert to absolute paths — the subprocess CWD is video_rag_preprocessing/
    # so relative paths from the project root won't resolve.
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)

    script = os.path.join(PREPROC_DIR, "run_pipeline.py")
    cmd = [
        sys.executable, script,
        "--video_path", video_path,
        "--output_dir", output_dir,
    ]
    logger.info(f"Launching local pipeline: {Path(video_path).name} → {output_dir}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=PREPROC_DIR,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Stream stdout/stderr in real-time
    await asyncio.gather(
        _stream_subprocess_output(process, "[LOCAL]", "[LOCAL-ERR]"),
        process.wait()
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Local pipeline failed (exit code {process.returncode}). "
            f"Check [LOCAL] lines above."
        )

    # Extract metadata from the produced video
    reconstructed = os.path.join(output_dir, "reconstructed.mp4")
    meta = _get_video_metadata(reconstructed) if os.path.exists(reconstructed) else {}
    meta_path = os.path.join(output_dir, "video_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def _get_video_metadata(video_path: str) -> dict:
    """Extract fps and duration from a video file using ffprobe."""
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


# ──────────────────────────────────────────────────────────────────────
# Fallback scene extraction (if Azure fails)
# ──────────────────────────────────────────────────────────────────────

def generate_fallback_scenes(video_duration: float, target_chunk_sec: float = 11.0) -> list:
    """
    Generate uniform scene boundaries as fallback when Azure is unavailable.
    Produces scenes of ~target_chunk_sec each, covering the full video.
    """
    if video_duration <= 0:
        video_duration = 60.0  # safe default

    scenes = []
    t = 0.0
    scene_id = 1
    while t < video_duration:
        end = min(t + target_chunk_sec, video_duration)
        h_s, m_s, s_s = int(t // 3600), int((t % 3600) // 60), t % 60
        h_e, m_e, s_e = int(end // 3600), int((end % 3600) // 60), end % 60
        scenes.append({
            "id": scene_id,
            "instances": [{
                "start": f"{h_s}:{m_s:02d}:{s_s:06.3f}",
                "end": f"{h_e}:{m_e:02d}:{s_e:06.3f}",
            }]
        })
        scene_id += 1
        t = end
    return scenes


# ──────────────────────────────────────────────────────────────────────
# Graph pipeline invocation (subprocess with correct CWD)
# ──────────────────────────────────────────────────────────────────────

async def invoke_graph_pipeline(outputs_dir: str):
    """
    Invoke the graph pipeline as an async subprocess with CWD=video_rag_preprocessing/
    so that all relative imports (data_loader, graph_store, temporal_clip_graph, etc.) resolve.
    """
    outputs_dir = os.path.abspath(outputs_dir)
    script = os.path.join(PREPROC_DIR, "run_neo4j_pipeline.py")
    cmd = [sys.executable, script, outputs_dir]

    logger.info(f"Invoking graph pipeline for: {outputs_dir}")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=PREPROC_DIR,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    await asyncio.gather(
        _stream_subprocess_output(process, "[GRAPH]", "[GRAPH-ERR]"),
        process.wait()
    )

    if process.returncode != 0:
        raise RuntimeError(f"Graph pipeline failed (exit code {process.returncode})")


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

class VideoOrchestrator:
    def __init__(self, input_dir="input", outputs_dir="outputs"):
        self.input_dir = Path(input_dir)
        self.outputs_dir = Path(outputs_dir)
        self.input_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)

        load_dotenv()

        self.avi_client = AVIClient(
            api_key=os.getenv("AZURE_VIDEO_INDEXER_API_KEY", ""),
            account_id=os.getenv("VIDEO_INDEXER_ACCOUNT_ID", ""),
            location=os.getenv("VIDEO_INDEXER_LOCATION", "trial"),
            account_type=os.getenv("VIDEO_INDEXER_ACCOUNT_TYPE", "trial"),
        )

        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.state_tracker = {}
        self.processing_files = set()

    # ── State tracking ───────────────────────────────────────────────

    def _update_state(self, video_id: str, state: str, message: str = ""):
        self.state_tracker[video_id] = state
        parts = [f"[{video_id}]", f"[{state}]"]
        if message:
            parts.append(message)
        logger.info(" - ".join(parts))

    # ── Local pipeline worker ────────────────────────────────────────

    async def _run_local_pipeline(self, video_path: str, video_id: str) -> dict:
        output_dir = str(self.outputs_dir / video_id)
        reconstructed_path = os.path.join(output_dir, "reconstructed.mp4")
        meta_path = os.path.join(output_dir, "video_metadata.json")
        
        if os.path.exists(reconstructed_path) and os.path.exists(meta_path):
            self._update_state(video_id, "LOCAL_SKIPPED", f"Outputs already exist for {Path(video_path).name}")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return {"output_dir": output_dir, "meta": meta}

        self._update_state(video_id, "LOCAL_RUNNING", f"Processing {Path(video_path).name}")

        try:
            meta = await run_local_pipeline_subprocess(video_path, output_dir)
            self._update_state(video_id, "LOCAL_COMPLETED",
                               f"duration={meta.get('duration', '?')}s fps={meta.get('fps', '?')}")
            return {"output_dir": output_dir, "meta": meta}
        except Exception as e:
            self._update_state(video_id, "LOCAL_FAILED", str(e))
            raise

    # ── Azure pipeline worker ────────────────────────────────────────

    async def _run_azure_pipeline(self, video_path: str, video_id: str) -> dict:
        output_dir = str(self.outputs_dir / video_id)
        expected_azure_files = ["transcript.json", "ocr.json", "scenes.json", "keywords.json", "rag_chunks.json"]
        if all(os.path.exists(os.path.join(output_dir, f)) for f in expected_azure_files):
            self._update_state(video_id, "AZURE_SKIPPED", "Azure outputs already exist.")
            azure_data = {}
            for f in expected_azure_files:
                with open(os.path.join(output_dir, f), "r") as fh:
                    azure_data[f] = json.load(fh)
            # raw_insights.json might not be strictly needed, but let's include if exists
            raw_path = os.path.join(output_dir, "raw_insights.json")
            if os.path.exists(raw_path):
                with open(raw_path, "r") as fh:
                    azure_data["raw_insights.json"] = json.load(fh)
            return azure_data

        self._update_state(video_id, "AZURE_RUNNING", "Uploading and indexing via AVI...")

        loop = asyncio.get_running_loop()
        try:
            avi_video_id = await loop.run_in_executor(
                self.thread_pool, self.avi_client.upload_video, video_path, f"video_{video_id}"
            )
            full_insight = await loop.run_in_executor(
                self.thread_pool, self.avi_client.wait_for_index, avi_video_id
            )
            extracted = self.avi_client.extract_structured_data(full_insight)
            chunks = self.avi_client.generate_rag_chunks(video_id, extracted)

            self._update_state(video_id, "AZURE_COMPLETED",
                               f"scenes={len(extracted.get('scenes', []))}, "
                               f"transcript_segs={len(extracted.get('transcript', []))}")
            return {
                "raw_insights.json": full_insight,
                "transcript.json": extracted["transcript"],
                "ocr.json": extracted["ocr"],
                "scenes.json": extracted["scenes"],
                "keywords.json": extracted["keywords"],
                "rag_chunks.json": chunks,
            }
        except Exception as e:
            self._update_state(video_id, "AZURE_FAILED", str(e))
            return None

    # ── Per-video lifecycle ──────────────────────────────────────────

    async def process_video(self, video_path: Path):
        import hashlib
        video_id = hashlib.md5(video_path.name.encode()).hexdigest()[:8]
        self.processing_files.add(str(video_path))
        self._update_state(video_id, "PENDING", f"Registered: {video_path.name}")

        try:
            # ── 1. Parallel Execution ────────────────────────────────
            local_task = asyncio.create_task(self._run_local_pipeline(str(video_path), video_id))
            azure_task = asyncio.create_task(self._run_azure_pipeline(str(video_path), video_id))

            # return_exceptions=True: Azure failure won't crash local pipeline
            results = await asyncio.gather(local_task, azure_task, return_exceptions=True)
            local_result, azure_data = results

            # Local pipeline MUST succeed — it produces the video source
            if isinstance(local_result, Exception):
                self._update_state(video_id, "FAILED", f"Local pipeline crashed: {local_result}")
                return

            output_dir = local_result["output_dir"]
            video_meta = local_result["meta"]

            # Azure fallback: generate uniform scenes based on actual video duration
            if isinstance(azure_data, Exception) or azure_data is None:
                self._update_state(video_id, "AZURE_FALLBACK",
                                   "Generating local scene boundaries as fallback...")
                fallback_scenes = generate_fallback_scenes(video_meta.get("duration", 60.0))
                azure_data = {
                    "scenes.json": fallback_scenes,
                    "transcript.json": [],
                    "ocr.json": [],
                    "keywords.json": [],
                    "rag_chunks.json": [],
                }

            # ── 2. Output Synchronization ────────────────────────────
            self._update_state(video_id, "SYNCING", "Writing Azure data to unified output dir...")
            target_dir = Path(output_dir)
            for filename, data in azure_data.items():
                with open(target_dir / filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            # ── 3. Graph Pipeline ────────────────────────────────────
            self._update_state(video_id, "GRAPH_BUILDING",
                               "Invoking graph pipeline for clip construction + Neo4j ingestion...")
            await invoke_graph_pipeline(str(target_dir))

            self._update_state(video_id, "COMPLETED", "End-to-end pipeline successful.")

            # Move processed video out of input queue
            processed_dir = self.input_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            shutil.move(str(video_path), str(processed_dir / video_path.name))

        except Exception as e:
            self._update_state(video_id, "FAILED",
                               f"Pipeline crashed: {e}\n{traceback.format_exc()}")
        finally:
            self.processing_files.discard(str(video_path))

    # ── Input watcher ────────────────────────────────────────────────

    async def watch_input_directory(self):
        logger.info(f"Watching {self.input_dir} for new .mp4 files...")
        while True:
            for file_path in sorted(self.input_dir.glob("*.mp4")):
                if str(file_path) not in self.processing_files:
                    asyncio.create_task(self.process_video(file_path))
            await asyncio.sleep(5)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    orchestrator = VideoOrchestrator()
    try:
        asyncio.run(orchestrator.watch_input_directory())
    except KeyboardInterrupt:
        logger.info("Orchestrator shutting down.")
