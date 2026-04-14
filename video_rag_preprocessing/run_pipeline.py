import argparse
import logging
import os
import json
import subprocess
import cv2
import numpy as np

from pipeline import load_config, VideoIngestor, FeatureExtractor, CompressorEngine
from pipeline.clip_grouping.grouping import group_frames, calculate_token_cost


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SemanticCompressor")


def run(video_path: str, output_dir: str, audio_path: str = None):
    logger = setup_logging()
    
    if audio_path is None:
        audio_path = video_path # Default to source video's audio
        
    logger.info(f"Starting Semantic Compressor for video: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    config = load_config()
    logger.info("Configuration loaded.")

    ingestor = VideoIngestor(video_path=video_path)
    extractor = FeatureExtractor(
        clip_dim=config["dimensions"]["clip"],
        dino_dim=config["dimensions"]["dino"],
    )
    engine = CompressorEngine(config=config)

    logger.info(
        "Pipeline components initialized (native FPS=%.2f). Commencing frame streaming...",
        ingestor.native_fps,
    )

    frame_count = 0
    for frame_data, timestamp in ingestor.stream_frames():
        native_frame = extractor.process_frame(frame_data, timestamp)
        engine.push(native_frame)
        frame_count += 1

        if frame_count % 30 == 0:
            logger.info(
                f"Processed {frame_count} native frames. "
                f"Compressor time target: {engine.target_time:.2f}s"
            )

    # Flush remaining buffers
    output_frames = engine.finalize()
    logger.info(
        f"Emission complete. Generated {len(output_frames)} standardized "
        f"{engine.fps} FPS semantic output frames from {frame_count} native frames."
    )

    # ── Save score telemetry ──────────────────────────────────────────
    dump_path = os.path.join(output_dir, "scores.json")
    payload = []
    syn_count = 0

    for f in output_frames:
        if f.is_synthetic:
            syn_count += 1
        payload.append({
            "target": f.target_timestamp,
            "native": f.native_timestamp,
            "is_synthetic": f.is_synthetic,
            "scores": f.scores,
        })

    with open(dump_path, "w") as fp:
        json.dump(payload, fp, indent=2)

    logger.info(
        f"Saved execution telemetry to {dump_path}. "
        f"Included {syn_count} synthetic gap-fills."
    )

    # ── Reconstruct 24 FPS output video ──────────────────────────────
    if frame_count == 0:
        logger.warning("No frames read from video. Exiting without reconstruction.")
        return

    logger.info("Starting video reconstruction...")
    out_video_path = os.path.join(output_dir, "reconstructed.mp4")

    # Determine output resolution from first real frame
    height, width = None, None
    for of in output_frames:
        if of.frame_data is not None:
            height, width = of.frame_data.shape[:2]
            break

    if height is None:
        logger.error("No frame_data found in output frames; cannot write video.")
        return

    _group_and_save_clips(
        output_frames=output_frames,
        output_dir=output_dir,
        config=config,
        fps=engine.fps,
        height=height,
        width=width,
        logger=logger,
    )

    _write_video_ffmpeg(
        output_frames=output_frames,
        out_path=out_video_path,
        fps=engine.fps,
        height=height,
        width=width,
        logger=logger,
        audio_path=audio_path,
    )
    logger.info(f"Pipeline executed successfully. Final video saved at: {out_video_path}")


# ── helpers ───────────────────────────────────────────────────────────

def _get_audio_info(media_path):
    cmd = [
        "ffprobe", "-v", "error", 
        "-show_entries", "stream=codec_type,duration", 
        "-of", "json", media_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(res.stdout)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio" and "duration" in stream:
                return float(stream["duration"])
    except Exception:
        pass
    return None

def _write_video_ffmpeg(output_frames, out_path, fps, height, width, logger, audio_path=None):
    """
    Encode output frames to an H.264 MP4 using a piped FFmpeg subprocess.
    Frames are written as raw BGR bytes; FFmpeg converts BGR→YUV and encodes
    with libx264 at CRF=23 (visually lossless at roughly half the mp4v size).
    Also syncs audio by forcing a constant 24fps timeline and stretching audio via atempo.
    """
    audio_duration = _get_audio_info(audio_path) if audio_path else None
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]

    has_audio = audio_duration is not None
    filter_complex = None
    
    if has_audio:
        video_duration = len(output_frames) / float(fps)
        drift = abs(video_duration - audio_duration)
        logger.info(f"Video duration: {video_duration:.3f}s, Audio duration: {audio_duration:.3f}s. Drift: {drift:.3f}s")
        
        cmd.extend(["-i", audio_path])
        
        if drift > 0.1:  # Threshold of 100ms
            ratio = audio_duration / video_duration if video_duration > 0 else 1.0
            logger.info(f"Applying atempo filter with ratio {ratio:.4f} to fix drift.")
            
            # atempo supports 0.5 to 100.0, we chain them if ratio < 0.5
            if ratio < 0.5:
                # E.g. ratio = 0.25 -> atempo=0.5,atempo=0.5
                num_filters = int(np.ceil(np.log(ratio) / np.log(0.5)))
                actual_ratio = ratio ** (1.0 / num_filters)
                chain = ",".join([f"atempo={actual_ratio}"] * num_filters)
                filter_complex = f"[1:a]{chain}[outa]"
            else:
                ratio = min(100.0, ratio) # cap at 100
                filter_complex = f"[1:a]atempo={ratio}[outa]"
    
    cmd.extend([
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "medium",
        "-g", str(fps), 
        "-movflags", "+faststart"
    ])
    
    if has_audio:
        if filter_complex:
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "0:v:0", "-map", "[outa]"])
        else:
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])
            
    cmd.append(out_path)
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for i, out_frame in enumerate(output_frames):
        if out_frame.frame_data is not None:
            frame_img = out_frame.frame_data
        else:
            # Synthetic frame – interpolate between neighbouring real frames
            prev_img = _find_neighbour(output_frames, i, direction=-1)
            next_img = _find_neighbour(output_frames, i, direction=+1)

            if prev_img is not None and next_img is not None:
                prev_idx = _find_neighbour_idx(output_frames, i, -1)
                next_idx = _find_neighbour_idx(output_frames, i, +1)
                total_gap = next_idx - prev_idx
                alpha = (i - prev_idx) / total_gap if total_gap > 0 else 0.5
                frame_img = cv2.addWeighted(prev_img, 1 - alpha, next_img, alpha, 0)
            elif prev_img is not None:
                frame_img = prev_img
            elif next_img is not None:
                frame_img = next_img
            else:
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)

        proc.stdin.write(frame_img.tobytes())

    proc.stdin.close()
    proc.wait()
    logger.info(f"FFmpeg encoder finished (exit code {proc.returncode}).")

def _find_neighbour(frames, idx, direction):
    """Walk in `direction` from `idx` and return the first real frame_data."""
    j = idx + direction
    while 0 <= j < len(frames):
        if frames[j].frame_data is not None:
            return frames[j].frame_data
        j += direction
    return None


def _find_neighbour_idx(frames, idx, direction):
    j = idx + direction
    while 0 <= j < len(frames):
        if frames[j].frame_data is not None:
            return j
        j += direction
    return idx


def _group_and_save_clips(output_frames, output_dir, config, fps, height, width, logger):
    grouping_cfg = config.get("clip_grouping", {}) if isinstance(config, dict) else {}
    if grouping_cfg.get("enabled", True) is False:
        logger.info("Clip grouping disabled by config.")
        return

    clusters = group_frames(
        raw_frames=output_frames,
        effective_context_limit=float(grouping_cfg.get("effective_context_limit", 1200.0)),
        base_merge_threshold=float(grouping_cfg.get("base_merge_threshold", 0.55)),
        default_visual_token_cost=float(grouping_cfg.get("default_visual_token_cost", 30.0)),
        entity_token_cost=float(grouping_cfg.get("entity_token_cost", 2.0)),
        subtitle_token_cost=float(grouping_cfg.get("subtitle_token_cost", 0.5)),
        max_drop_ratio=float(grouping_cfg.get("max_drop_ratio", 0.5)),
    )

    if not clusters:
        logger.info("Clip grouping produced 0 clips.")
        return

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    manifest = []
    for i, cluster in enumerate(clusters):
        start_idx = max(0, cluster.boundary_frames[0].index)
        end_idx = min(len(output_frames) - 1, cluster.boundary_frames[1].index)
        if end_idx < start_idx:
            continue

        clip_frames = output_frames[start_idx : end_idx + 1]
        clip_path = os.path.join(clips_dir, f"clip_{i:04d}.mp4")
        _write_video_ffmpeg(
            output_frames=clip_frames,
            out_path=clip_path,
            fps=fps,
            height=height,
            width=width,
            logger=logger,
        )

        manifest.append(
            {
                "clip_index": i,
                "cluster_id": cluster.id,
                "start_frame_index": start_idx,
                "end_frame_index": end_idx,
                "start_time_sec": start_idx / float(fps),
                "end_time_sec": end_idx / float(fps),
                "frame_count": len(clip_frames),
                "cluster_frame_count": len(cluster.frames),
                "token_cost": calculate_token_cost(
                    cluster,
                    entity_token_cost=float(grouping_cfg.get("entity_token_cost", 2.0)),
                    subtitle_token_cost=float(grouping_cfg.get("subtitle_token_cost", 0.5)),
                ),
                "score_profile": cluster.cluster_score_profile,
                "path": clip_path,
            }
        )

    manifest_path = os.path.join(clips_dir, "clips.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    logger.info("Clip grouping complete. Wrote %d clips to %s", len(manifest), clips_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the 24 FPS Semantic Video Compressor"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Directory to save output data",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Optional path to custom audio track. Defaults to the source video audio."
    )
    args = parser.parse_args()

    run(args.video_path, args.output_dir, args.audio_path)
