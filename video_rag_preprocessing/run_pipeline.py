import argparse
import logging
import os
import json
import cv2
import numpy as np

from pipeline import load_config, VideoIngestor, FeatureExtractor, CompressorEngine


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SemanticCompressor")


def run(video_path: str, output_dir: str):
    logger = setup_logging()
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, engine.fps, (width, height))

    for i, out_frame in enumerate(output_frames):
        if out_frame.frame_data is not None:
            writer.write(out_frame.frame_data)
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

            writer.write(frame_img)

    writer.release()
    logger.info(f"Pipeline executed successfully. Final video saved at: {out_video_path}")


# ── helpers ───────────────────────────────────────────────────────────

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
    args = parser.parse_args()

    run(args.video_path, args.output_dir)
