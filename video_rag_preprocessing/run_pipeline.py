import argparse
import logging
import os
import json
from pipeline import load_config, VideoIngestor, FeatureExtractor, CompressorEngine

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("SemanticCompressor")

def run(video_path: str, output_dir: str):
    logger = setup_logging()
    logger.info(f"Starting Semantic Compressor for video: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    config = load_config()
    logger.info("Configuration loaded.")
    
    ingestor = VideoIngestor(video_path=video_path, native_fps=30.0, duration=5.0)
    extractor = FeatureExtractor(
        clip_dim=config["dimensions"]["clip"],
        dino_dim=config["dimensions"]["dino"]
    )
    engine = CompressorEngine(config=config)
    
    logger.info("Pipeline components initialized. Commencing frame streaming...")
    
    frame_count = 0
    for frame_data, timestamp in ingestor.stream_frames():
        native_frame = extractor.process_frame(frame_data, timestamp)
        engine.push(native_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            logger.info(f"Processed {frame_count} native frames. Compressor time target: {engine.target_time:.2f}s")
    
    # Flush remaining buffers
    output_frames = engine.finalize()
    logger.info(f"Emission complete. Generated {len(output_frames)} standardized 24 FPS semantic output frames.")
    
    # Save dummy metrics dump
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
            "scores": f.scores
        })
        
    with open(dump_path, 'w') as f:
        json.dump(payload, f, indent=2)
        
    logger.info(f"Saved execution telemetry to {dump_path}. Included {syn_count} synthetic gap-fills.")
    logger.info("Pipeline executed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 24 FPS Semantic Video Compressor")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Directory to save output data")
    args = parser.parse_args()
    
    run(args.video_path, args.output_dir)
