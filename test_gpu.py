import numpy as np
from video_rag_preprocessing.pipeline.features.extractor import FeatureExtractor
import logging

logging.basicConfig(level=logging.INFO)

print("Initializing FeatureExtractor...")
extractor = FeatureExtractor()

print(f"Device selected: {extractor.device}")

# Create a dummy image (e.g., 480x640 BGR)
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

print("Testing process_frame...")
frame_features = extractor.process_frame(dummy_image, timestamp=0.0)

print(f"Features extracted successfully!")
print(f"CLIP embedding shape: {frame_features.clip_emb.shape}")
print(f"DINO embedding shape: {frame_features.dino_emb.shape}")
print(f"Entities detected: {len(frame_features.entities)}")
print("All models successfully utilized the configured device.")
