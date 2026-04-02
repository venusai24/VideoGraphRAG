import numpy as np
from typing import List, Any
from ..models import NativeFrame

class FeatureExtractor:
    """
    Simulated wrapper for extracting signals from raw frames.
    In production, this would batch process videos through Torch models.
    """
    def __init__(self, clip_dim: int = 512, dino_dim: int = 384):
        self.clip_dim = clip_dim
        self.dino_dim = dino_dim

    def process_frame(self, frame_data: Any, timestamp: float) -> NativeFrame:
        # Simulated Feature Extraction
        clip_emb = np.random.randn(self.clip_dim)
        dino_emb = np.random.randn(self.dino_dim)
        
        # L2 norm approximations
        clip_emb /= np.linalg.norm(clip_emb) + 1e-8
        dino_emb /= np.linalg.norm(dino_emb) + 1e-8
        
        blur_variance = np.random.uniform(5.0, 500.0)
        optical_flow_mag = np.random.uniform(0.0, 10.0)
        
        num_entities = np.random.randint(0, 10)
        entities = list(range(num_entities))

        return NativeFrame(
            timestamp=timestamp,
            clip_emb=clip_emb,
            dino_emb=dino_emb,
            blur_variance=blur_variance,
            optical_flow_mag=optical_flow_mag,
            entities=entities
        )
