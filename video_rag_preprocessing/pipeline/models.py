import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List

@dataclass
class NativeFrame:
    """Represents a decoded native frame with extracted features."""
    timestamp: float
    clip_emb: np.ndarray
    dino_emb: np.ndarray
    blur_variance: float
    optical_flow_mag: float
    entities: List[Any]

@dataclass
class OutputFrame:
    """The final deterministic structure passed downstream."""
    target_timestamp: float           
    native_timestamp: Optional[float]
    clip_emb: np.ndarray              
    dino_emb: np.ndarray              
    is_synthetic: bool
    scores: Dict[str, float] = field(default_factory=dict)
