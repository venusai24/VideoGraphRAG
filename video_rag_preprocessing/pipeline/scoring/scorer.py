import numpy as np
from typing import List, Dict, Any
from ..models import NativeFrame

def clip_norm(val: float, p25: float, p75: float, eps: float = 1e-5) -> float:
    denom = (p75 - p25) + eps
    return np.clip((val - p25) / denom, 0.0, 2.0)

def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    if emb1 is None or emb2 is None:
        return 0.0
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    sim = np.dot(emb1.flatten(), emb2.flatten()) / (norm1 * norm2)
    return 1.0 - sim

def calculate_entity_delta(entities1: List[Any], entities2: List[Any], eps: float = 1e-5) -> float:
    l1, l2 = len(entities1), len(entities2)
    total = l1 + l2
    if total == 0:
        return 0.0
    delta = abs(l1 - l2)
    return delta / (total + eps)

class Scorer:
    def __init__(self, weights: Dict[str, float]):
        total_w = sum(weights.values())
        self.w = {k: v / total_w for k, v in weights.items()}
        self.eps = 1e-5

    def score_frame(self, 
                    frame: NativeFrame,
                    n_hat: float, 
                    q_hat: float, 
                    m_hat: float, 
                    e_hat: float, 
                    c_val: float, 
                    d_val: float,
                    tau: float,
                    temperature: float) -> Dict[str, float]:
        
        gate_arg = (n_hat - tau) / (temperature + self.eps)
        gate = 1.0 / (1.0 + np.exp(-gate_arg))

        s_sem = (self.w.get('semantic', 0.0) * n_hat) + \
                (self.w.get('motion', 0.0) * m_hat) + \
                (self.w.get('entity', 0.0) * e_hat)

        s_blur = (self.w.get('blur', 0.0) * q_hat) + \
                 (self.w.get('motion', 0.0) * m_hat) + \
                 (self.w.get('entity', 0.0) * e_hat)

        s_comp = (gate * s_sem) + ((1.0 - gate) * s_blur)
        
        penalties = (self.w.get('consistency', 0.0) * c_val) + \
                    (self.w.get('diversity', 0.0) * d_val)

        final_score = s_comp - penalties

        return {
            "semantic": s_sem,
            "blur": s_blur,
            "motion": m_hat,
            "entity": e_hat,
            "consistency": c_val,
            "diversity": d_val,
            "gate": gate,
            "total": final_score
        }
