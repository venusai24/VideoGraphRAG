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

def _bbox_iou(box1: List[float], box2: List[float]) -> float:
    """Intersection-over-Union for two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


def calculate_entity_delta(entities1: List[Any], entities2: List[Any], eps: float = 1e-5) -> float:
    """
    Compute entity-change signal between two frames.
    Supports both simple (int/str) entity lists and full detection dicts
    with 'bbox' and 'class_id' keys (from YOLO).
    """
    # Fast-path: both empty → no change
    if not entities1 and not entities2:
        return 0.0
    # Fast-path: one empty → full change
    if not entities1 or not entities2:
        return 1.0

    # If entities are dicts with bbox/class_id, use IoU matching
    if isinstance(entities1[0], dict) and "bbox" in entities1[0]:
        matched = 0
        used = set()
        for e1 in entities1:
            best_iou = 0.0
            best_j = -1
            for j, e2 in enumerate(entities2):
                if j in used:
                    continue
                if e1.get("class_id") != e2.get("class_id"):
                    continue
                iou = _bbox_iou(e1["bbox"], e2["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou > 0.3:
                matched += 1
                used.add(best_j)
        total = max(len(entities1), len(entities2))
        return 1.0 - (matched / (total + eps))

    # Fallback: set-based comparison for simple entity lists
    set1 = set(entities1)
    set2 = set(entities2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    iou = intersection / (union + eps)
    return 1.0 - iou

class Scorer:
    def __init__(self, weights: Dict[str, float]):
        if not weights:
            weights = {
                "semantic": 1.0,
                "blur": 0.5,
                "motion": 0.8,
                "entity": 0.4,
                "consistency": 0.3,
                "diversity": 0.2
            }
        total_w = sum(weights.values())
        self.w = {k: v / total_w for k, v in weights.items()}
        self.eps = 1e-5

    def score_frame(self, 
                    cand: NativeFrame,
                    n_hat: float, 
                    q_hat: float, 
                    m_hat: float, 
                    e_hat: float, 
                    c_val: float, 
                    d_val: float,
                    tau: float,
                    temperature: float) -> Dict[str, float]:
        
        gate_arg = (n_hat - tau) / (temperature + self.eps)
        gate_arg = float(np.clip(gate_arg, -30.0, 30.0))
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
            "semantic": float(s_sem),
            "blur": float(s_blur),
            "motion": float(m_hat),
            "entity": float(e_hat),
            "consistency": float(c_val),
            "diversity": float(d_val),
            "gate": float(gate),
            "total": float(final_score)
        }
