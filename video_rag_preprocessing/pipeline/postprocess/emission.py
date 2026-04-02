import numpy as np
import collections
from typing import List, Optional
from scipy.ndimage import gaussian_filter1d
from ..models import OutputFrame

class EmissionBuffer:
    def __init__(self, smoothing_threshold: float = 0.85):
        self.smoothing_threshold = smoothing_threshold
        self.emission_buffer = collections.deque()
        self.smoothed_output_stream = []

    def push_populated(self, frame: OutputFrame, history: List[np.ndarray]):
        self.emission_buffer.append(frame)
        self._resolve_emissions(history)

    def push_empty(self, target_time: float, history: List[np.ndarray]):
        self.emission_buffer.append(OutputFrame(
            target_timestamp=target_time,
            native_timestamp=None,
            clip_emb=np.zeros(0), 
            dino_emb=np.zeros(0),
            is_synthetic=True,
            scores={}
        ))
        self._resolve_emissions(history)

    def _resolve_emissions(self, history: List[np.ndarray]):
        if len(self.emission_buffer) < 3:
            return
            
        prev_f = self.emission_buffer[0]
        curr_f = self.emission_buffer[1]
        next_f = self.emission_buffer[2]
        
        if curr_f.is_synthetic and prev_f.is_synthetic is False and next_f.is_synthetic is False:
            curr_f.clip_emb = (prev_f.clip_emb + next_f.clip_emb) / 2.0
            curr_f.dino_emb = (prev_f.dino_emb + next_f.dino_emb) / 2.0
            
            curr_f.scores = {"total": 0.0, "lerped": 1.0}
            
            if len(history) >= 2:
                history[-2] = curr_f.dino_emb
            
        released = self.emission_buffer.popleft()
        self.smoothed_output_stream.append(released)
        self._apply_conditional_smoothing()

    def _apply_conditional_smoothing(self):
        if len(self.smoothed_output_stream) < 5:
            return
            
        embs = np.array([f.clip_emb for f in self.smoothed_output_stream[-5:]])
        
        norms = np.linalg.norm(embs, axis=1) + 1e-8
        norm_embs = embs / norms[:, np.newaxis]
        sims = np.dot(norm_embs, norm_embs.T)
        avg_sim = np.mean(sims)
        
        if avg_sim > self.smoothing_threshold:
            smoothed = gaussian_filter1d(embs, sigma=0.5, axis=0)
            self.smoothed_output_stream[-3].clip_emb = smoothed[2]

    def finalize(self) -> List[OutputFrame]:
        while self.emission_buffer:
            released = self.emission_buffer.popleft()
            self.smoothed_output_stream.append(released)
            self._apply_conditional_smoothing()
        return self.smoothed_output_stream
