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

    def push_empty(self, target_time: float, history: List[np.ndarray], clip_dim: int = 512, dino_dim: int = 384):
        self.emission_buffer.append(OutputFrame(
            target_timestamp=target_time,
            native_timestamp=None,
            clip_emb=np.zeros(clip_dim), 
            dino_emb=np.zeros(dino_dim),
            is_synthetic=True,
            scores={}
        ))
        self._resolve_emissions(history)

    def _interpolate_gaps(self, history: List[np.ndarray]):
        if not self.emission_buffer or self.emission_buffer[-1].is_synthetic:
            return
            
        last_real_idx = -1
        for i in range(len(self.emission_buffer) - 2, -1, -1):
            if not self.emission_buffer[i].is_synthetic:
                last_real_idx = i
                break
                
        if last_real_idx != -1:
            num_gaps = len(self.emission_buffer) - 1 - last_real_idx - 1
            if num_gaps > 0:
                prev_real = self.emission_buffer[last_real_idx]
                next_real = self.emission_buffer[-1]
                for i in range(1, num_gaps + 1):
                    alpha = i / (num_gaps + 1.0)
                    syn_frame = self.emission_buffer[last_real_idx + i]
                    
                    syn_frame.clip_emb = (1 - alpha) * prev_real.clip_emb + alpha * next_real.clip_emb
                    syn_frame.dino_emb = (1 - alpha) * prev_real.dino_emb + alpha * next_real.dino_emb
                    
                    n_clip = np.linalg.norm(syn_frame.clip_emb)
                    if n_clip > 1e-8:
                        syn_frame.clip_emb /= n_clip
                    n_dino = np.linalg.norm(syn_frame.dino_emb)
                    if n_dino > 1e-8:
                        syn_frame.dino_emb /= n_dino
                    
                    syn_frame.scores = {"total": 0.0, "lerped": 1.0}
                    
                    dist_from_end = len(self.emission_buffer) - 1 - (last_real_idx + i)
                    hist_idx = len(history) - 1 - dist_from_end
                    if 0 <= hist_idx < len(history):
                        history[hist_idx] = syn_frame.dino_emb

    def _resolve_emissions(self, history: List[np.ndarray]):
        self._interpolate_gaps(history)
        
        if len(self.emission_buffer) < 4:
            return
            
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
            res = smoothed[2]
            n_res = np.linalg.norm(res) + 1e-8
            self.smoothed_output_stream[-3].clip_emb = res / n_res

    def finalize(self) -> List[OutputFrame]:
        # Handle trailing gaps
        for i in range(len(self.emission_buffer)):
            if self.emission_buffer[i].is_synthetic and i > 0 and not self.emission_buffer[i-1].is_synthetic:
                self.emission_buffer[i].clip_emb = self.emission_buffer[i-1].clip_emb.copy()
                self.emission_buffer[i].dino_emb = self.emission_buffer[i-1].dino_emb.copy()
                self.emission_buffer[i].scores = {"total": 0.0, "lerped": 1.0, "trailing": 1.0}
                
        while self.emission_buffer:
            released = self.emission_buffer.popleft()
            self.smoothed_output_stream.append(released)
            self._apply_conditional_smoothing()
        return self.smoothed_output_stream
