import numpy as np
from typing import List, Dict, Optional
from ..models import NativeFrame, OutputFrame
from ..memory.tracker import EWMA, RollingPercentile, FaissMemoryBank
from ..scoring.scorer import Scorer, clip_norm, cosine_distance, calculate_entity_delta
from ..postprocess.emission import EmissionBuffer
from ..config.loader import load_config

class CompressorEngine:
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()
            
        opts = config.get("pipeline", {})
        mem_opts = config.get("memory", {})
        dim_opts = config.get("dimensions", {})
        score_opts = config.get("scoring", {})

        self.fps = opts.get("target_fps", 24)
        self.stride_sec = opts.get("stride_sec", 1/24)
        self.window_sec = opts.get("target_window_sec", 1/12)
        self.eps = score_opts.get("eps", 1e-5)

        # Trackers
        self.ctx_short = EWMA(alpha=mem_opts.get("ewma_alpha", 0.3))
        self.entity_ewma = EWMA(alpha=0.3)
        self.bank_long = FaissMemoryBank(
            dim=dim_opts.get("clip", 512), 
            max_elements=mem_opts.get("faiss_max_elements", 10000)
        )
        
        w_size = mem_opts.get("percentile_window", 120)
        self.p_blur = RollingPercentile(window_size=w_size)
        self.p_mot  = RollingPercentile(window_size=w_size)
        self.p_nov  = RollingPercentile(window_size=w_size)

        self.scorer = Scorer(score_opts.get("weights", {}))
        self.emission = EmissionBuffer(
            smoothing_threshold=score_opts.get("conditional_smoothing_threshold", 0.85)
        )

        self.prev_native_entities = []
        self.output_history_dino = []
        self.target_time = 0.0
        self.pending_natives = []

    def push(self, frame: NativeFrame):
        self.pending_natives.append(frame)
        self._process_windows()

    def _process_windows(self):
        while self.pending_natives:
            latest_ts = self.pending_natives[-1].timestamp
            window_end = self.target_time + self.window_sec / 2.0
            
            if latest_ts >= window_end:
                self._evaluate_current_window()
                self.target_time += self.stride_sec
            else:
                break

    def _evaluate_current_window(self):
        w_start = self.target_time - self.window_sec / 2.0
        w_end = self.target_time + self.window_sec / 2.0
        
        candidates = [f for f in self.pending_natives if w_start <= f.timestamp < w_end]
        self.pending_natives = [f for f in self.pending_natives if f.timestamp >= w_start]

        if not candidates:
            self.emission.push_empty(self.target_time, self.output_history_dino)
            return
            
        best_frame = None
        best_score_val = -float('inf')
        best_scores_dict = {}

        for cand in candidates:
            self.p_blur.update(cand.blur_variance)
            self.p_mot.update(cand.optical_flow_mag)

        for i, cand in enumerate(candidates):
            q_hat = clip_norm(cand.blur_variance, self.p_blur.p25(), self.p_blur.p75(), self.eps)
            m_hat = clip_norm(cand.optical_flow_mag, self.p_mot.p25(), self.p_mot.p75(), self.eps)
            
            raw_ent = calculate_entity_delta(self.prev_native_entities, cand.entities, self.eps)
            e_hat = self.entity_ewma.update(np.array([raw_ent]))[0]
            self.prev_native_entities = cand.entities
            
            ctx_v = self.ctx_short.get()
            n_short = cosine_distance(cand.clip_emb, ctx_v) if ctx_v is not None else 1.0
            n_long = self.bank_long.query_novelty(cand.clip_emb)
            n_val = min(n_short, n_long)
            self.p_nov.update(n_val)
            
            n_hat = clip_norm(n_val, self.p_nov.p25(), self.p_nov.p75(), self.eps)

            c_val = 0.0
            if len(self.output_history_dino) > 0:
                dist1 = cosine_distance(cand.dino_emb, self._get_hist(1))
                dist5 = cosine_distance(cand.dino_emb, self._get_hist(5))
                dist10 = cosine_distance(cand.dino_emb, self._get_hist(10))
                c_val = 0.6 * dist1 + 0.3 * dist5 + 0.1 * dist10

            d_val = 0.0
            if len(candidates) >= 3:
                sims = [1.0 - cosine_distance(cand.clip_emb, other.clip_emb) 
                        for j, other in enumerate(candidates) if j != i]
                d_val = max(sims) if sims else 0.0

            tau = self.p_nov.p75()
            temperature = max(self.p_nov.p75() - self.p_nov.p25(), self.eps)

            scores = self.scorer.score_frame(
                cand, n_hat, q_hat, m_hat, e_hat, c_val, d_val, tau, temperature
            )

            if scores['total'] > best_score_val:
                best_score_val = scores['total']
                best_frame = cand
                best_scores_dict = scores

        self._emit_selected(best_frame, best_scores_dict)

    def _get_hist(self, steps_back: int) -> Optional[np.ndarray]:
        idx = max(0, len(self.output_history_dino) - steps_back)
        return self.output_history_dino[idx]

    def _emit_selected(self, frame: NativeFrame, scores: Dict[str, float]):
        self.ctx_short.update(frame.clip_emb)
        self.bank_long.add(frame.clip_emb)
        self.output_history_dino.append(frame.dino_emb)
        
        out = OutputFrame(
            target_timestamp=self.target_time,
            native_timestamp=frame.timestamp,
            clip_emb=frame.clip_emb.copy(),
            dino_emb=frame.dino_emb.copy(),
            is_synthetic=False,
            scores=scores
        )
        self.emission.push_populated(out, self.output_history_dino)

    def finalize(self) -> List[OutputFrame]:
        return self.emission.finalize()
