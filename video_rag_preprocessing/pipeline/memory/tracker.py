import numpy as np
import collections
import bisect
import faiss

class EWMA:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.state = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = value.copy()
        else:
            self.state = self.alpha * value + (1 - self.alpha) * self.state
        return self.state

    def get(self) -> np.ndarray:
        return self.state

class RollingPercentile:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.queue = collections.deque(maxlen=window_size)
        self.sorted_list = []

    def update(self, value: float):
        if len(self.queue) == self.window_size:
            oldest = self.queue.popleft()
            idx = bisect.bisect_left(self.sorted_list, oldest)
            if idx < len(self.sorted_list) and self.sorted_list[idx] == oldest:
                self.sorted_list.pop(idx)
        
        self.queue.append(value)
        bisect.insort(self.sorted_list, value)

    def get_percentile(self, p: float) -> float:
        if not self.sorted_list:
            return 0.0
        idx = int((p / 100.0) * (len(self.sorted_list) - 1))
        return self.sorted_list[idx]

    def p25(self) -> float:
        return self.get_percentile(25)

    def p75(self) -> float:
        return self.get_percentile(75)

class FaissMemoryBank:
    def __init__(self, dim: int, max_elements: int = 10000):
        self.index = faiss.IndexFlatIP(dim)
        self.max_elements = max_elements
        self.count = 0

    def add(self, embedding: np.ndarray):
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        self.index.add(norm_emb.astype(np.float32).reshape(1, -1))
        self.count += 1

    def query_novelty(self, embedding: np.ndarray, top_k: int = 5) -> float:
        if self.count == 0:
            return 1.0
        
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        norm_emb = norm_emb.astype(np.float32).reshape(1, -1)
        
        k = min(top_k, self.count)
        similarities, indices = self.index.search(norm_emb, k)
        mean_sim = np.mean(similarities[0])
        return max(0.0, 1.0 - mean_sim)
