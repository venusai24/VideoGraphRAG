# VideoGraphRAG

Graph-Based Temporal Reasoning for Video Retrieval — Transforming raw video into semantically-rich, graph-indexed representations.

---

## 📽️ Core Component: Semantic Video Preprocessing Pipeline
A production-grade, modular "Semantic Compressor" designed to extract high-signal, non-redundant representations from video. This pipeline is the foundational ingestion layer for Video Retrieval-Augmented Generation (Video RAG) systems.

### 🎯 Objective

To transform high-bitrate, redundant raw video into a compact stream of **OutputFrames** that preserve:

1. **Semantic Diversity**: Novel events and objects are prioritized.
2. **Visual Clarity**: Sharp, high-quality representative frames are selected over blurry ones.
3. **Temporal Continuity**: Smooth transitions and structural consistency for downstream reconstruction.

---

### 🚀 Architectural Pillars

#### 1. Target-Synchronous Emission (TSE)

Unlike traditional keyframe extractors that sample at fixed intervals, this pipeline maintains a **Target Timeline** (default: 24 FPS). It decouples input jitter from output consistency using an adaptive `EmissionBuffer`. If a temporal slot lacks high-quality native frames, the system can mark it for synthetic interpolation.

#### 2. Multi-Modal Feature Engineering

The pipeline extracts a high-dimensional feature set for every native frame:

* **CLIP (Semantics)**: Captures high-level concepts and text-image alignment.
* **DINOv2 (Structure)**: Captures spatial continuity and visual consistency.
* **YOLOv8 (Entities)**: Detects and tracks human-level objects with bounding box persistence.
* **Optical Flow**: Measures motion dynamics to identify high-activity segments.

#### 3. Dynamic Scoring Surface

The scoring engine adapts to video conditions in real-time. Using **Rolling Percentile** trackers, it normalizes metrics like blur and motion relative to the recent history of the specific video being processed, rather than using hard-coded global thresholds.

#### 4. Multi-Tier Memory Systems

* **EWMA (Short-Term)**: Maintains a running context of recent semantic states.
* **FAISS (Long-Term)**: A vector memory bank used to detect global novelty across the entire video duration.

---

### 🧠 The Pipeline Engine (`CompressorEngine`)

The engine operates on a sliding-window evaluation loop:

#### Window-Based Selection

For each target timestamp $T$, the engine gathers all native frames within a window $[T - \Delta t, T + \Delta t]$. It calculates a holistic score for each candidate:

$$Score = Gate \cdot S_{semantic} + (1 - Gate) \cdot S_{blur} - Penalties$$

* **$S_{semantic}$**: Composite score of Novelty (CLIP), Motion (Flow), and Entity Delta (YOLO).
* **$S_{blur}$**: Composite score of Sharpness (Laplacian), Motion, and Entity Delta.
* **$Gate$**: A sigmoid switch ($\sigma$) controlled by semantic novelty. High novelty opens the gate for semantic priority; low novelty (stagnant scenes) prioritizes technical sharpness.
* **$Penalties$**: Deductions for lack of visual continuity (DINOv2 distance) and lack of distinctiveness within the current window.

---

### 🛠️ Core Modules

| Module | Responsibility |
| :--- | :--- |
| **`features/extractor.py`** | GPU-accelerated extraction of CLIP, DINOv2, and YOLOv8. |
| **`memory/tracker.py`** | EWMA, FaissMemoryBank, and RollingPercentile logic. |
| **`scoring/scorer.py`** | Implementation of the gated multi-modal scoring function. |
| **`selection/window.py`** | Orchestrates window evaluation and the selection loop. |
| **`postprocess/emission.py`** | Manages the `EmissionBuffer` for jitter-free output. |

---

### ⚙️ Configuration
The pipeline is highly configurable via `config/`:
```python
pipeline:
  target_fps: 24             # Consistent output frame rate
  target_window_sec: 0.083   # Evaluation window size (~2 frames at 24fps)

scoring:
  weights:
    semantic: 1.0           # Priority for novel content
    blur: 0.5               # Priority for technical quality
    motion: 0.8             # Prioritize activity
    entity: 0.4             # Prioritize object changes
```

---

### 📦 Output Characteristics
The pipeline produces a deterministic JSON and image sequence output:
*   **85–95% Semantic Compression**: Massive reduction in storage and compute costs for downstream RAG.
*   **Graph-Ready Enrichment**: Each frame includes CLIP/DINO embeddings and YOLO entity lists.
*   **Deterministic Metadata**: Timestamps are aligned to a perfect 24 FPS grid, enabling frame-accurate retrieval and reconstruction.

---

### 🏁 Getting Started

1. **Install Dependencies**:

   ```bash
   pip install torch torchvision ultralytics transformers numpy opencv-python faiss-cpu
   ```

2. **Run the Pipeline**:

   ```python
   from video_rag_preprocessing.pipeline.selection.window import CompressorEngine
   from video_rag_preprocessing.pipeline.features.extractor import FeatureExtractor

   # Initialize engine and extractor
   engine = CompressorEngine()
   extractor = FeatureExtractor()

   # Process frames...
   ```

3. **Verify Output**:

   Check the `outputs/` directory for `scores.json` and the compressed keyframe sequence.
scene segmentation)
Deep learning (CLIP + DINOv2 embeddings)
Efficient algorithms (temporal grouping)