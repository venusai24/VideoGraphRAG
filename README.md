# 🎞️ VideoGraphRAG: Semantic Preprocessing Pipeline

This repository hosts a production-grade **Semantic Video Compressor** dedicated for graph-based video reasoning. It transforms raw, native-FPS video streams into a standardized, dense, and continuous **24 FPS** representational sequence by analyzing spatial semantics, motion dynamics, and absolute blur without resorting to naive threshold drops.

## 🚀 Key Architecture Features
*   **Adaptive Gated Scoring:** Abandons standard mathematical thresholds by implementing Sigmoid soft-gating weighted mathematically between structural image quality and novelty. 
*   **Continuous 24 FPS Streaming:** Guarantees downstream processing avoids time-jumping using an identical bounded step size ($1/12\text{s}$ Target Window, $1/24\text{s}$ Stride), safely substituting native dropped frames via semantic embeddings interpolation.
*   **Robust Normalization Limits:** `p25-p75` tracked bounds actively regularize dynamic action movies and static dialogue identically without exploding gradient suppression. 
*   **O(1) Statistics & FAISS Indexing**: Dual-memory structures matching localized 2-second EWMA checks against permanent historical indexing searches targeting long-term scene dynamics dynamically.

## 📁 Repository Modular Structure
The codebase strictly enforces the separation of logical processing boundaries into deterministic isolated namespaces under `/pipeline/`:

*   `/ingestion/` - Wrapper mapping real upstream CV2/PyAV frames against unified timestamp intervals.
*   `/features/` - Wrappers surrounding expensive CLIP / DINO / YOLO inference mapping arrays efficiently. 
*   `/memory/` - Controls EWMA temporal drift trackers, percentile states mapping $O(1)$, and FAISS index loops.
*   `/scoring/` - Hosts mathematical scaling combinations, penalty bounds, and distance norms (`Cosine DINO / CLIP`, IoU Denoising).
*   `/selection/` - Sliding target evaluations mapping continuous streams against bounded frames executing multi-step consistency checks.
*   `/postprocess/` - Hosts the explicit 3-window lookahead emission queues interpolating synthetic bounds and executing 1D temporal Gaussian continuous sweeps.

## 💻 Running the Pipeline

### Setup Constraints
Ensure you have PyTorch, NumPy, and FAISS successfully loaded.
```bash
pip install -r requirements.txt 
```
### Execution
Trigger pipeline evaluations directly from the orchestrator proxy:
```bash
python video_rag_preprocessing/run_pipeline.py --video_path /test_data/sample.mp4 --output_dir ./outputs/
```

## 📊 Standard Outputs Breakdown
The script generates explicitly modeled output records representing individual 24 FPS timestamps identically:
1.  **Interpolated Boundaries**: The model explicitly tracks `"is_synthetic": True` arrays when native payloads are missing. 
2.  **Telemetry Reporting**: An isolated JSON payload natively exports components containing metrics (`semantic`, `blur`, `motion`, `diversity`). Useful downstream to trace execution pathways inside reasoning engines directly.