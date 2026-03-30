# VideoGraphRAG
 Graph-Based Temporal Reasoning for Video Retrieval - Using Graph-Retrieval to retrieve information from videos


## Preprocessing Phase
Video RAG Preprocessing Pipeline

A production-grade, modular pipeline for extracting high-quality, non-redundant keyframes from videos, optimized for Video Retrieval-Augmented Generation (Video RAG) systems.

This pipeline is designed for offline preprocessing, ensuring that downstream retrieval systems operate on semantically meaningful and compact visual representations.

🎯 Objective

Given an input video, the pipeline:

Samples frames at 12 FPS uniformly
Removes blurry / low-quality frames early
Detects scene boundaries
Groups frames based on semantic similarity
Selects one representative frame per group
Stores metadata and embeddings for retrieval
🧠 Why This Pipeline Exists

Raw videos are:

Highly redundant (adjacent frames are similar)
Noisy (motion blur, compression artifacts)
Expensive to process during retrieval

This pipeline reduces video data by 70–85%, while preserving:

Semantic diversity
Visual clarity
Retrieval relevance
🧱 Architecture Overview
Video Input
   ↓
Frame Sampling (12 FPS)
   ↓
Blur Filtering
   ↓
Scene Detection
   ↓
Embedding Extraction (CLIP + DINOv2)
   ↓
Similarity-Based Grouping
   ↓
Representative Frame Selection
   ↓
Storage (JSON + Images)

Each stage is modular, enabling independent optimization and testing.

⚙️ Pipeline Stages
1. 🎞️ Frame Sampling

Goal: Uniformly extract frames at 12 FPS regardless of input video FPS.

Key Design Choices:

Uses timestamp-based sampling (not frame skipping)
Ensures consistency across videos with different frame rates
Avoids temporal bias

Output:

[(frame, timestamp)]
2. 🧹 Blur Filtering

Goal: Remove low-quality frames early to save compute downstream.

Method:

Uses Variance of Laplacian to measure sharpness

Behavior:

Frames below threshold (default: 120) are discarded
Grayscale conversion is done once for efficiency

Why early filtering matters:

Reduces embedding computation cost
Improves final keyframe quality

Output:

(frame, timestamp, blur_score)
3. 🎬 Scene Detection

Goal: Split video into semantically coherent segments.

Method:

Detects boundaries using:
Embedding similarity drop OR
Histogram differences

Logic:

If similarity between consecutive frames falls below threshold → new scene

Design Constraints:

Lightweight (no heavy external libraries)
Fast enough for large-scale preprocessing

Output:

[
  [frame1, frame2, ...],  # Scene 1
  [frame3, frame4, ...],  # Scene 2
]
4. 🧠 Embedding Extraction

Goal: Represent frames numerically for similarity and retrieval.

Models Used:

CLIP → Cross-modal retrieval (text ↔ image)
DINOv2 → Visual semantic grouping

Key Features:

Batch processing (critical for performance)
GPU/CPU compatible
Embeddings are normalized

Output per frame:

{
  "embedding_clip": vector,
  "embedding_dino": vector
}
5. 🔗 Similarity-Based Grouping

Goal: Remove redundancy by grouping similar frames.

Algorithm: Temporal-Aware Grouping

For each frame:

Compare with:
Last frame in group (local continuity)
First frame in group (anchor constraint)

Conditions:

cosine(frame, last_frame) > t1 AND
cosine(frame, first_frame) > t2

Thresholds:

t1 = 0.92 → ensures smooth transitions
t2 = 0.88 → prevents drift within group

Why this works:

Avoids O(n²) comparisons
Maintains temporal consistency
Prevents long-term semantic drift

Output:

[
  [frame1, frame2, ...],  # Group 1
  [frame3, frame4, ...],  # Group 2
]
6. ⭐ Representative Frame Selection

Goal: Pick the best frame per group.

Scoring Function:

score = w1 * blur_score + w2 * centrality

Where:

blur_score → sharpness
centrality → similarity to other frames in group

Weights:

w1 = 0.6 → prioritize visual quality
w2 = 0.4 → ensure representativeness

Why this is important:

Avoids picking blurry frames
Ensures selected frame represents the group well

Output:

One keyframe per group
7. 💾 Storage

Goal: Persist outputs for downstream RAG systems.

Stored Data:

Each frame:

{
  "frame_id": int,
  "timestamp": float,
  "scene_id": int,
  "blur_score": float,
  "embedding_clip": [...],
  "embedding_dino": [...],
  "image_path": "path/to/image.jpg"
}

Additional Features:

Images saved with deterministic naming
JSON for easy indexing and retrieval
Compatible with vector databases
🔧 Configuration

All parameters are centralized:

FPS_SAMPLE = 12
BLUR_THRESHOLD = 120
SIM_T1 = 0.92
SIM_T2 = 0.88
W_BLUR = 0.6
W_CENTRALITY = 0.4
BATCH_SIZE = 32

This allows:

Easy tuning
Experiment reproducibility
Dataset-specific optimization
🚀 Pipeline Orchestration

The main pipeline executes:

video
 → sampling
 → blur filtering
 → scene detection
 → embedding extraction
 → grouping
 → selection
 → storage
Logging & Monitoring

At each stage, the pipeline logs:

Total sampled frames
Frames after blur filtering
Number of scenes
Number of groups
Final selected keyframes
⚡ Performance Considerations
Batch processing for embeddings (critical speedup)
Early filtering reduces unnecessary computation
Avoids redundant image copies
Linear-time grouping algorithm
📦 Output Characteristics

For a typical video:

70–85% frame reduction
High-quality, sharp keyframes
Strong semantic coverage
Ready for:
Vector DB indexing
Cross-modal retrieval (text ↔ video)
Graph-based RAG systems

This pipeline transforms raw video into a compact, high-signal representation by combining:

Signal processing (blur detection)
Computer vision (scene segmentation)
Deep learning (CLIP + DINOv2 embeddings)
Efficient algorithms (temporal grouping)