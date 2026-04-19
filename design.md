# VideoGraphRAG: Multimodal Video RAG System

A Local-First, Multimodal, Two-Layer Knowledge Graph Video RAG System with Hybrid Cloud Augmentation and Grounded Semantic Reasoning.

## 1. Overview

**Problem Statement**
Standard single-layer knowledge graphs and embedding-based RAG systems struggle with complex reasoning over video content. They lack the ability to effectively separate high-level multi-hop semantic reasoning from low-level frame-accurate video grounding.

**Goals**
Transform raw video into a structured, queryable set of representations to enable correct, interpretable multi-hop reasoning over visual content, while ensuring all answers remain completely grounded in the original video clips.

**High-Level Approach**
The system processes video into a two-layer knowledge graph (Grounding and Reasoning layers) using a split multimodal pipeline (vision and audio processed separately, then fused). It runs locally by default but can optionally utilize Azure for augmentation.

## 2. Core Design Decisions

- **Local-First Pipeline**: Optimized for local execution to ensure cost-safety and complete architectural control without mandatory cloud dependencies.
- **Hybrid Azure Usage**: Azure Video Indexer is used as an optional, non-critical augmentation layer for benchmarking or handling edge cases (due to credit limits).
- **Two-Layer Knowledge Graph Architecture**: Strictly separates multi-hop semantic reasoning (Layer 2) from factual, clip-level grounding (Layer 1) to maximize search efficiency and interpretability.
- **Separated Multimodal Processing**: Audio and visual features are processed independently with specialized models, then fused, ensuring higher accuracy over single unified multimodal models which can hallucinate under local compute constraints.
- **Temporal Clip Segmentation**: Enforces strict constraints bounds of 3–6 seconds per clip to maintain temporal rhythm, bounded token costs, and semantic coherence.

## 3. System Architecture

**End-to-End Pipeline Flow**

1. **Video Ingestion**: Raw video undergoes temporal frame extraction.
2. **Preprocessing**: Extracted frames are grouped into 3-6 second clips via semantic embedding similarity and token budget constraints.
3. **Multimodal Extraction**: Clips are passed to the Local Vision Pipeline (VLMs) and Local Audio Pipeline (Whisper/ASR) to extract entities, actions, and scenes.
4. **Fusion**: Audio and visual modality outputs are aligned temporally.
5. **Knowledge Graph Construction**:
   - **Layer 1 (Clip KG)** is built for fast retrieval and grounding.
   - **Layer 2 (Semantic KG)** models high-level abstractions, mapping actions and entities.
6. **Query & Retrieval**: User queries traverse Layer 2 for multi-hop reasoning, mapping back to Layer 1 for clip-level grounding and context generation.

## 4. Video Processing Pipeline

**Frame Extraction & Scoring**

- Frames are extracted and stored as `FrameNode` representations.
- Each node tracks embeddings, sequence index, token cost, and multiple scores (semantic, motion, entity continuity).

**Clip Grouping Logic**

- Uses a greedy, hierarchical algorithm driven by merge affinity metrics.
- Defines hard scene boundaries and aborts merge if cosine similarity falls below `0.2`.
- Combines visual frame costs, subtitle costs, and weighted entity frequencies to compute maximum token limits.

**Constraints**

- **Duration Constraints**: Strict boundaries between 3–6 seconds (forced merge below minimum, blocked merge above maximum).
- **Adaptive Squeeze Constraint**: When token budgets are exceeded, less semantically important intermediate frames are dynamically iteratively removed while preserving first, last, and continuity frames.

## 5. Vision Processing

**Model Selection**

- Qwen3.5-VL (9B) is primarily used for localized, structured inference.

**Outputs**

- **Entities**: Objects, individuals, key items.
- **Actions**: Physical movements, interactions.
- **Scenes**: High-level semantic environment descriptions.
- **Summaries**: Structured textual overviews of clip contents.

**Design Reasoning**

- Executing a high-accuracy, controllable 9B parameter VLM locally guarantees JSON compliance for deterministic downstream parsing, easily scaling into knowledge graph ingestion while staying under memory boundaries.

## 6. Audio Processing

**Model Selection**

- Transcription handled by Cohere-Transcribe or Whisper model variants for high-quality audio extraction.

**Transcript Handling**

- Raw audio tracks are parsed into deeply timestamp-aligned textual transcripts.

**Textual Extraction**

- Extracted transcripts are parsed by an LLM to derive explicit spoken entities, locations, and actions complementing visual data.

## 7. Multimodal Fusion

- Audio transcript metadata and visual VLM metadata are chronologically synchronized using explicit start/end timestamps.
- Inferred visual actions and extracted textual concepts are unified.
- Yields a single, dense semantic representation matrix per original clip interval.

## 8. Knowledge Graph Design

### Layer 1: Clip Graph (Grounding Layer)

- **Nodes**: Represent specific temporal clip segments. Stores `clip_id`, timestamps, generated summaries, embeddings, and inferred scenes.
- **Edges**: Encodes chronographic continuity (temporal adjacency), semantic closeness (embedding similarity), and intersecting data limits (entity overlap).
- **Purpose**: Operates functionally as the direct retrieval mechanism allowing system responses to map firmly back to truth sources (video segments).

### Layer 2: Semantic Graph (Reasoning Layer)

- **Nodes**: Purely abstract concepts (Entities, Locations, Scenes, Categories). Optional representation of fixed timestamp anchors.
- **Edges**: Defines hard relational facts acting as edges: `(Entity) --[action]--> (Entity)`, or `(Entity) --[located_in]--> (Scene)`. Examples: `person → buys → item`.
- **Purpose**: Engineered for logical multi-hop analysis, isolating query decomposition and context connections away from raw video data limits.

### Cross-Layer Connections

- Strict mapping connects each Semantic Node (Layer 2) downward into overlapping Clip Nodes (Layer 1).
- Allows top-down query execution: reason over abstract connections globally, answer deterministically with local clip intervals.

## 9. Query Execution Flow

1. **Query Alignment**: Input query is processed, generating embeddings and determining query intent/entities.
2. **Semantic Traversal (Layer 2)**: Entities and actions mapped to graph structure. System executes relations and traverses matching network nodes to achieve multi-hop deduction.
3. **Clip Identification**: Abstract results yield direct mapping pointers into Layer 1 logic.
4. **Hardware Retrieval (Layer 1)**: Matching clips and temporal blocks retrieved using graph intersections.
5. **Answer Generation**: Summarized content, raw frames, and structured contexts synthetically compiled to execute grounded user response generation.

## 10. Azure Integration (Alternate)

**Where it is used**

- Functions as an optional, separate processing branch operating Azure AI Video Indexer.

**Why it is NOT a dependency**

- Due to strict processing credit constraints (~100 limitations), hard dependency avoids system breakage or paywalls during core development loops.

**How it augments local outputs**

- Deployed selectively on critical intervals/edge cases as a benchmarking validation metric to dynamically enrich missing metadata nodes or verify correctness of Qwen-based entity/action generation.

## 11. Constraints and Trade-Offs

- **Memory Limits**: Architected to function under macOS M1 Pro strict boundaries (≤15GB workable RAM limit), dictating strict split constraints over continuous memory loads.
- **Model Size Decisions**: Chosen models (e.g. 9B VLMs) are optimized balances between factual capabilities and restrictive memory thresholds without needing heavy external compute APIs.
- **Local vs Cloud Trade-offs**: Emphasizes local environments prioritizing control and strict financial ceilings, trading off higher speed cloud multi-modal compute for privacy and custom format parsing constraints.
- **Accuracy vs Control**: Uses highly deterministic structured extraction steps to provide massive scaling control to graphs, potentially losing some nuanced un-structured visual context as a functional trade-off.

## 12. Future Improvements

- **Enhanced Entity & Action Extraction**: Migrate to refined extraction schema parameters and dynamic tracking identity matching to persist specific person/object IDs reliably.
- **Learning-Based Segmentation**: Swap greedy hierarchical merge frameworks out for global dynamically programmed temporal distribution constraints.
- **Better Token Estimation**: Overhaul naive string cost equations with raw LLM-specific tiktoken encoders.
- **Graph Structure Evolution**: Replace forced clip sub-sampling compression mechanics entirely with generative summarization, allowing perfect continuous temporal metadata encoding.
