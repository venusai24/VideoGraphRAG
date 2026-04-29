# VideoGraphRAG

> **Multi-hop video question answering powered by knowledge graphs and LLMs.**

---

## Overview

VideoGraphRAG is a production-grade Retrieval-Augmented Generation system that answers natural language questions about video content by combining:

- **Knowledge Graph Reasoning** — A two-layer Neo4j graph (Entity Graph + Clip Graph) enables structured, multi-hop traversal over video semantics.
- **Multimodal Evidence** — Transcripts, OCR, visual summaries, and raw video clips are fused to ground every answer.
- **LLM Orchestration** — Cerebras → Groq failover for query decomposition; Gemini for grounded answer generation with strict citation control.

Unlike flat RAG pipelines, VideoGraphRAG decomposes queries into graph traversal plans, resolves entities against a canonical knowledge base, and produces answers that cite specific video clips with timestamps — eliminating hallucination by design.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                           │
│  Video → Frame Extraction → Feature Scoring → Clip Selection   │
│  → Azure Video Indexer (transcript, OCR, scenes, keywords)     │
│  → Neo4j Graph Construction (Entity Graph + Clip Graph)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                              │
│                                                                 │
│  1. Query Decomposition (Cerebras/Groq LLM)                    │
│     → Structured execution plan with entity resolution          │
│                                                                 │
│  2. Graph Traversal (Beam-Search Engine)                        │
│     → Entity resolution → APPEARS_IN / NEXT / SHARES_ENTITY    │
│     → Diversity-aware pruning + keyword fallback                │
│                                                                 │
│  3. Answer Generation (Gemini)                                  │
│     → Tiered context window (high/mid/low priority clips)       │
│     → Strict JSON output with citation validation               │
│     → Confidence scoring + hallucination guard                  │
└─────────────────────────────────────────────────────────────────┘
```

### Retrieval Pipeline

Decomposes a natural language query into a structured graph execution plan using LLMs, then executes multi-hop traversals across Entity and Clip graphs. A beam-search engine with diversity-aware pruning, edge-aware scoring, and keyword fallback ensures high-recall retrieval.

### Generation Pipeline

Ranks retrieved clips into tiered evidence windows (high/mid/low priority), constructs a structured prompt with allowed-citation constraints, and invokes Gemini with strict JSON output schema enforcement. Every answer includes citations, reasoning, and a computed confidence score.

---

## Tech Stack

| Component              | Technology                                              |
| ---------------------- | ------------------------------------------------------- |
| **Query Decomposition** | Cerebras (`qwen-3-235b`), Groq (`qwen3-32b`, `gpt-oss-120b`) |
| **Answer Generation**  | Google Gemini (`gemini-flash-latest`, `gemini-2.5-flash`) |
| **Entity Resolution**  | SentenceTransformer (`BAAI/bge-large-en-v1.5`) + Hybrid scoring |
| **Graph Database**     | Neo4j Aura (3 instances: Clip, Entity, Mapping)         |
| **Mapping Store**      | SQLite (entity ↔ clip bipartite index)                  |
| **Video Indexing**     | Azure Video Indexer (transcription, OCR, scene detection) |
| **Vision Models**      | CLIP, DINOv2, YOLOv8 (preprocessing pipeline)          |
| **Web Portal**         | FastAPI (backend) + Next.js (frontend)                  |
| **Language**           | Python 3.10+                                            |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/venusai24/VideoGraphRAG.git
cd VideoGraphRAG
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r video_rag_preprocessing/requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# ═══════════════════════════════════════════════════
# LLM API Keys (comma-separated for round-robin)
# ═══════════════════════════════════════════════════
CEREBRAS_API_KEYS=key1,key2,key3
GROQ_API_KEYS=key1,key2
GEMINI_API_KEYS=key1,key2

# ═══════════════════════════════════════════════════
# Neo4j Aura Instances
# ═══════════════════════════════════════════════════

# Clip Graph
NEO4J_CLIP_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_CLIP_USER=neo4j
NEO4J_CLIP_PASSWORD=your_password

# Entity Graph
NEO4J_ENTITY_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_ENTITY_USER=neo4j
NEO4J_ENTITY_PASSWORD=your_password

# ═══════════════════════════════════════════════════
# Azure Video Indexer
# ═══════════════════════════════════════════════════
AZURE_VIDEO_INDEXER_API_KEY=your_key
VIDEO_INDEXER_ACCOUNT_ID=your_account_id
VIDEO_INDEXER_LOCATION=trial
VIDEO_INDEXER_ACCOUNT_TYPE=trial
```

> **Important:** At minimum, you need `CEREBRAS_API_KEYS` or `GROQ_API_KEYS` for query decomposition, and `GEMINI_API_KEYS` for answer generation.

---

## How to Run

### Ingestion Pipeline (Process Videos)

Place `.mp4` files in the `input/` directory, then:

```bash
python pipeline_orchestrator.py
```

This will:
1. Extract key frames and build clips (local pipeline)
2. Send to Azure Video Indexer for transcription/OCR/scene detection
3. Build the Neo4j knowledge graph

### Query Pipeline (Ask Questions)

#### Full Pipeline (Retrieval + Answer Generation)

```bash
python -m video_rag_query.run_query
```

#### API Server

```bash
uvicorn video_rag_query.api_server:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /query/retrieve` — Retrieval only (returns ranked clips)
- `POST /query/answer` — Full pipeline (retrieval + grounded answer)
- `GET /health` — System health check

---

## Example Query

**Input:**
```
Which airport was built on swamp land?
```

**Output:**
```json
{
  "answer": "The Navi Mumbai International Airport was constructed on reclaimed swampy land consisting of mangroves and mudflats.",
  "citations": ["79f019e3_297.0_308.0"],
  "reasoning": "Clip 79f019e3_297.0_308.0 contains transcript evidence describing the airport site as swampy mangrove wetlands that required extensive land reclamation.",
  "confidence": 0.87
}
```

---

## Key Features

- **Multi-hop Graph Reasoning** — Queries are decomposed into structured execution plans that traverse entity relationships across multiple graph hops.
- **Multimodal Evidence Fusion** — Combines video transcripts, OCR text, visual summaries, and raw clip media for comprehensive grounding.
- **Provider Failover** — Automatic failover across Cerebras → Groq (decomposition) and Gemini model tiers (generation) with per-key round-robin and cooldown management.
- **Grounded Answers** — Strict citation validation ensures every claim maps to a provided clip. Hallucinated references are rejected and retried.
- **Keyword Fallback** — When entity resolution yields low confidence, a weighted keyword search across clip text fields provides backup retrieval.
- **Beam-Search Traversal** — Diversity-aware pruning prevents result homogeneity while maintaining bounded computation.
- **LLM Observability** — Full telemetry (provider, model, latency, retry count) logged per query to `logs/llm_usage_logs.jsonl`.

---

## Limitations

- **Latency** — End-to-end query latency is 5–15s due to LLM API calls (decomposition ~3s, generation ~8s). Media uploads to Gemini add further delay.
- **Graph Quality Dependency** — Answer quality is bounded by the completeness and accuracy of the ingested knowledge graph. Poor entity extraction during preprocessing degrades retrieval.
- **Azure Dependency** — The ingestion pipeline relies on Azure Video Indexer for transcription and scene detection. Fallback produces uniform scene boundaries without semantic segmentation.
- **Rate Limits** — Heavy query load may exhaust API key quotas across providers, triggering cooldown periods.

---

## Future Improvements

- **Faster Generation** — Stream Gemini responses and parallelize retrieval + generation stages.
- **Better Entity Resolution** — Fine-tune embedding models on domain-specific entity corpora for higher recall.
- **UI Improvements** — Real-time query status, clip playback with timestamp seeking, and graph visualization in the web portal.
- **Caching Layer** — Cache frequent entity resolutions and traversal results to reduce Neo4j load.
- **Batch Ingestion** — Parallel video processing with progress tracking and resumption.

---

## Project Structure

```
VideoGraphRAG/
├── pipeline_orchestrator.py          # End-to-end ingestion orchestrator
├── Makefile                          # Build targets for local pipeline
├── QNA.json                          # Evaluation question bank
│
├── video_rag_query/                  # Query pipeline
│   ├── run_query.py                  # CLI entry point
│   ├── api_server.py                 # FastAPI server
│   ├── query_decomposer.py           # LLM-based query decomposition
│   ├── llm_client.py                 # Cerebras/Groq client with failover
│   ├── traversal.py                  # Beam-search graph traversal engine
│   ├── graph_api.py                  # Neo4j abstraction layer
│   ├── answer_generator.py           # Gemini-powered answer generation
│   ├── key_manager.py                # API key rotation + cooldown
│   ├── models.py                     # Pydantic data models
│   ├── prompts.py                    # System prompts for LLMs
│   └── utils.py                      # Keyword extraction utilities
│
├── video_rag_preprocessing/          # Ingestion pipeline
│   ├── run_pipeline.py               # Video processing entry point
│   ├── run_neo4j_pipeline.py         # Graph construction entry point
│   ├── avi_client.py                 # Azure Video Indexer client
│   ├── data_loader.py                # Data loading utilities
│   ├── semantic_graph.py             # Entity graph builder
│   ├── temporal_clip_graph.py        # Clip graph builder
│   ├── entity_normalizer.py          # Entity deduplication
│   ├── retrieval.py                  # Retrieval utilities
│   ├── pipeline/                     # Frame extraction + scoring modules
│   ├── graph_store/                  # Neo4j connection + mapping store
│   ├── config/                       # Pipeline configuration
│   └── utils/                        # Image processing utilities
│
├── portal/                           # Web interface
│   ├── backend/                      # FastAPI backend server
│   └── frontend/                     # Next.js frontend
│
├── logs/                             # Runtime logs (auto-generated)
├── outputs/                          # Processed video outputs
├── input/                            # Raw video input directory
└── deprecated/                       # Archived utility scripts
```

---

## License

This project is developed as an academic research project.