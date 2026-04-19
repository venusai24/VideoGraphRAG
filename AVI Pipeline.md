### Pipeline Phase 1: Perceptual Indexing (Azure Video Indexer)

To preserve Azure credits, AVI is used strictly for its standard indexing capabilities, which generate high-fidelity raw metadata without utilizing Azure's native generative AI wrappers.

1. **Ingestion**: Call the `POST /Videos` API to upload and index your media. Use the `advanced` video/audio preset to ensure all required fields (Face tracking, Object detection, OCR) are populated.
    
2. **Metadata Retrieval**: Execute a `GET /Index` call to retrieve the raw insights JSON. This file contains the atomic building blocks: `insights/scenes`, `insights/shots`, `insights/faces`, `insights/labels`, and `insights/ocr`.
    
3. **Keyframe Export**: (Optional) Sample one keyframe per shot ID. These visual anchors can be passed to Gemini to resolve visual ambiguities that text metadata alone cannot solve.
    

---

### Pipeline Phase 2: Information Extraction & Semantic Lifting (Gemini API)

Gemini processes the raw AVI JSON to populate the layers. Because Gemini 1.5 Pro supports a 2-million-token context, it can ingest the entire AVI output (often several MBs) in a single reasoning pass.

#### Layer 1: Clip Graph Population (Grounding Layer)

- **Nodes**: Gemini parses the `insights/scenes` and `insights/shots` arrays from the AVI JSON to create Layer 1 nodes.
    
    - **Mapping**: `clip_id` (AVI scene ID), `timestamps` (start/end offsets), and `ocr` text.
        
    - **Generation**: Gemini summarizes the specific labels and transcript within each segment to create a narrative `generated_summary`.
        
- **Edges**:
    
    - **Chronographic**: Gemini establishes directed edges between $Clip_{n}$ and $Clip_{n+1}$ based on timestamp adjacency.
        
    - **Semantic**: Use Gemini’s `embedContent` API to generate vectors for each summary, creating edges where similarity scores exceed a defined threshold.
        

#### Layer 2: Semantic Graph Population (Reasoning Layer)

Gemini performs "Entity Resolution" to transform localized detections into abstract concept nodes.

- **Nodes**:
    
    - **Entities**: Collates `insights/faces` and `insights/observedPeople` across the whole video. For example, "Face ID 1" and "Observed Person 5" are resolved into a single conceptual node: "Warehouse Worker".
        
    - **Categories**: Uses AVI's `insights/topics` to create high-level thematic nodes.
        
- **Edges**:
    
    - **Actions**: Gemini reasons over the sequential labels and transcript to extract `(Entity) --[action]--> (Entity)` triples (e.g., `Worker --[drops]--> Box`).
        
    - **Location**: Maps entities to scenes via `(Entity) --[located_in]--> (Scene)` edges.
        

---

### Pipeline Phase 3: Uniform Output Schema (JSON-LD)

To ensure Layer 1 can be "lifted" into Layer 2, every triple in the output must include a `grounding` field that refers to the `clip_id` from Layer 1.

**Recommended Gemini Output Schema (JSON Mode):**

JSON

```
{
  "@context": "https://schema.org/",
  "clip_graph_layer_1":,
  "semantic_graph_layer_2": {
    "entities":,
    "triples": [
      {
        "subject": "ent_worker_01",
        "predicate": "buys",
        "object": "ent_item_42",
        "grounding": "scene_01" 
      }
    ]
  }
}
```

---

### Phase 4: Knowledge Graph Construction & VideoRAG Integration

The resulting JSON-LD is ingested into a graph database (e.g., Neo4j or a triple store).

1. **Top-Down Query Execution**: When a user asks "Why did the worker drop the box?", the system first queries Layer 2 to find the causal path: `(Worker) --[causes]--> (Accident)`.
    
2. **Deterministic Retrieval**: The system follows the `grounding` edge from the "Accident" node down to the specific `clip_id` in Layer 1.
    
3. **Result**: The VideoRAG system retrieves the exact video segment (timestamp range) from the original source file to provide the visual answer.