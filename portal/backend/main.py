import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="VideoGraphRAG Demo Portal API")

# Setup CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ENRICHMENT_PATH = OUTPUTS_DIR / "enrichment_vision.jsonl"
TRANSCRIPT_PATH = OUTPUTS_DIR / "full_transcript.json"

# Mount outputs for static file serving (videos, keyframes)
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

class QueryRequest(BaseModel):
    query: str
    video_id: Optional[str] = "default"

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str

# In-memory job state (for demo purposes)
JOBS = {
    "demo_job": {
        "status": "COMPLETED",
        "progress": 100.0,
        "message": "Project Hail Mary trailer processed successfully."
    }
}

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": os.path.getmtime(ENRICHMENT_PATH) if ENRICHMENT_PATH.exists() else None}

@app.get("/api/jobs")
async def get_jobs():
    return JOBS

@app.get("/api/clips")
async def get_clips():
    """Returns all enriched clips from the .jsonl file."""
    if not ENRICHMENT_PATH.exists():
        return []
    
    clips = []
    with open(ENRICHMENT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                clips.append(json.loads(line))
    return clips

@app.get("/api/graph")
async def get_graph():
    """Transforms enrichment data into a Cytoscape-friendly GraphJSON."""
    if not ENRICHMENT_PATH.exists():
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []
    
    # Track unique entities to build Layer 2
    entities_map = {} # name -> node_data
    
    with open(ENRICHMENT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            clip = data["clip"]
            vision = data["vision"]
            
            # Layer 1 Node (Clip)
            clip_node_id = f"clip_{clip['clip_id']}"
            nodes.append({
                "data": {
                    "id": clip_node_id,
                    "label": f"Clip {clip['clip_id']}",
                    "type": "clip",
                    "start": clip["start_time_sec"],
                    "end": clip["end_time_sec"],
                    "summary": vision.get("summary", "No summary available") if vision else "Processing..."
                },
                "classes": "layer1"
            })
            
            # Layer 2 Nodes (Entities) & Cross-Layer Edges
            entities = vision.get("entities", []) if vision else []
            for ent in entities:
                ent_name = ent["name"]
                if ent_name not in entities_map:
                    entities_map[ent_name] = {
                        "id": f"entity_{ent_name}",
                        "label": ent_name,
                        "type": "entity",
                        "category": ent.get("category", "unknown")
                    }
                    nodes.append({
                        "data": entities_map[ent_name],
                        "classes": "layer2"
                    })
                
                # Edge: Entity SEEN_IN Clip
                edges.append({
                    "data": {
                        "id": f"edge_{ent_name}_{clip['clip_id']}",
                        "source": f"entity_{ent_name}",
                        "target": clip_node_id,
                        "label": "SEEN_IN",
                        "weight": ent.get("confidence", 1.0)
                    }
                })

            # Layer 2 Edges (Actions)
            actions = vision.get("actions", []) if vision else []
            for action in actions:
                # simplified: just connect subject to object if they exist
                subj = action.get("subject")
                obj = action.get("object")
                if subj and obj:
                    edges.append({
                        "data": {
                            "id": f"action_{subj}_{obj}_{clip['clip_id']}",
                            "source": f"entity_{subj}",
                            "target": f"entity_{obj}",
                            "label": action["description"],
                            "type": "action_edge"
                        },
                        "classes": "semantic_edge"
                    })

    return {"nodes": nodes, "edges": edges}

@app.post("/api/query")
async def run_query(request: QueryRequest):
    """Simple grounded reasoning mock."""
    query = request.query.lower()
    
    # 1. Parse query for entities (Mock)
    # In a real system, we'd use an LLM here.
    # For the demo, we search keywords in our graph data.
    
    clips = await get_clips()
    relevant_clips = []
    
    for c in clips:
        vision = c.get("vision")
        summary = (vision.get("summary") or "").lower() if vision else ""
        if any(word in summary for word in query.split()):
            relevant_clips.append(c)
            
    if not relevant_clips:
        # Fallback to first few if no match for demo
        relevant_clips = clips[:2]

    # Construct reasoning trace
    steps = [
        {"step": "QUERY_PARSE", "content": f"Identified intent to find information about keywords in: '{request.query}'"},
        {"step": "GRAPH_TRAVERSAL", "content": f"Traversed Semantic Layer for related entities. Found matches in {len(relevant_clips)} clips."},
        {"step": "GROUNDING_FETCH", "content": f"Retrieved {len(relevant_clips)} clip segments as factual evidence."}
    ]
    
    answer = f"Based on the video analysis, I found {len(relevant_clips)} relevant segments. "
    if relevant_clips:
        first_vision = relevant_clips[0].get("vision")
        if first_vision:
            answer += f"Specifically, the first clip shows: {first_vision.get('summary', 'an analyzed scene')}"

    return {
        "answer": answer,
        "reasoning_steps": steps,
        "evidence_clips": [c["clip"]["clip_id"] for c in relevant_clips]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
