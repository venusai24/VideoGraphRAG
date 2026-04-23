import math
from typing import List, Dict, Tuple, Set
from pydantic import BaseModel

# ==============================================================================
# DATA MODELS
# ==============================================================================

class ClipNode(BaseModel):
    clip_id: str
    video_id: str
    start_time_ms: int
    end_time_ms: int
    transcript_text: str
    ocr_text: str
    keywords: List[str]
    evidence_ids: List[str]

class ClipEdge(BaseModel):
    from_clip: str
    to_clip: str
    edge_type: str
    weight: float

# ==============================================================================
# MOCK DEPENDENCIES (Stage 3)
# ==============================================================================

def embed_clip(clip: ClipNode) -> List[float]:
    """
    Mock embedding function.
    Returns a deterministic 3-dimensional normalized vector based on transcript.
    """
    if not clip.transcript_text:
        return [0.0, 0.0, 0.0]
    
    val = sum(ord(c) for c in clip.transcript_text)
    vec = [float(val % 10), float((val * 7) % 10), float((val * 13) % 10)]
    
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return [0.0, 0.0, 0.0]
    return [x / norm for x in vec]

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Helper function to calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    norm1 = math.sqrt(sum(v * v for v in vec1))
    norm2 = math.sqrt(sum(v * v for v in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# ==============================================================================
# EDGE BUILDERS
# ==============================================================================

def build_temporal_edges(clips: List[ClipNode]) -> List[ClipEdge]:
    """
    Builds temporal NEXT_CLIP and PREV_CLIP edges.
    Operates strictly within the boundary of the same video.
    """
    edges: List[ClipEdge] = []
    
    # Group by video_id to prevent temporal links across different videos
    video_groups: Dict[str, List[ClipNode]] = {}
    for clip in clips:
        video_groups.setdefault(clip.video_id, []).append(clip)
        
    for video_id, group in video_groups.items():
        # Sort sequentially by start time, fallback to clip_id for determinism
        sorted_group = sorted(group, key=lambda c: (c.start_time_ms, c.clip_id))
        
        for i in range(len(sorted_group) - 1):
            curr_clip = sorted_group[i]
            next_clip = sorted_group[i + 1]
            
            edges.append(ClipEdge(
                from_clip=curr_clip.clip_id,
                to_clip=next_clip.clip_id,
                edge_type="NEXT_CLIP",
                weight=1.0
            ))
            edges.append(ClipEdge(
                from_clip=next_clip.clip_id,
                to_clip=curr_clip.clip_id,
                edge_type="PREV_CLIP",
                weight=1.0
            ))
            
    return edges

def build_similarity_edges(clips: List[ClipNode]) -> List[ClipEdge]:
    """
    Builds SEMANTIC_SIMILARITY edges for top-K neighbors exceeding threshold.
    Avoids true O(N^2) penalty on expensive operations by caching embeddings.
    """
    if len(clips) < 2:
        return []

    # Configuration 
    K = 3
    THRESHOLD = 0.8
    
    # Cache embeddings to avoid recomputing (addressing performance constraint)
    embeddings: Dict[str, List[float]] = {}
    for clip in clips:
        try:
            embeddings[clip.clip_id] = embed_clip(clip)
        except Exception:
            embeddings[clip.clip_id] = [] # Fallback safely
            
    edges: List[ClipEdge] = []
    
    # Sort clips for deterministic traversal
    sorted_clips = sorted(clips, key=lambda c: c.clip_id)
    
    # NOTE: For massive scale (millions of clips), this exhaustive matching 
    # should be swapped with a vector DB search (e.g., FAISS). 
    # Kept separated to easily replace with an ANN query layer.
    for i, clip1 in enumerate(sorted_clips):
        emb1 = embeddings.get(clip1.clip_id)
        if not emb1:
            continue
            
        candidates: List[Tuple[float, str]] = []
        
        for clip2 in sorted_clips:
            if clip1.clip_id == clip2.clip_id:
                continue # No self-loops
                
            emb2 = embeddings.get(clip2.clip_id)
            if not emb2:
                continue
                
            sim = _cosine_similarity(emb1, emb2)
            if sim >= THRESHOLD:
                candidates.append((sim, clip2.clip_id))
                
        # Deterministic sorting: highest similarity first, then clip_id ascending
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        # Select Top-K
        for sim, target_id in candidates[:K]:
            edges.append(ClipEdge(
                from_clip=clip1.clip_id,
                to_clip=target_id,
                edge_type="SEMANTIC_SIMILARITY",
                weight=round(sim, 4)
            ))
            
    return edges

def build_keyword_edges(clips: List[ClipNode]) -> List[ClipEdge]:
    """
    Optional enhancement: Builds SAME_KEYWORD edges if clips share keywords.
    """
    edges: List[ClipEdge] = []
    sorted_clips = sorted(clips, key=lambda c: c.clip_id)
    
    for i in range(len(sorted_clips)):
        for j in range(i + 1, len(sorted_clips)):
            clip1 = sorted_clips[i]
            clip2 = sorted_clips[j]
            
            shared = set(clip1.keywords).intersection(set(clip2.keywords))
            if shared:
                edges.append(ClipEdge(
                    from_clip=clip1.clip_id,
                    to_clip=clip2.clip_id,
                    edge_type="SAME_KEYWORD",
                    weight=0.7
                ))
                edges.append(ClipEdge(
                    from_clip=clip2.clip_id,
                    to_clip=clip1.clip_id,
                    edge_type="SAME_KEYWORD",
                    weight=0.7
                ))
    return edges

# ==============================================================================
# MAIN GRAPH ORCHESTRATOR
# ==============================================================================

def build_clip_graph(clips: List[ClipNode]) -> List[ClipEdge]:
    """
    Main orchestration function to build the full clip graph.
    """
    if not clips:
        return []
        
    all_edges: List[ClipEdge] = []
    
    # 1. Build all edge types
    all_edges.extend(build_temporal_edges(clips))
    all_edges.extend(build_similarity_edges(clips))
    all_edges.extend(build_keyword_edges(clips))
    
    # 2. Deduplicate edges using composite key
    unique_edges: Dict[Tuple[str, str, str], ClipEdge] = {}
    
    for edge in all_edges:
        key = (edge.from_clip, edge.to_clip, edge.edge_type)
        if key not in unique_edges:
            unique_edges[key] = edge
        else:
            # If duplicate occurs, safely keep the one with the higher weight
            if edge.weight > unique_edges[key].weight:
                unique_edges[key] = edge
                
    # 3. Deterministic return order
    final_edges = list(unique_edges.values())
    final_edges.sort(key=lambda e: (e.from_clip, e.to_clip, e.edge_type))
    
    return final_edges

# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    # Create mock ClipNodes
    mock_clips = [
        ClipNode(
            clip_id="clip_001",
            video_id="vid_alpha",
            start_time_ms=0,
            end_time_ms=5000,
            transcript_text="The quick brown fox",
            ocr_text="FOX",
            keywords=["animal", "nature"],
            evidence_ids=["ev_1"]
        ),
        ClipNode(
            clip_id="clip_002",
            video_id="vid_alpha",
            start_time_ms=5000,
            end_time_ms=10000,
            transcript_text="jumps over the lazy dog",
            ocr_text="DOG",
            keywords=["animal", "action"],
            evidence_ids=["ev_2"]
        ),
        ClipNode(
            clip_id="clip_003",
            video_id="vid_alpha",
            start_time_ms=10000,
            end_time_ms=15000,
            transcript_text="A completely different topic about cars",
            ocr_text="CAR",
            keywords=["vehicle"],
            evidence_ids=["ev_3"]
        ),
        # Identical text to clip_001 to guarantee a high SEMANTIC_SIMILARITY match
        ClipNode(
            clip_id="clip_004",
            video_id="vid_beta",
            start_time_ms=0,
            end_time_ms=5000,
            transcript_text="The quick brown fox",
            ocr_text="FOX",
            keywords=["nature"],
            evidence_ids=["ev_4"]
        )
    ]
    
    # Build graph
    edges = build_clip_graph(mock_clips)
    
    # Print edges
    print(f"{'FROM':<10} | {'TO':<10} | {'EDGE TYPE':<20} | {'WEIGHT'}")
    print("-" * 55)
    for e in edges:
        print(f"{e.from_clip:<10} | {e.to_clip:<10} | {e.edge_type:<20} | {e.weight}")