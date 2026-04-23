"""
retrieval/query_engine.py

GraphRAG Query Engine Module
Retrieves relevant ClipNodes using semantic graphs, cross-layer links, and clip graphs.
"""

import re
from typing import Set, Tuple, Optional, Any
from pydantic import BaseModel

class Graph(BaseModel):
    nodes: list[dict]
    edges: list[dict]

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", 
    "for", "with", "is", "are", "am", "was", "were", "of", "by"
}

def build_indices(graph: Graph) -> dict[str, Any]:
    """
    Builds adjacency maps to avoid scanning the full graph repeatedly.
    Maps node_id -> node, and builds incoming/outgoing edge lists.
    """
    indices: dict[str, Any] = {
        "nodes_by_id": {},
        "out_edges": {},
        "in_edges": {}
    }
    
    for node in graph.nodes:
        indices["nodes_by_id"][node["id"]] = node
        indices["out_edges"][node["id"]] = []
        indices["in_edges"][node["id"]] = []

    for edge in graph.edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src in indices["out_edges"]:
            indices["out_edges"][src].append(edge)
        if tgt in indices["in_edges"]:
            indices["in_edges"][tgt].append(edge)

    return indices


def extract_query_terms(query: str) -> list[str]:
    """
    Query Understanding: lowercases, tokenizes, and removes stopwords.
    """
    words = re.findall(r'\b\w+\b', query.lower())
    return [w for w in words if w not in STOPWORDS]


def find_matching_semantic_nodes(
    query_terms: list[str], 
    graph: Graph, 
    indices: Optional[dict[str, Any]] = None
) -> list[str]:
    """
    Semantic Node Matching: matches query terms to node properties 'name' and 'aliases'.
    """
    matched_ids = []
    nodes = indices["nodes_by_id"].values() if indices else graph.nodes
    
    for node in nodes:
        if node.get("type") != "semantic":
            continue
            
        props = node.get("properties", {})
        name = str(props.get("name", "")).lower()
        aliases = [str(a).lower() for a in props.get("aliases", [])]

        for term in query_terms:
            if term in name or any(term in alias for alias in aliases):
                matched_ids.append(node["id"])
                break  # Count node match once
                
    return matched_ids


def get_clips_from_semantic_nodes(
    node_ids: list[str], 
    graph: Graph, 
    indices: Optional[dict[str, Any]] = None
) -> set[str]:
    """
    Cross-Layer Retrieval: traverses MENTIONED_IN edges from semantic nodes to clips.
    """
    clips: set[str] = set()
    
    if indices:
        for nid in node_ids:
            for edge in indices["out_edges"].get(nid, []):
                if edge.get("type") == "MENTIONED_IN":
                    tgt = edge["target"]
                    if indices["nodes_by_id"].get(tgt, {}).get("type") == "clip":
                        clips.add(tgt)
    else:
        node_ids_set = set(node_ids)
        for edge in graph.edges:
            if edge.get("type") == "MENTIONED_IN" and edge.get("source") in node_ids_set:
                clips.add(edge["target"])
                
    return clips


def expand_clip_candidates(
    clip_ids: set[str], 
    graph: Graph, 
    indices: Optional[dict[str, Any]] = None
) -> set[str]:
    """
    Graph Expansion: adds NEXT_CLIP, PREV_CLIP, and SEMANTIC_SIMILARITY neighbors.
    """
    expanded: set[str] = set()
    allowed_edges = {"NEXT_CLIP", "PREV_CLIP", "SEMANTIC_SIMILARITY"}

    if indices:
        for cid in clip_ids:
            # Check outgoing
            for edge in indices["out_edges"].get(cid, []):
                if edge.get("type") in allowed_edges:
                    tgt = edge["target"]
                    if indices["nodes_by_id"].get(tgt, {}).get("type") == "clip":
                        expanded.add(tgt)
            # Check incoming (reciprocal/undirected similarities)
            for edge in indices["in_edges"].get(cid, []):
                if edge.get("type") in allowed_edges:
                    src = edge["source"]
                    if indices["nodes_by_id"].get(src, {}).get("type") == "clip":
                        expanded.add(src)
    else:
        for edge in graph.edges:
            if edge.get("type") in allowed_edges:
                if edge["source"] in clip_ids:
                    expanded.add(edge["target"])
                if edge["target"] in clip_ids:
                    expanded.add(edge["source"])

    # Ensure we only return purely expanded nodes (not the original seeds)
    return expanded - clip_ids


def score_clip(
    clip_id: str, 
    query_terms: list[str], 
    graph: Graph,
    indices: Optional[dict[str, Any]] = None,
    matched_semantic_nodes: Optional[list[str]] = None,
    direct_clips: Optional[set[str]] = None
) -> Tuple[float, dict[str, Any]]:
    """
    Scoring Function: calculates hybrid score for a candidate clip.
    """
    if indices is None:
        indices = build_indices(graph)
    if matched_semantic_nodes is None:
        matched_semantic_nodes = find_matching_semantic_nodes(query_terms, graph, indices)
    if direct_clips is None:
        direct_clips = get_clips_from_semantic_nodes(matched_semantic_nodes, graph, indices)

    clip_node = indices["nodes_by_id"].get(clip_id, {})
    in_edges = indices["in_edges"].get(clip_id, [])
    props = clip_node.get("properties", {})

    # Determine which matched semantic nodes point to this clip
    linked_semantics = []
    matched_set = set(matched_semantic_nodes)
    for edge in in_edges:
        if edge.get("type") == "MENTIONED_IN" and edge.get("source") in matched_set:
            linked_semantics.append(edge["source"])

    # A. Semantic match (0.4 weight)
    semantic_score = 0.0
    if matched_semantic_nodes:
        semantic_score = min(1.0, len(linked_semantics) / max(len(matched_semantic_nodes), 1))

    # B. Keyword match (0.2 weight)
    keywords = props.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = [str(k).lower() for k in keywords]
    
    matched_keywords = [t for t in query_terms if any(t in k for k in keywords)]
    keyword_score = min(1.0, len(matched_keywords) / max(len(query_terms), 1))

    # C. Graph proximity (0.2 weight)
    is_direct = (clip_id in direct_clips)
    proximity_score = 1.0 if is_direct else 0.5
    if not matched_semantic_nodes:
        # If no semantic nodes matched, everything relies on keyword fallback
        proximity_score = 1.0

    # D. Edge confidence (0.2 weight)
    avg_weight = 0.0
    if in_edges:
        avg_weight = sum(float(e.get("weight", 0.0)) for e in in_edges) / len(in_edges)
    edge_score = min(1.0, avg_weight)

    # Final weighted score
    final_score = (0.4 * semantic_score) + \
                  (0.2 * keyword_score) + \
                  (0.2 * proximity_score) + \
                  (0.2 * edge_score)

    reason = {
        "matched_terms": matched_keywords,
        "semantic_hits": linked_semantics,
        "expansion_used": not is_direct
    }
    
    return final_score, reason


def query(query_text: str, graph: Graph, top_k: int = 5) -> list[dict]:
    """
    Main entry point: accepts natural language query, retrieves, and ranks clip IDs.
    """
    if not graph.nodes:
        return []

    # 9. Performance: Build indices once per query
    indices = build_indices(graph)
    
    # 1. Query Understanding
    query_terms = extract_query_terms(query_text)
    if not query_terms:
        return []

    # 2. Semantic Node Matching
    matched_semantics = find_matching_semantic_nodes(query_terms, graph, indices)
    
    # 3. Cross-Layer Retrieval
    direct_clips = get_clips_from_semantic_nodes(matched_semantics, graph, indices)
    expanded_clips = set()

    # 10. Edge Cases: No semantic match fallback
    if not matched_semantics:
        for node_id, node in indices["nodes_by_id"].items():
            if node.get("type") == "clip":
                props = node.get("properties", {})
                kw = props.get("keywords", [])
                if isinstance(kw, str): 
                    kw = [kw]
                kw = [str(k).lower() for k in kw]
                # Check for overlap
                if any(any(t in k for k in kw) for t in query_terms):
                    direct_clips.add(node_id)
    else:
        # 4. Graph Expansion
        expanded_clips = expand_clip_candidates(direct_clips, graph, indices)

    candidate_clips = direct_clips.union(expanded_clips)

    # 5 & 6. Scoring and Ranking
    results = []
    for cid in candidate_clips:
        score, reason = score_clip(cid, query_terms, graph, indices, matched_semantics, direct_clips)
        results.append({
            "clip_id": cid,
            "score": score,
            "reason": reason
        })

    # 8. Determinism: Sort descending by score, ascending by clip_id for ties
    results.sort(key=lambda x: (-x["score"], x["clip_id"]))

    return results[:top_k]


def query_graph(query_text: str, graph: Graph, top_k: int = 5) -> list[dict]:
    """
    Backward-compatible alias used by pipeline.py.
    """
    return query(query_text=query_text, graph=graph, top_k=top_k)


# =====================================================================
# TESTING BLOCK
# =====================================================================
if __name__ == "__main__":
    # Create mock graph representing semantic concepts and clips
    mock_graph = Graph(
        nodes=[
            {"id": "sem_1", "type": "semantic", "properties": {"name": "Coffee", "aliases": ["espresso"]}},
            {"id": "sem_2", "type": "semantic", "properties": {"name": "Cup", "aliases": ["mug"]}},
            {"id": "sem_3", "type": "semantic", "properties": {"name": "Person", "aliases": ["human"]}},
            {"id": "sem_4", "type": "semantic", "properties": {"name": "Kitchen", "aliases": ["cooking area"]}},
            {"id": "clip_1", "type": "clip", "properties": {"keywords": ["coffee", "cup", "morning"]}},
            {"id": "clip_2", "type": "clip", "properties": {"keywords": ["empty", "room"]}},
            {"id": "clip_3", "type": "clip", "properties": {"keywords": ["person", "kitchen", "cooking"]}},
            {"id": "clip_4", "type": "clip", "properties": {"keywords": ["chef", "food"]}}
        ],
        edges=[
            {"source": "sem_1", "target": "clip_1", "type": "MENTIONED_IN", "weight": 0.9, "properties": {}},
            {"source": "sem_2", "target": "clip_1", "type": "MENTIONED_IN", "weight": 0.8, "properties": {}},
            {"source": "sem_3", "target": "clip_3", "type": "MENTIONED_IN", "weight": 1.0, "properties": {}},
            {"source": "sem_4", "target": "clip_3", "type": "MENTIONED_IN", "weight": 0.9, "properties": {}},
            {"source": "sem_4", "target": "clip_4", "type": "MENTIONED_IN", "weight": 0.6, "properties": {}},
            {"source": "clip_1", "target": "clip_2", "type": "NEXT_CLIP", "weight": 1.0, "properties": {}},
            {"source": "clip_3", "target": "clip_4", "type": "SEMANTIC_SIMILARITY", "weight": 0.75, "properties": {}}
        ]
    )

    print("--- Test 1: 'coffee cup' ---")
    res1 = query("coffee cup", mock_graph)
    for r in res1:
        print(r)

    print("\n--- Test 2: 'person in kitchen' ---")
    res2 = query("person in kitchen", mock_graph)
    for r in res2:
        print(r)
        
    print("\n--- Test 3: Fallback 'morning room' ---")
    res3 = query("morning room", mock_graph)
    for r in res3:
        print(r)