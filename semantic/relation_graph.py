import math
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from itertools import combinations
from pydantic import BaseModel

# ==========================================
# Data Models
# ==========================================

class SemanticNode(BaseModel):
    concept_id: str
    canonical_name: str
    aliases: List[str]
    embedding: List[float]
    support_count: int

class ClipNode(BaseModel):
    clip_id: str
    video_id: str
    start_time_ms: int
    end_time_ms: int
    transcript_text: str
    ocr_text: str
    keywords: List[str]
    evidence_ids: List[str]

class SemanticEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float

# ==========================================
# Helpers
# ==========================================

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot_product / (mag1 * mag2)

# ==========================================
# Relation Graph Builders
# ==========================================

def build_cooccurrence_edges(mapping: Dict[str, List[str]]) -> List[SemanticEdge]:
    """
    Build CO_OCCURS edges by pairing concepts within the same clip.
    Weights are normalized based on the maximum co-occurrence frequency.
    """
    co_occurrences: Counter[Tuple[str, str]] = Counter()

    # Sort keys for determinism
    for clip_id in sorted(mapping.keys()):
        concept_ids = mapping[clip_id]
        if not concept_ids or len(concept_ids) < 2:
            continue
        
        # Deduplicate within the clip and sort to enforce deterministic pairing (A, B)
        unique_concepts = sorted(list(set(concept_ids)))
        
        for c1, c2 in combinations(unique_concepts, 2):
            co_occurrences[(c1, c2)] += 1

    if not co_occurrences:
        return []

    max_co_occurrence = max(co_occurrences.values())
    edges = []

    # Sort items for deterministic graph generation
    for (source, target), count in sorted(co_occurrences.items()):
        edges.append(SemanticEdge(
            source=source,
            target=target,
            relation="CO_OCCURS",
            weight=count / max_co_occurrence
        ))
        
    return edges

def build_similarity_edges(
    semantic_nodes: List[SemanticNode], 
    threshold: float = 0.8, 
    top_k: int = 5
) -> List[SemanticEdge]:
    """
    Build RELATED_TO edges based on cosine similarity of semantic node embeddings.
    Enforces density control by limiting neighbors to top-K per node.
    """
    # Sort nodes for deterministic processing
    nodes = sorted(semantic_nodes, key=lambda x: x.concept_id)
    n = len(nodes)
    
    # adjacency list for tracking similarity scores per node
    adj: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

    # Compute pairwise similarities (O(N^2) required here, optimized via sorted pairing)
    for i in range(n):
        for j in range(i + 1, n):
            n1, n2 = nodes[i], nodes[j]
            sim = _cosine_similarity(n1.embedding, n2.embedding)
            if sim >= threshold:
                adj[n1.concept_id].append((sim, n2.concept_id))
                adj[n2.concept_id].append((sim, n1.concept_id))

    edges_set: Set[Tuple[str, str, float]] = set()

    for node_id in sorted(adj.keys()):
        neighbors = adj[node_id]
        # Sort by similarity descending, then neighbor ID ascending for deterministic top-K
        neighbors.sort(key=lambda x: (-x[0], x[1]))
        top_neighbors = neighbors[:top_k]

        for sim, neighbor_id in top_neighbors:
            # Sort source and target to avoid directional duplicates
            source, target = sorted([node_id, neighbor_id])
            edges_set.add((source, target, sim))

    edges = []
    # Sort the resulting set for final deterministic ordering
    for source, target, sim in sorted(edges_set):
        edges.append(SemanticEdge(
            source=source,
            target=target,
            relation="RELATED_TO",
            weight=sim
        ))
        
    return edges

def build_action_on_edges(
    clip_nodes: List[ClipNode], 
    mapping: Dict[str, List[str]], 
    semantic_nodes: List[SemanticNode]
) -> List[SemanticEdge]:
    """
    Build ACTION_ON edges by scanning clip transcripts for interaction patterns
    between concept pairs present in the clip.
    """
    node_lookup = {node.concept_id: node for node in semantic_nodes}
    edges_dict: Dict[Tuple[str, str], float] = {}
    
    # Lightweight extraction vocabulary
    action_verbs = ["picks up", "places", "moves", "hits", "pushes", "holds"]

    # Sort for determinism
    for clip in sorted(clip_nodes, key=lambda x: x.clip_id):
        concept_ids = sorted(list(set(mapping.get(clip.clip_id, []))))
        if len(concept_ids) < 2:
            continue
            
        transcript = clip.transcript_text.lower()
        if not transcript:
            continue

        for c1, c2 in combinations(concept_ids, 2):
            n1 = node_lookup.get(c1)
            n2 = node_lookup.get(c2)
            if not n1 or not n2:
                continue
                
            name1 = n1.canonical_name.lower()
            name2 = n2.canonical_name.lower()

            for verb in action_verbs:
                pattern_forward = f"{name1} {verb} {name2}"
                pattern_reverse = f"{name2} {verb} {name1}"

                if pattern_forward in transcript:
                    edges_dict[(c1, c2)] = 0.7
                if pattern_reverse in transcript:
                    edges_dict[(c2, c1)] = 0.7

    edges = []
    for (source, target), weight in sorted(edges_dict.items()):
        edges.append(SemanticEdge(
            source=source,
            target=target,
            relation="ACTION_ON",
            weight=weight
        ))
        
    return edges

def build_semantic_relation_graph(
    semantic_nodes: List[SemanticNode],
    clip_nodes: List[ClipNode],
    mapping: Dict[str, List[str]]
) -> List[SemanticEdge]:
    """
    Orchestrate the creation of Layer 2 semantic relations graph.
    Returns a unified deterministic list of SemanticEdges.
    """
    edges: List[SemanticEdge] = []
    
    # 1. CO_OCCURS Edges
    edges.extend(build_cooccurrence_edges(mapping))
    
    # 2. RELATED_TO Edges
    edges.extend(build_similarity_edges(semantic_nodes))
    
    # 3. ACTION_ON Edges
    edges.extend(build_action_on_edges(clip_nodes, mapping, semantic_nodes))
    
    return edges

# ==========================================
# Testing Block
# ==========================================

if __name__ == "__main__":
    # 1. Mock Semantic Nodes
    n_person = SemanticNode(
        concept_id="c_001", canonical_name="Person", aliases=["man"], 
        embedding=[0.9, 0.1, 0.0], support_count=10
    )
    n_cup = SemanticNode(
        concept_id="c_002", canonical_name="Cup", aliases=["mug"], 
        embedding=[0.1, 0.9, 0.0], support_count=5
    )
    n_bottle = SemanticNode(
        concept_id="c_003", canonical_name="Bottle", aliases=["flask"], 
        embedding=[0.1, 0.85, 0.2], support_count=8
    )
    n_table = SemanticNode(
        concept_id="c_004", canonical_name="Table", aliases=["desk"], 
        embedding=[0.0, 0.0, 0.9], support_count=3
    )
    
    nodes = [n_person, n_cup, n_bottle, n_table]

    # 2. Mock Clip Nodes
    c1 = ClipNode(
        clip_id="clip_1", video_id="v1", start_time_ms=0, end_time_ms=1000,
        transcript_text="The person picks up cup from the desk.", ocr_text="",
        keywords=[], evidence_ids=[]
    )
    c2 = ClipNode(
        clip_id="clip_2", video_id="v1", start_time_ms=1000, end_time_ms=2000,
        transcript_text="Person holding a bottle.", ocr_text="",
        keywords=[], evidence_ids=[]
    )
    c3 = ClipNode(
        clip_id="clip_3", video_id="v2", start_time_ms=0, end_time_ms=5000,
        transcript_text="Table with a cup on it.", ocr_text="",
        keywords=[], evidence_ids=[]
    )
    
    clips = [c1, c2, c3]

    # 3. Mock Clip -> Concept Mapping
    mapping = {
        "clip_1": ["c_001", "c_002", "c_004"],
        "clip_2": ["c_001", "c_003"],
        "clip_3": ["c_002", "c_004"]
    }

    # 4. Build Graph
    final_edges = build_semantic_relation_graph(nodes, clips, mapping)

    # 5. Output Result
    for edge in final_edges:
        print(f"Source: {edge.source:5} | Target: {edge.target:5} | Relation: {edge.relation:10} | Weight: {edge.weight:.3f}")