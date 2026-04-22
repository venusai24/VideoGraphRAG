from typing import List, Dict, Set
from pydantic import BaseModel


class ClipNode(BaseModel):
    clip_id: str
    video_id: str
    start_time_ms: int
    end_time_ms: int
    transcript_text: str
    ocr_text: str
    keywords: List[str]
    evidence_ids: List[str]


class SemanticNode(BaseModel):
    concept_id: str
    canonical_name: str
    aliases: List[str]
    embedding: List[float]
    support_count: int


class MentionEdge(BaseModel):
    semantic_id: str
    clip_id: str
    match_text: str
    confidence: float
    source: str   # "transcript" | "ocr" | "keyword" | "derived"


def create_cross_layer_links(
    clips: List[ClipNode],
    semantics: List[SemanticNode],
    mapping: Dict[str, List[str]]
) -> List[MentionEdge]:
    """
    Creates Cross-Layer MentionEdges linking SemanticNodes to ClipNodes
    based on the Stage 5 canonicalization mapping.
    """
    # 7. Performance: Use dictionaries for fast lookup
    clip_dict: Dict[str, ClipNode] = {c.clip_id: c for c in clips}
    semantic_dict: Dict[str, SemanticNode] = {s.concept_id: s for s in semantics}
    
    edges: List[MentionEdge] = []
    seen_edges: Set[str] = set()

    # 9. Determinism: Sort mapping keys to ensure consistent processing order
    for clip_id in sorted(mapping.keys()):
        # 8. Edge Cases: Handle missing clip_id
        if clip_id not in clip_dict:
            continue
            
        clip = clip_dict[clip_id]
        concept_ids = mapping[clip_id]
        
        # 9. Determinism & 5. Deduplication: Unique concepts per clip, sorted
        unique_concept_ids = sorted(list(set(concept_ids)))
        
        for concept_id in unique_concept_ids:
            # 8. Edge Cases: Handle missing concept_id
            if concept_id not in semantic_dict:
                continue
                
            semantic = semantic_dict[concept_id]
            
            # Deduplication check (safety measure)
            edge_key = f"{concept_id}::{clip_id}"
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            # 2. Match Text Selection & 3. Source Attribution & 4. Confidence Scoring
            best_priority = 4  # 1: transcript, 2: ocr, 3: keyword, 4: derived
            base_conf = 0.7
            best_match_text = semantic.canonical_name
            total_occurrences = 0
            
            # Search pool includes canonical name and all aliases
            aliases_to_check = [semantic.canonical_name] + semantic.aliases
            
            for alias in aliases_to_check:
                if not alias:
                    continue
                    
                alias_lower = alias.lower()
                
                # Count occurrences in each source safely
                t_count = clip.transcript_text.lower().count(alias_lower) if clip.transcript_text else 0
                o_count = clip.ocr_text.lower().count(alias_lower) if clip.ocr_text else 0
                k_count = sum(1 for kw in clip.keywords if alias_lower in kw.lower()) if clip.keywords else 0
                
                total_occurrences += (t_count + o_count + k_count)
                
                # Update best match based on priority
                if t_count > 0 and best_priority > 1:
                    best_priority = 1
                    base_conf = 0.9
                    best_match_text = alias
                elif o_count > 0 and best_priority > 2:
                    best_priority = 2
                    base_conf = 0.85
                    best_match_text = alias
                elif k_count > 0 and best_priority > 3:
                    best_priority = 3
                    base_conf = 0.80
                    best_match_text = alias

            # Map priority back to source string
            source_map = {1: "transcript", 2: "ocr", 3: "keyword", 4: "derived"}
            final_source = source_map[best_priority]
            
            # 6. Optional Enhancement: Boost confidence slightly if multiple appearances
            extra_occurrences = max(0, total_occurrences - 1)
            final_conf = min(1.0, base_conf + (extra_occurrences * 0.01))
            final_conf = round(final_conf, 3)
            
            # Create the edge
            edge = MentionEdge(
                semantic_id=concept_id,
                clip_id=clip_id,
                match_text=best_match_text,
                confidence=final_conf,
                source=final_source
            )
            edges.append(edge)

    # 9. Determinism: Final sort of edges
    edges.sort(key=lambda e: (e.clip_id, e.semantic_id))
    return edges


def link_entities_to_clips(
    clips: List[ClipNode],
    semantics: List[SemanticNode],
    mapping: Dict[str, List[str]]
) -> List[MentionEdge]:
    """
    Backward-compatible alias used by pipeline.py.
    """
    return create_cross_layer_links(clips, semantics, mapping)


if __name__ == "__main__":
    # TESTING REQUIREMENT
    
    # Create mock ClipNodes
    mock_clips = [
        ClipNode(
            clip_id="clip_001",
            video_id="vid_A",
            start_time_ms=0,
            end_time_ms=5000,
            transcript_text="We implemented a new neural network architecture.",
            ocr_text="NN Arch v2",
            keywords=["machine learning", "deep learning"],
            evidence_ids=["ev_1"]
        ),
        ClipNode(
            clip_id="clip_002",
            video_id="vid_A",
            start_time_ms=5000,
            end_time_ms=10000,
            transcript_text="The system is very fast.",
            ocr_text="Latency: 12ms",
            keywords=["performance", "speed", "neural network"],
            evidence_ids=["ev_2"]
        )
    ]
    
    # Create mock SemanticNodes
    mock_semantics = [
        SemanticNode(
            concept_id="sem_nn",
            canonical_name="Neural Network",
            aliases=["neural net", "neural network"],
            embedding=[0.1, 0.2, 0.3],
            support_count=5
        ),
        SemanticNode(
            concept_id="sem_perf",
            canonical_name="System Performance",
            aliases=["latency", "speed"],
            embedding=[0.4, 0.5, 0.6],
            support_count=3
        ),
        SemanticNode(
            concept_id="sem_missing",
            canonical_name="Missing Concept",
            aliases=["ghost"],
            embedding=[0.0, 0.0, 0.0],
            support_count=1
        )
    ]
    
    # Create mock mapping (Stage 5 output)
    mock_mapping = {
        "clip_001": ["sem_nn", "sem_missing"],  # sem_nn in transcript, sem_missing derived
        "clip_002": ["sem_nn", "sem_perf", "sem_nn"],  # sem_nn in keyword, sem_perf in OCR, duplicate sem_nn
        "clip_999": ["sem_nn"]  # Edge case: clip_id missing in nodes
    }
    
    # Run linking
    generated_edges = create_cross_layer_links(mock_clips, mock_semantics, mock_mapping)
    
    # Print MentionEdges
    print(f"Generated {len(generated_edges)} MentionEdges:\n")
    for e in generated_edges:
        print(e.model_dump_json(indent=2))