import json
from typing import Any
from pydantic import BaseModel, Field

# ==========================================
# Input Models (Contracts)
# ==========================================

class ClipNode(BaseModel):
    clip_id: str
    video_id: str
    start_time_ms: int
    end_time_ms: int
    transcript_text: str
    ocr_text: str
    keywords: list[str]
    evidence_ids: list[str]

class SemanticNode(BaseModel):
    concept_id: str
    canonical_name: str
    aliases: list[str]
    embedding: list[float]
    support_count: int

class ClipEdge(BaseModel):
    from_clip: str
    to_clip: str
    edge_type: str
    weight: float

class MentionEdge(BaseModel):
    semantic_id: str
    clip_id: str
    match_text: str
    confidence: float
    source: str

class SemanticEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float

# ==========================================
# Output Model
# ==========================================

class Graph(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

# ==========================================
# Graph Assembly Module
# ==========================================

class GraphBuilder:
    """
    Assembles a unified graph from clips, semantics, and their relational edges.
    Ensures deduplication, strict validation against missing nodes, and determinism.
    """
    
    def __init__(self) -> None:
        self.nodes_map: dict[str, dict[str, Any]] = {}
        self.edges_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    def build_clip_nodes(self, clips: list[ClipNode]) -> None:
        for clip in clips:
            if clip.clip_id not in self.nodes_map:
                self.nodes_map[clip.clip_id] = {
                    "id": clip.clip_id,
                    "type": "clip",
                    "properties": {
                        "video_id": clip.video_id,
                        "start_time_ms": clip.start_time_ms,
                        "end_time_ms": clip.end_time_ms,
                        "keywords": clip.keywords.copy(),
                        # Omit heavy text (transcript/ocr) for lightweight nodes, keeping essential IDs
                        "evidence_ids": clip.evidence_ids.copy()
                    }
                }

    def build_semantic_nodes(self, semantics: list[SemanticNode]) -> None:
        for node in semantics:
            if node.concept_id not in self.nodes_map:
                self.nodes_map[node.concept_id] = {
                    "id": node.concept_id,
                    "type": "semantic",
                    "properties": {
                        "name": node.canonical_name,
                        "aliases": node.aliases.copy(),
                        "support_count": node.support_count
                        # Explicit omission: embedding is not included
                    }
                }

    def _add_edge(self, source: str, target: str, edge_type: str, weight: float, properties: dict[str, Any] | None = None) -> None:
        # Validation: Ignore edges where source or target nodes do not exist
        if source not in self.nodes_map or target not in self.nodes_map:
            return

        # Deduplication: unique key = (source, target, type)
        key = (source, target, edge_type)
        if key not in self.edges_map:
            self.edges_map[key] = {
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight,
                "properties": properties or {}
            }

    def build_clip_edges(self, edges: list[ClipEdge]) -> None:
        for edge in edges:
            self._add_edge(
                source=edge.from_clip,
                target=edge.to_clip,
                edge_type=edge.edge_type,
                weight=edge.weight
            )

    def build_mention_edges(self, edges: list[MentionEdge]) -> None:
        for edge in edges:
            self._add_edge(
                source=edge.semantic_id,
                target=edge.clip_id,
                edge_type="MENTIONED_IN",
                weight=edge.confidence,
                properties={
                    "match_text": edge.match_text,
                    "source_type": edge.source
                }
            )

    def build_semantic_edges(self, edges: list[SemanticEdge]) -> None:
        for edge in edges:
            self._add_edge(
                source=edge.source,
                target=edge.target,
                edge_type=edge.relation,
                weight=edge.weight
            )

    def assemble(self, 
                 clips: list[ClipNode], 
                 semantics: list[SemanticNode],
                 clip_edges: list[ClipEdge], 
                 mention_edges: list[MentionEdge],
                 semantic_edges: list[SemanticEdge]) -> Graph:
        
        # Reset internal state to ensure clean assembly
        self.nodes_map.clear()
        self.edges_map.clear()

        # Build nodes first (dependency for edge validation)
        self.build_clip_nodes(clips)
        self.build_semantic_nodes(semantics)

        # Build edges
        self.build_clip_edges(clip_edges)
        self.build_mention_edges(mention_edges)
        self.build_semantic_edges(semantic_edges)

        # Ensure determinism: Sort nodes by ID and edges by (source, target, type)
        sorted_nodes = sorted(list(self.nodes_map.values()), key=lambda x: x["id"])
        sorted_edges = sorted(list(self.edges_map.values()), key=lambda x: (x["source"], x["target"], x["type"]))

        return Graph(nodes=sorted_nodes, edges=sorted_edges)


def build_graph(
    clips: list[ClipNode],
    semantics: list[SemanticNode],
    clip_edges: list[ClipEdge],
    mention_edges: list[MentionEdge],
    semantic_edges: list[SemanticEdge],
) -> Graph:
    """
    Backward-compatible module API used by pipeline.py.
    """
    return GraphBuilder().assemble(
        clips=clips,
        semantics=semantics,
        clip_edges=clip_edges,
        mention_edges=mention_edges,
        semantic_edges=semantic_edges,
    )

# ==========================================
# Testing Requirement
# ==========================================

if __name__ == "__main__":
    # 1. Create mock nodes
    mock_clips = [
        ClipNode(
            clip_id="clip_001", video_id="vid_A", start_time_ms=0, end_time_ms=5000,
            transcript_text="Look at this coffee cup.", ocr_text="COFFEE", 
            keywords=["coffee", "cup"], evidence_ids=["ev_1"]
        ),
        ClipNode(
            clip_id="clip_002", video_id="vid_A", start_time_ms=5000, end_time_ms=10000,
            transcript_text="It is very hot.", ocr_text="", 
            keywords=["hot"], evidence_ids=["ev_2"]
        )
    ]
    
    mock_semantics = [
        SemanticNode(
            concept_id="concept_1", canonical_name="coffee cup", 
            aliases=["mug", "cup of coffee"], embedding=[0.1, 0.2, 0.3], support_count=5
        ),
        SemanticNode(
            concept_id="concept_2", canonical_name="heat", 
            aliases=["hot", "temperature"], embedding=[0.4, 0.5, 0.6], support_count=3
        )
    ]

    # 2. Create mock edges
    mock_clip_edges = [
        ClipEdge(from_clip="clip_001", to_clip="clip_002", edge_type="NEXT", weight=1.0)
    ]
    
    mock_mention_edges = [
        MentionEdge(semantic_id="concept_1", clip_id="clip_001", match_text="coffee cup", confidence=0.95, source="transcript"),
        MentionEdge(semantic_id="concept_2", clip_id="clip_002", match_text="hot", confidence=0.88, source="transcript"),
        # Invalid edge: target clip does not exist (will be safely skipped)
        MentionEdge(semantic_id="concept_1", clip_id="clip_999", match_text="coffee", confidence=0.5, source="ocr")
    ]
    
    mock_semantic_edges = [
        SemanticEdge(source="concept_1", target="concept_2", relation="HAS_PROPERTY", weight=0.7)
    ]

    # 3. Build graph
    builder = GraphBuilder()
    graph = builder.assemble(
        clips=mock_clips,
        semantics=mock_semantics,
        clip_edges=mock_clip_edges,
        mention_edges=mock_mention_edges,
        semantic_edges=mock_semantic_edges
    )

    # 4. Print final graph
    print(json.dumps(graph.model_dump(), indent=2))