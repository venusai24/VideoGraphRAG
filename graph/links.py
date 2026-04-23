"""Cross-layer linking placeholder between clip and semantic graphs."""

from __future__ import annotations

from videographrag.graph.models import GraphBundle


def link_graph_layers(clip_graph: GraphBundle, semantic_graph: GraphBundle) -> GraphBundle:
    """Create cross-layer links and return merged graph bundle (placeholder)."""
    _ = (clip_graph, semantic_graph)
    return GraphBundle()
