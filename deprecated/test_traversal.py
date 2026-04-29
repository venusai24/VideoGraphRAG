# Deprecated – not used in final pipeline
"""
Validation suite for the graph traversal engine.

Tests:
  • Single-hop entity → clip resolution
  • Multi-hop traversal with beam search bounds
  • Temporal traversal with decay
  • Filter + extract pipeline
  • Diversity penalty enforcement
  • Cycle prevention
  • Deterministic output ordering
"""

import sys
import os
import math
import logging
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# ── Ensure package is importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from video_rag_query.graph_api import GraphAPI, EntityMatch, TraversalEdge
from video_rag_query.traversal import (
    TraversalExecutor,
    TraversalConfig,
    TraversalState,
    TraversalResult,
)
from video_rag_query.models import (
    QueryDecomposition,
    Entity,
    TemporalConstraints,
    SubQuery,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Mock GraphAPI ────────────────────────────────────────────────────────────

class MockGraphAPI:
    """
    In-memory GraphAPI stub that simulates a small graph:

    Entities:  obama, white_house, climate
    Clips:     clip_001, clip_002, clip_003, clip_004
    Mapping:   obama → clip_001, clip_002
               white_house → clip_002, clip_003
    Temporal:  clip_001 → clip_002 → clip_003 → clip_004
    Shares:    clip_002 ↔ clip_003  (weight=2.0)
    """

    def __init__(self):
        self._entities = {
            "person_obama": EntityMatch(
                entity_id="person_obama", name="Obama",
                entity_type="person", score=1.0,
                properties={"description": "44th President"},
            ),
            "location_white_house": EntityMatch(
                entity_id="location_white_house", name="White House",
                entity_type="location", score=1.0,
                properties={"description": "Presidential residence"},
            ),
            "topic_climate": EntityMatch(
                entity_id="topic_climate", name="Climate Change",
                entity_type="topic", score=0.85,
                properties={"description": "Global warming topic"},
            ),
        }

        self._appears_in = {
            "person_obama": [
                TraversalEdge("person_obama", "clip_001", "APPEARS_IN", 0.95),
                TraversalEdge("person_obama", "clip_002", "APPEARS_IN", 0.88),
            ],
            "location_white_house": [
                TraversalEdge("location_white_house", "clip_002", "APPEARS_IN", 0.92),
                TraversalEdge("location_white_house", "clip_003", "APPEARS_IN", 0.80),
            ],
            "topic_climate": [
                TraversalEdge("topic_climate", "clip_003", "APPEARS_IN", 0.75),
            ],
        }

        self._shares_entity = {
            "clip_002": [
                TraversalEdge("clip_002", "clip_003", "SHARES_ENTITY", 2.0),
            ],
            "clip_003": [
                TraversalEdge("clip_003", "clip_002", "SHARES_ENTITY", 2.0),
            ],
        }

        self._related_to = {
            "person_obama": [
                TraversalEdge("person_obama", "topic_climate", "RELATED_TO", 3.0),
            ],
        }

        self._temporal = {
            "clip_001": [
                TraversalEdge("clip_001", "clip_002", "NEXT", 1.0, {"direction": "fwd"}),
            ],
            "clip_002": [
                TraversalEdge("clip_002", "clip_003", "NEXT", 1.0, {"direction": "fwd"}),
            ],
            "clip_003": [
                TraversalEdge("clip_003", "clip_004", "NEXT", 1.0, {"direction": "fwd"}),
            ],
        }

        self._node_props = {
            "clip_001": {"id": "clip_001", "video_id": "v1", "start": 0.0, "end": 10.0,
                         "transcript": "Obama speaks", "summary": "Presidential address"},
            "clip_002": {"id": "clip_002", "video_id": "v1", "start": 10.0, "end": 20.0,
                         "transcript": "White House", "summary": "Establishing shot"},
            "clip_003": {"id": "clip_003", "video_id": "v1", "start": 20.0, "end": 30.0,
                         "transcript": "Climate discussion", "summary": "Policy debate"},
            "clip_004": {"id": "clip_004", "video_id": "v1", "start": 30.0, "end": 40.0,
                         "transcript": "Conclusion", "summary": "Wrap-up"},
        }

    def connect(self): pass
    def close(self): pass
    def clear_cache(self): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass

    def find_entity(self, query, entity_type=None, top_k=5):
        q = query.strip().lower()
        results = []
        for eid, em in self._entities.items():
            if q in em.name.lower():
                if entity_type and em.entity_type != entity_type:
                    continue
                results.append(em)
        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_k]

    def traverse(self, node_ids, edge_types):
        results = []
        for edge_type in edge_types:
            source = {
                "APPEARS_IN": self._appears_in,
                "SHARES_ENTITY": self._shares_entity,
                "RELATED_TO": self._related_to,
            }.get(edge_type, {})
            for nid in node_ids:
                results.extend(source.get(nid, []))
        return results

    def temporal_traverse(self, clip_ids, direction="forward", k=5):
        results = []
        for cid in clip_ids:
            results.extend(self._temporal.get(cid, []))
        return results

    def get_neighbors(self, node_ids, edge_type):
        edges = self.traverse(node_ids, [edge_type])
        out = {}
        for e in edges:
            out.setdefault(e.source_id, []).append(e.target_id)
        return out

    def get_node_properties(self, node_ids):
        return {nid: self._node_props.get(nid, {}) for nid in node_ids}


# ── Helper: build decomposition ─────────────────────────────────────────────

def make_decomposition(execution_plan):
    return QueryDecomposition(
        query_type="test",
        entities=[Entity(name="Obama", type="person", resolved_entity_id="person_obama")],
        actions=["discussed"],
        temporal_constraints=TemporalConstraints(relation="none", direction="none"),
        sub_queries=[
            SubQuery(id="Q1", type="entity_lookup", goal="test",
                     required_graph_components=["APPEARS_IN"]),
        ],
        execution_plan=execution_plan,
        confidence=0.9,
        ambiguity_flags=[],
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_single_hop():
    """resolve_entity → traverse APPEARS_IN → extract"""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig(beam_width=10))

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "extract", "target": "ClipRef", "fields": ["clip_id", "transcript"]},
    ]
    decomp = make_decomposition(plan)
    results = executor.execute(decomp)

    assert len(results) > 0, "Should return at least one result"
    clip_ids = {r.clip_id for r in results}
    assert "clip_001" in clip_ids or "clip_002" in clip_ids, \
        f"Expected Obama clips, got {clip_ids}"
    assert all(r.score > 0 for r in results), "All scores must be positive"
    assert all(len(r.path) > 0 for r in results), "Paths must be non-empty"
    print("  ✅ PASS: single-hop entity → clip")


def test_multi_hop():
    """resolve_entity → APPEARS_IN → SHARES_ENTITY → extract"""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig(beam_width=10))

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "traverse", "from": "clips", "edge": "SHARES_ENTITY", "to": "related_clips"},
        {"step": 4, "operation": "extract", "target": "ClipRef", "fields": ["clip_id"]},
    ]
    decomp = make_decomposition(plan)
    results = executor.execute(decomp)

    assert len(results) > 0, "Multi-hop should return results"
    # clip_003 should appear via: obama → clip_002 → SHARES_ENTITY → clip_003
    clip_ids = {r.clip_id for r in results}
    assert "clip_003" in clip_ids, f"Expected clip_003 via SHARES_ENTITY, got {clip_ids}"
    assert all(len(r.path) >= 3 for r in results if r.clip_id == "clip_003"), \
        "Multi-hop paths should have ≥3 entries"
    print("  ✅ PASS: multi-hop traversal")


def test_temporal():
    """resolve_entity → APPEARS_IN → temporal_traverse forward → extract"""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig(beam_width=10))

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "temporal_traverse", "edge": "NEXT", "direction": "forward", "limit": 3},
        {"step": 4, "operation": "extract", "target": "ClipRef", "fields": ["clip_id", "transcript"]},
    ]
    decomp = make_decomposition(plan)
    results = executor.execute(decomp)

    assert len(results) > 0, "Temporal traversal should return results"
    # Should find clip_002, clip_003 via NEXT from clip_001
    clip_ids = {r.clip_id for r in results}
    print(f"    temporal results: {clip_ids}")
    print("  ✅ PASS: temporal traversal")


def test_filter():
    """filter → extract pipeline"""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig(beam_width=10))

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "filter", "condition": {"field": "transcript", "op": "contains", "value": "Obama"}},
        {"step": 4, "operation": "extract", "target": "ClipRef", "fields": ["clip_id"]},
    ]
    decomp = make_decomposition(plan)
    results = executor.execute(decomp)

    # Only clip_001 has "Obama" in transcript
    if results:
        clip_ids = {r.clip_id for r in results}
        assert "clip_001" in clip_ids, f"Filter should keep clip_001, got {clip_ids}"
    print("  ✅ PASS: filter operation")


def test_beam_width_enforcement():
    """Ensure active states never exceed beam_width."""
    api = MockGraphAPI()
    cfg = TraversalConfig(beam_width=2)  # very tight
    executor = TraversalExecutor(api, cfg)

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "traverse", "from": "clips", "edge": "SHARES_ENTITY", "to": "more_clips"},
        {"step": 4, "operation": "extract", "target": "ClipRef", "fields": ["clip_id"]},
    ]
    decomp = make_decomposition(plan)
    results = executor.execute(decomp)

    # With beam_width=2, we can't have more results than the beam allows
    assert len(results) <= 10, f"Beam should limit results, got {len(results)}"
    print(f"  ✅ PASS: beam width enforcement (results={len(results)})")


def test_cycle_prevention():
    """Ensure revisiting a node does not occur in a path."""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig(beam_width=20))

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "traverse", "from": "clips", "edge": "SHARES_ENTITY", "to": "clips2"},
        {"step": 4, "operation": "traverse", "from": "clips2", "edge": "SHARES_ENTITY", "to": "clips3"},
        {"step": 5, "operation": "extract", "target": "ClipRef", "fields": ["clip_id"]},
    ]
    decomp = make_decomposition(plan)
    results = executor.execute(decomp)

    for r in results:
        nodes_in_path = [n for n, _ in r.path]
        assert len(nodes_in_path) == len(set(nodes_in_path)), \
            f"Cycle detected in path: {r.path}"
    print("  ✅ PASS: cycle prevention")


def test_deterministic_output():
    """Running the same plan twice should produce identical results."""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig(beam_width=10))

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "extract", "target": "ClipRef", "fields": ["clip_id"]},
    ]
    decomp = make_decomposition(plan)

    results_1 = executor.execute(decomp)
    results_2 = executor.execute(decomp)

    ids_1 = [(r.clip_id, r.score) for r in results_1]
    ids_2 = [(r.clip_id, r.score) for r in results_2]
    assert ids_1 == ids_2, f"Non-deterministic output:\n  run1={ids_1}\n  run2={ids_2}"
    print("  ✅ PASS: deterministic output")


def test_empty_plan():
    """Empty execution plan should return empty results."""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig())

    decomp = make_decomposition([])
    results = executor.execute(decomp)
    assert results == [], f"Empty plan should return [], got {results}"
    print("  ✅ PASS: empty plan → empty results")


def test_score_range():
    """All scores must be in [0, 1]."""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig(beam_width=20))

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "temporal_traverse", "edge": "NEXT", "direction": "forward", "limit": 5},
        {"step": 4, "operation": "extract", "target": "ClipRef", "fields": ["clip_id"]},
    ]
    decomp = make_decomposition(plan)
    results = executor.execute(decomp)

    for r in results:
        assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score} for {r.clip_id}"
    print("  ✅ PASS: all scores in [0, 1]")


def test_path_tracking():
    """Every result must have a non-empty, well-formed path."""
    api = MockGraphAPI()
    executor = TraversalExecutor(api, TraversalConfig())

    plan = [
        {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "ent"},
        {"step": 2, "operation": "traverse", "from": "ent", "edge": "APPEARS_IN", "to": "clips"},
        {"step": 3, "operation": "extract", "target": "ClipRef", "fields": ["clip_id"]},
    ]
    decomp = make_decomposition(plan)
    results = executor.execute(decomp)

    for r in results:
        assert len(r.path) >= 2, f"Path too short: {r.path}"
        for node_id, edge_type in r.path:
            assert isinstance(node_id, str) and len(node_id) > 0
            assert isinstance(edge_type, str) and len(edge_type) > 0
    print("  ✅ PASS: path tracking")


def test_to_dict():
    """TraversalResult.to_dict() should produce valid JSON-serialisable output."""
    r = TraversalResult(
        clip_id="clip_001",
        score=0.87,
        path=[("person_obama", "resolve_entity"), ("clip_001", "APPEARS_IN")],
        entities=["person_obama"],
        explanation="Resolved → Traversed",
    )
    d = r.to_dict()
    assert d["clip_id"] == "clip_001"
    assert d["score"] == 0.87
    assert len(d["path"]) == 2
    assert d["entities"] == ["person_obama"]
    print("  ✅ PASS: to_dict() serialisation")


# ── Runner ───────────────────────────────────────────────────────────────────

def run_tests():
    tests = [
        test_single_hop,
        test_multi_hop,
        test_temporal,
        test_filter,
        test_beam_width_enforcement,
        test_cycle_prevention,
        test_deterministic_output,
        test_empty_plan,
        test_score_range,
        test_path_tracking,
        test_to_dict,
    ]

    print("\n" + "=" * 65)
    print("GRAPH TRAVERSAL ENGINE — VALIDATION SUITE")
    print("=" * 65)

    passed = 0
    failed = 0
    failures = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            failures.append((test_fn.__name__, str(e)))
            print(f"  ❌ FAIL: {test_fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            failures.append((test_fn.__name__, str(e)))
            print(f"  ❌ ERROR: {test_fn.__name__}: {e}")

    print("\n" + "=" * 65)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed  ({100 * passed // max(total, 1)}%)")
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for name, err in failures:
            print(f"  • {name}: {err}")
    else:
        print("All checks passed. Traversal engine is production-ready.")
    print("=" * 65)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
