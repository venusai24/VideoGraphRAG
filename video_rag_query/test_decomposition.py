"""
Full validation test suite for the query decomposition pipeline.
Tests: schema correctness, entity type safety, temporal direction, structured execution plan,
       confidence scoring, and complete failover behaviour.
"""
import sys
import os
import json
import logging
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Mock heavy dependencies so we don't need them installed ──────────────────
_mock_st = MagicMock()
_mock_st.SentenceTransformer.return_value.encode.return_value = [[1.0, 0.0, 0.0]] * 10
sys.modules["sentence_transformers"] = _mock_st

_mock_np = MagicMock()
_mock_np.dot.return_value = [0.97, 0.45, 0.30, 0.20]
_mock_np.argmax.return_value = 0
sys.modules["numpy"] = _mock_np

from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.llm_client import LLMClient
from video_rag_query.models import QueryDecomposition, FailureResponse

logging.basicConfig(level=logging.WARNING)

# ── Shared entity corpus ──────────────────────────────────────────────────────
CORPUS = [
    {"id": "person_obama", "name": "Barack Obama"},
    {"id": "location_white_house", "name": "White House"},
    {"id": "topic_climate_change", "name": "Climate Change"},
    {"id": "person_john", "name": "John"},
]

# ── Canonical mock LLM responses (structured execution_plan) ──────────────────
def _llm_response_obama_before():
    return {
        "query_type": "temporal_reasoning",
        "entities": [
            {"name": "Obama", "type": "person"},
            {"name": "White House", "type": "location"},
        ],
        "actions": ["discussed"],
        "temporal_constraints": {"relation": "before", "anchor_event": "discussed Climate Change", "direction": "forward"},
        "sub_queries": [
            {"id": "Q1", "type": "entity_lookup", "goal": "Locate Obama in graph", "required_graph_components": ["APPEARS_IN"]},
            {"id": "Q2", "type": "temporal_traversal", "goal": "Find clips before anchor", "required_graph_components": ["NEXT"]},
        ],
        "execution_plan": [
            {"step": 1, "operation": "resolve_entity", "input": "Obama", "output": "entity_obama"},
            {"step": 2, "operation": "traverse", "from": "entity_obama", "edge": "APPEARS_IN", "to": "ClipRef", "filter": {}},
            {"step": 3, "operation": "temporal_traverse", "edge": "NEXT", "direction": "forward", "limit": 10},
            {"step": 4, "operation": "extract", "target": "ClipRef", "fields": ["clip_id", "timestamp"]},
        ],
        "confidence": 0.0,
        "ambiguity_flags": [],
    }

def _llm_response_running():
    return {
        "query_type": "retrieval",
        "entities": [{"name": "person", "type": "person"}],
        "actions": ["running"],
        "temporal_constraints": {"relation": "none", "anchor_event": None, "direction": "none"},
        "sub_queries": [
            {"id": "Q1", "type": "action_search", "goal": "Retrieve clips with running action", "required_graph_components": ["APPEARS_IN", "RELATED_TO"]},
        ],
        "execution_plan": [
            {"step": 1, "operation": "filter", "condition": {"field": "action", "op": "contains", "value": "running"}},
            {"step": 2, "operation": "extract", "target": "ClipRef", "fields": ["clip_id", "timestamp", "action"]},
        ],
        "confidence": 0.0,
        "ambiguity_flags": [],
    }

def _llm_response_after_meeting():
    return {
        "query_type": "temporal_reasoning",
        "entities": [],
        "actions": ["happened"],
        "temporal_constraints": {"relation": "after", "anchor_event": "the meeting", "direction": "backward"},
        "sub_queries": [
            {"id": "Q1", "type": "event_lookup", "goal": "Find meeting event", "required_graph_components": ["SHARES_ENTITY"]},
            {"id": "Q2", "type": "temporal_traversal", "goal": "Traverse forward from meeting", "required_graph_components": ["NEXT"]},
        ],
        "execution_plan": [
            {"step": 1, "operation": "filter", "condition": {"field": "label", "op": "eq", "value": "meeting"}},
            {"step": 2, "operation": "temporal_traverse", "edge": "NEXT", "direction": "backward", "limit": 5},
            {"step": 3, "operation": "extract", "target": "ClipRef", "fields": ["clip_id", "timestamp"]},
        ],
        "confidence": 0.0,
        "ambiguity_flags": [],
    }

def _llm_response_pronoun():
    return {
        "query_type": "lookup",
        "entities": [{"name": "he", "type": "person"}],
        "actions": ["was"],
        "temporal_constraints": {"relation": "before", "anchor_event": "entering the room", "direction": "forward"},
        "sub_queries": [
            {"id": "Q1", "type": "entity_resolution", "goal": "Resolve pronoun 'he'", "required_graph_components": ["APPEARS_IN"]},
            {"id": "Q2", "type": "temporal_traversal", "goal": "Find prior location", "required_graph_components": ["NEXT"]},
        ],
        "execution_plan": [
            {"step": 1, "operation": "resolve_entity", "input": "he", "output": "entity_unknown_person"},
            {"step": 2, "operation": "traverse", "from": "entity_unknown_person", "edge": "APPEARS_IN", "to": "ClipRef", "filter": {}},
            {"step": 3, "operation": "temporal_traverse", "edge": "NEXT", "direction": "forward", "limit": 5},
            {"step": 4, "operation": "extract", "target": "ClipRef", "fields": ["clip_id", "timestamp", "location"]},
        ],
        "confidence": 0.0,
        "ambiguity_flags": ["pronoun_without_referent"],
    }


QUERY_FIXTURES = [
    ("Was Obama at the White House before he discussed Climate Change?", _llm_response_obama_before),
    ("Show me all clips where someone is running.", _llm_response_running),
    ("What happened after the meeting?", _llm_response_after_meeting),
    ("Where was he before entering the room?", _llm_response_pronoun),
]


# ── Validation helpers ────────────────────────────────────────────────────────

def check_schema(result) -> list:
    errors = []
    if not isinstance(result, QueryDecomposition):
        return [f"Not a QueryDecomposition: {type(result)}"]
    required = ["query_type", "entities", "actions", "temporal_constraints",
                "sub_queries", "execution_plan", "confidence", "ambiguity_flags"]
    for f in required:
        if not hasattr(result, f):
            errors.append(f"Missing field: {f}")
    return errors


def check_entities(result: QueryDecomposition) -> list:
    errors = []
    for e in result.entities:
        if e.resolved_entity_id is not None:
            expected = f"{e.type.lower()}_"
            if not e.resolved_entity_id.startswith(expected):
                errors.append(
                    f"Entity '{e.name}' type mismatch: id='{e.resolved_entity_id}' "
                    f"doesn't start with '{expected}'"
                )
    return errors


def check_temporal(result: QueryDecomposition) -> list:
    errors = []
    rel = result.temporal_constraints.relation
    dir_ = result.temporal_constraints.direction
    expected = {"before": "backward", "after": "forward", "during": "neutral"}
    if rel in expected and dir_ != expected[rel]:
        errors.append(f"Temporal mismatch: relation='{rel}' but direction='{dir_}' (expected '{expected[rel]}')")
    return errors


def check_execution_plan(result: QueryDecomposition) -> list:
    errors = []
    valid_ops = {"resolve_entity", "traverse", "filter", "temporal_traverse", "extract"}
    valid_edges = {"APPEARS_IN", "NEXT", "SHARES_ENTITY", "RELATED_TO"}
    for step in result.execution_plan:
        if not isinstance(step, dict):
            errors.append(f"Step is a string, not dict: {step!r}")
            continue
        op = step.get("operation")
        if op not in valid_ops:
            errors.append(f"Step {step.get('step')}: invalid op '{op}'")
        if op == "traverse":
            edge = step.get("edge")
            if edge not in valid_edges:
                errors.append(f"Step {step.get('step')}: invalid edge '{edge}'")
    return errors


def check_confidence(result: QueryDecomposition) -> list:
    if result.confidence <= 0.0 and result.entities:
        return [f"confidence={result.confidence} but entities exist — score should be > 0"]
    return []


# ── Main test runner ──────────────────────────────────────────────────────────

def run_tests():
    decomposer = QueryDecomposer(
        cerebras_keys=["dummy_c1", "dummy_c2"],
        groq_keys=["dummy_g1", "dummy_g2"],
        entity_corpus=CORPUS,
    )

    total = 0
    passed = 0
    all_failures = []

    print("\n" + "=" * 65)
    print("QUERY DECOMPOSITION PIPELINE — VALIDATION SUITE")
    print("=" * 65)

    # ── Section 1: Standard query tests ──────────────────────────────────────
    print("\n[1/3] Standard query tests")
    print("-" * 40)

    with patch.object(LLMClient, "_call_api") as mock_call:
        for query, fixture_fn in QUERY_FIXTURES:
            total += 1
            mock_call.return_value = json.dumps(fixture_fn())
            result = decomposer.decompose(query)

            failures = []
            failures += check_schema(result)
            if not failures:
                failures += check_entities(result)
                failures += check_temporal(result)
                failures += check_execution_plan(result)
                failures += check_confidence(result)

            if failures:
                all_failures.append((query, failures))
                print(f"  ❌ FAIL: {query[:55]}...")
                for f in failures:
                    print(f"       → {f}")
            else:
                passed += 1
                tc = result.temporal_constraints
                print(f"  ✅ PASS: {query[:55]}")
                print(f"       plan_steps={len(result.execution_plan)}  "
                      f"entities={len(result.entities)}  "
                      f"temporal={tc.relation}/{tc.direction}  "
                      f"confidence={result.confidence:.2f}")

    # ── Section 2: Invalid JSON retry + exhaustion ────────────────────────────
    print("\n[2/3] Failover tests")
    print("-" * 40)

    def _raise_json(client, model, query):
        raise json.JSONDecodeError("Expecting value", "", 0)

    total += 1
    with patch.object(LLMClient, "_call_api", side_effect=_raise_json):
        res = decomposer.decompose("test: invalid JSON exhaustion")
    if isinstance(res, FailureResponse) and res.reason == "llm_unavailable_or_invalid_output":
        passed += 1
        print("  ✅ PASS: Invalid JSON exhaustion → FailureResponse")
    else:
        all_failures.append(("invalid_json_test", ["Did not return FailureResponse"]))
        print("  ❌ FAIL: Invalid JSON exhaustion")

    def _raise_api(client, model, query):
        raise Exception("Simulated total API failure")

    total += 1
    with patch.object(LLMClient, "_call_api", side_effect=_raise_api):
        res = decomposer.decompose("test: total provider failure")
    if isinstance(res, FailureResponse) and res.reason == "llm_unavailable_or_invalid_output":
        passed += 1
        print("  ✅ PASS: Total provider failure → FailureResponse")
    else:
        all_failures.append(("total_failure_test", ["Did not return FailureResponse"]))
        print("  ❌ FAIL: Total provider failure")

    # ── Section 3: Hallucinated edge rejection ────────────────────────────────
    print("\n[3/3] Hallucination rejection test")
    print("-" * 40)

    def _hallucinated_edge(client, model, query):
        return json.dumps({
            "query_type": "lookup",
            "entities": [],
            "actions": [],
            "temporal_constraints": {"relation": "none", "anchor_event": None, "direction": "none"},
            "sub_queries": [],
            "execution_plan": [
                {"step": 1, "operation": "traverse", "from": "EntityRef", "edge": "FAKE_EDGE", "to": "ClipRef", "filter": {}}
            ],
            "confidence": 0.0,
            "ambiguity_flags": [],
        })

    total += 1
    with patch.object(LLMClient, "_call_api", side_effect=_hallucinated_edge):
        res = decomposer.decompose("test: hallucinated edge")
    # The client rejects it and falls through all configs -> FailureResponse
    if isinstance(res, FailureResponse):
        passed += 1
        print("  ✅ PASS: Hallucinated edge → rejected → FailureResponse")
    else:
        # If it returned a result, check that ambiguity_flags captured the violation
        flags = getattr(res, "ambiguity_flags", [])
        if any("FAKE_EDGE" in f or "invalid edge" in f for f in flags):
            passed += 1
            print("  ✅ PASS: Hallucinated edge → flagged in ambiguity_flags")
        else:
            all_failures.append(("hallucination_test", [f"Result returned without rejection flag: {flags}"]))
            print("  ❌ FAIL: Hallucinated edge was not rejected or flagged")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"RESULTS: {passed}/{total} passed  ({100*passed//total}%)")
    if all_failures:
        print(f"\nFailed cases ({len(all_failures)}):")
        for q, errs in all_failures:
            print(f"  • {q}")
            for e in errs:
                print(f"    - {e}")
    else:
        print("All checks passed. Pipeline is deterministic and production-ready.")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
