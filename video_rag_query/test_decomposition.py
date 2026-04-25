import sys
import os
import json
import logging
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock sentence_transformers and numpy
mock_st = MagicMock()
mock_st.SentenceTransformer.return_value.encode.return_value = [[1.0, 0.0, 0.0]] * 10
sys.modules['sentence_transformers'] = mock_st

mock_np = MagicMock()
mock_np.dot.return_value = [0.95] * 10
mock_np.argmax.return_value = 0
sys.modules['numpy'] = mock_np

from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.llm_client import LLMClient
from video_rag_query.models import QueryDecomposition

logging.basicConfig(level=logging.INFO)

mock_corpus = [
    {"id": "person_barack_obama", "name": "Barack Obama"},
    {"id": "topic_climate_change", "name": "Climate Change"},
    {"id": "location_white_house", "name": "White House"},
    {"id": "person_john_doe", "name": "John Doe"}
]

queries = [
    "Was Obama at the White House before he discussed Climate Change?",
    "Find the person who discussed Climate Change and where they went next",
    "What happened after the meeting?",
    "Where was he before entering the room?"
]

def mock_llm_response(query):
    # Deterministic mock responses matching the query
    if "Was Obama at the White House" in query:
        return json.dumps({
            "query_type": "causal",
            "entities": [{"name": "Obama", "type": "person"}, {"name": "White House", "type": "location"}],
            "actions": ["discussed"],
            "temporal_constraints": {"relation": "before", "anchor_event": "discussed Climate Change", "direction": "forward"},
            "sub_queries": [{"id": "Q1", "type": "test", "goal": "test", "required_graph_components": ["APPEARS_IN"]}],
            "execution_plan": ["Resolve EntityRef(Obama) -> traverse APPEARS_IN"],
            "confidence": 0.0,
            "ambiguity_flags": []
        })
    elif "Find the person who discussed" in query:
        return json.dumps({
            "query_type": "find",
            "entities": [{"name": "person", "type": "person"}],
            "actions": ["discussed", "went"],
            "temporal_constraints": {"relation": "after", "anchor_event": "Climate Change", "direction": "none"},
            "sub_queries": [{"id": "Q1", "type": "test", "goal": "test", "required_graph_components": ["NEXT"]}],
            "execution_plan": ["Resolve EntityRef(person) -> traverse NEXT"],
            "confidence": 0.0,
            "ambiguity_flags": []
        })
    else:
        return json.dumps({
            "query_type": "info",
            "entities": [],
            "actions": [],
            "temporal_constraints": {"relation": "none", "anchor_event": "none", "direction": "none"},
            "sub_queries": [{"id": "Q1", "type": "test", "goal": "test", "required_graph_components": ["SHARES_ENTITY"]}],
            "execution_plan": ["filter ClipRef by action"],
            "confidence": 0.0,
            "ambiguity_flags": []
        })

def run_tests():
    decomposer = QueryDecomposer(cerebras_keys=["dummy_c1"], groq_keys=["dummy_g1"], entity_corpus=mock_corpus)
    
    total = 0
    passed = 0
    failures = []
    
    print("=== STARTING TESTS ===")
    
    with patch.object(LLMClient, '_call_api') as mock_call:
        for q in queries:
            total += 1
            mock_call.return_value = mock_llm_response(q)
            res = decomposer.decompose(q)
            
            # Validation Checks
            if not isinstance(res, QueryDecomposition):
                failures.append((q, "JSON/Schema Invalid"))
                continue
                
            # Entity Mapping
            entity_fail = False
            for e in res.entities:
                if e.resolved_entity_id:
                    if not e.resolved_entity_id.startswith(f"{e.type.lower()}_"):
                        entity_fail = True
            if entity_fail:
                failures.append((q, "Entity Mismatch"))
                continue
                
            # Temporal Logic
            rel = res.temporal_constraints.relation
            dir = res.temporal_constraints.direction
            if rel == "before" and dir != "backward":
                failures.append((q, "Temporal Error: before != backward"))
                continue
            if rel == "after" and dir != "forward":
                failures.append((q, "Temporal Error: after != forward"))
                continue
                
            # Execution Plan
            plan_fail = False
            for step in res.execution_plan:
                lower_step = step.lower()
                if not ("entityref" in lower_step or "clipref" in lower_step or "filter" in lower_step or "traverse" in lower_step or "extract" in lower_step):
                    plan_fail = True
            if plan_fail:
                failures.append((q, "Execution Plan Weak"))
                continue
                
            # Confidence
            if res.confidence <= 0.0:
                failures.append((q, "Confidence score <= 0.0"))
                continue
                
            passed += 1

    print(f"\nStandard Results: {passed}/{total} passed.")
    for fail in failures:
        print(f"FAILED: {fail[0]} -> {fail[1]}")
        
    print("\n=== STARTING FAILOVER TESTS ===")
    
    # 1. Invalid JSON (should retry)
    def mock_invalid_json(client, model, query):
        raise json.JSONDecodeError("Expecting value", "", 0)
        
    with patch.object(LLMClient, '_call_api', side_effect=mock_invalid_json):
        res = decomposer.decompose("test retry json")
        if getattr(res, "status", None) == "failure":
            passed += 1
            print("Invalid JSON Exhaustion Test: Passed")
        else:
            print("Invalid JSON Exhaustion Test: Failed")
            
    # 2. Total Failure
    def mock_total_failure(client, model, query):
        raise Exception("API Rate Limit")

    with patch.object(LLMClient, '_call_api', side_effect=mock_total_failure):
        res = decomposer.decompose("test total failure")
        if getattr(res, "status", None) == "failure" and res.reason == "llm_unavailable_or_invalid_output":
            passed += 1
            print("Total Failure Fallback Test: Passed")
        else:
            print("Total Failure Fallback Test: Failed")
            
    total += 2
    
    print(f"\nFinal Results: {passed}/{total} passed.")
    
if __name__ == "__main__":
    run_tests()
