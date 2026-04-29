# Deprecated – not used in final pipeline
import sys
import os
import json
import logging
from unittest.mock import patch, MagicMock

# Ensure we can import from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- MOCKING DEPENDENCIES FOR DETERMINISTIC DEMO ---
# (SentenceTransformer can be slow/heavy, so we mock for the demo)
mock_st = MagicMock()
mock_st.SentenceTransformer.return_value.encode.return_value = [[1.0, 0.0, 0.0]] * 10
sys.modules['sentence_transformers'] = mock_st

mock_np = MagicMock()
mock_np.dot.return_value = [0.98] * 10
mock_np.argmax.return_value = 0
sys.modules['numpy'] = mock_np

from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.llm_client import LLMClient
from video_rag_query.models import QueryDecomposition

logging.basicConfig(level=logging.WARNING)

# Sample Entity Corpus
demo_corpus = [
    {"id": "person_1", "name": "Man in Red Shirt"},
    {"id": "topic_running", "name": "Running"},
    {"id": "person_2", "name": "John"},
    {"id": "location_office", "name": "Office Room"}
]

demo_queries = [
    "Who is the person in the red shirt?",
    "Show me all clips where someone is running.",
    "Did John enter the room before or after the meeting started?",
    "Where was John before he went to the office?"
]

def get_mock_llm_logic(query):
    """Generates realistic graph-aligned responses for the demo."""
    if "red shirt" in query.lower():
        return {
            "query_type": "lookup",
            "entities": [{"name": "Man in Red Shirt", "type": "person"}],
            "actions": ["identify"],
            "temporal_constraints": {"relation": "none", "anchor_event": None, "direction": "none"},
            "sub_queries": [{"id": "Q1", "type": "find_entity", "goal": "Find clip where man in red shirt appears", "required_graph_components": ["APPEARS_IN"]}],
            "execution_plan": ["Resolve EntityRef(person_1) -> traverse APPEARS_IN -> extract timestamps"],
            "confidence": 0.98,
            "ambiguity_flags": []
        }
    elif "running" in query.lower():
        return {
            "query_type": "retrieval",
            "entities": [{"name": "Running", "type": "topic"}],
            "actions": ["running"],
            "temporal_constraints": {"relation": "none", "anchor_event": None, "direction": "none"},
            "sub_queries": [{"id": "Q1", "type": "action_search", "goal": "Find clips associated with running", "required_graph_components": ["APPEARS_IN", "RELATED_TO"]}],
            "execution_plan": ["Resolve EntityRef(topic_running) -> traverse APPEARS_IN -> filter by confidence"],
            "confidence": 0.95,
            "ambiguity_flags": []
        }
    elif "before" in query.lower() or "after" in query.lower():
        return {
            "query_type": "temporal_reasoning",
            "entities": [{"name": "John", "type": "person"}, {"name": "Office Room", "type": "location"}],
            "actions": ["enter", "went"],
            "temporal_constraints": {"relation": "before", "anchor_event": "entering office", "direction": "backward"},
            "sub_queries": [
                {"id": "Q1", "id": "Q1", "type": "find_event", "goal": "Find John entering the room", "required_graph_components": ["APPEARS_IN"]},
                {"id": "Q2", "type": "temporal_step", "goal": "Find preceding clips", "required_graph_components": ["NEXT"]}
            ],
            "execution_plan": [
                "Resolve EntityRef(person_2) -> traverse APPEARS_IN(Office Room)",
                "From event anchor -> traverse NEXT (backward) -> filter clips"
            ],
            "confidence": 0.92,
            "ambiguity_flags": []
        }
    return {
        "query_type": "general",
        "entities": [],
        "actions": [],
        "temporal_constraints": {"relation": "none", "anchor_event": None, "direction": "none"},
        "sub_queries": [],
        "execution_plan": ["Generic graph traversal"],
        "confidence": 0.5,
        "ambiguity_flags": ["no_entities_found"]
    }

def run_demo():
    decomposer = QueryDecomposer(cerebras_keys=["demo"], groq_keys=["demo"], entity_corpus=demo_corpus)
    
    print("\n" + "="*60)
    print("VIDEOGRAPH-RAG QUERY DECOMPOSITION DEMO")
    print("="*60)
    
    with patch.object(LLMClient, '_call_api') as mock_call:
        for q in demo_queries:
            mock_call.return_value = json.dumps(get_mock_llm_logic(q))
            result = decomposer.decompose(q)
            
            print(f"\nQUERY: {q}")
            print("-" * len(q))
            if hasattr(result, "query_type"):
                print(f"Plan Type: {result.query_type}")
                print(f"Entities: {[e.name + ' (' + (e.resolved_entity_id or 'unresolved') + ')' for e in result.entities]}")
                print(f"Temporal: {result.temporal_constraints.relation} (Direction: {result.temporal_constraints.direction})")
                print(f"Execution Steps:")
                for step in result.execution_plan:
                    print(f"  - {step}")
                print(f"Overall Confidence: {result.confidence:.2f}")
            else:
                print(f"FAILED: {result.reason}")

if __name__ == "__main__":
    run_demo()
