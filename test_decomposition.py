import os
import logging
from video_rag_query.query_decomposer import QueryDecomposer

logging.basicConfig(level=logging.INFO)

def main():
    # Load keys from environment or use dummy keys for testing failover
    cerebras_keys = os.environ.get("CEREBRAS_API_KEYS", "dummy_c1,dummy_c2").split(",")
    groq_keys = os.environ.get("GROQ_API_KEYS", "dummy_g1,dummy_g2").split(",")
    
    # Mock entity corpus for resolution testing
    mock_corpus = [
        {"id": "person_obama", "name": "Barack Obama"},
        {"id": "topic_politics", "name": "Politics"},
        {"id": "location_white_house", "name": "White House"}
    ]
    
    decomposer = QueryDecomposer(
        cerebras_keys=cerebras_keys,
        groq_keys=groq_keys,
        entity_corpus=mock_corpus
    )
    
    query = "Was Obama at the White House before he discussed politics?"
    print(f"\nTesting Query: {query}\n")
    
    result = decomposer.decompose(query)
    
    if hasattr(result, "model_dump_json"):
        print("\n--- FINAL JSON OUTPUT ---")
        print(result.model_dump_json(indent=2))
    else:
        print("\n--- FAILURE OUTPUT ---")
        print(result)

if __name__ == "__main__":
    main()
