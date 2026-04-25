import os
import sys
import json
import logging
from unittest.mock import patch, MagicMock

# Mock sentence_transformers and numpy before importing the decomposer
mock_st = MagicMock()
mock_st.SentenceTransformer.return_value.encode.return_value = [[1.0, 0.0, 0.0]] * 10
sys.modules['sentence_transformers'] = mock_st

mock_np = MagicMock()
mock_np.dot.return_value = [0.9] * 10
mock_np.argmax.return_value = 0
sys.modules['numpy'] = mock_np

from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.llm_client import LLMClient

logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)

def load_env():
    env_path = ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val.strip('"')

load_env()

cerebras_keys = os.environ.get("CEREBRAS_API_KEYS", "dummy").split(",")
groq_keys = os.environ.get("GROQ_API_KEYS", "dummy").split(",")

mock_corpus = [
    {"id": "person_barack_obama", "name": "Barack Obama"},
    {"id": "topic_climate_change", "name": "Climate Change"},
    {"id": "location_white_house", "name": "White House"},
    {"id": "person_john_doe", "name": "John Doe"}
]

queries = [
    "Who is Barack Obama?", # Simple lookup
    "Was Obama at the White House before he discussed Climate Change?", # Temporal query
    "Find the person who discussed Climate Change, and then show me where they went next.", # Multi-hop query
    "When he was at the White House, did he smile?" # Ambiguous query with pronouns
]

decomposer = QueryDecomposer(
    cerebras_keys=cerebras_keys,
    groq_keys=groq_keys,
    entity_corpus=mock_corpus
)

results = []

print("--- RUNNING STANDARD QUERIES ---")
for q in queries:
    res = decomposer.decompose(q)
    results.append({
        "query": q,
        "output": json.loads(res.model_dump_json()) if hasattr(res, 'model_dump_json') else str(res)
    })

print("\n--- RUNNING FAILOVER SIMULATION ---")
# Patch the _call_api to simulate rate limit for Cerebras
original_call = LLMClient._call_api

def mock_call_api(self, client, model, query):
    if "cerebras" in client.base_url.host:
        raise Exception("Simulated Rate Limit or Failure on Cerebras")
    return original_call(self, client, model, query)

with patch.object(LLMClient, '_call_api', new=mock_call_api):
    res = decomposer.decompose("Simulate failure fallback to Groq")
    results.append({
        "query": "Simulate failure fallback to Groq",
        "output": json.loads(res.model_dump_json()) if hasattr(res, 'model_dump_json') else str(res)
    })

# Now simulate TOTAL failure
def mock_total_failure(self, client, model, query):
    raise Exception("Simulated Total Failure")

with patch.object(LLMClient, '_call_api', new=mock_total_failure):
    res = decomposer.decompose("Simulate total failure")
    results.append({
        "query": "Simulate total failure",
        "output": json.loads(res.model_dump_json()) if hasattr(res, 'model_dump_json') else str(res)
    })

# Write markdown report
md_content = "# E2E Test Results\n\n"
for r in results:
    md_content += f"### Query: {r['query']}\n"
    md_content += "```json\n"
    md_content += json.dumps(r['output'], indent=2)
    md_content += "\n```\n\n"

with open("e2e_results.md", "w") as f:
    f.write(md_content)

print("Test complete. Results saved to e2e_results.md")
