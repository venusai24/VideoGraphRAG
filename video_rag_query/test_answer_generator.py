"""
Real Azure OpenAI integration tests for AnswerGenerator.

No mocks, no fake clients, no synthetic model responses.
"""

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from video_rag_query.answer_generator import (
    AnswerGenerator,
    load_and_validate_azure_openai_env,
)


def _load_real_traversal_results() -> List[Dict[str, Any]]:
    """
    Load real traversal output JSON from disk.

    Required env:
      VIDEO_RAG_TRAVERSAL_RESULTS_PATH=/abs/or/relative/path/to/traversal_results.json
    """
    results_path = os.getenv("VIDEO_RAG_TRAVERSAL_RESULTS_PATH", "").strip()
    if not results_path:
        raise RuntimeError("Missing required environment variable: VIDEO_RAG_TRAVERSAL_RESULTS_PATH")

    if not os.path.isabs(results_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        results_path = os.path.join(repo_root, results_path)

    if not os.path.exists(results_path):
        raise RuntimeError(f"Traversal results file not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "retrieved_results" in payload and isinstance(payload["retrieved_results"], list):
            payload = payload["retrieved_results"]
        else:
            raise RuntimeError(
                "Traversal results JSON must be a list or contain 'retrieved_results' list"
            )

    if not isinstance(payload, list):
        raise RuntimeError("Traversal results payload must be a list")

    if not payload:
        raise RuntimeError("No grounding evidence found in traversal results payload")

    return payload


def _assert_output_shape(output: Dict[str, Any], evidence_clip_ids: set[str]) -> None:
    required_keys = {"answer", "citations", "reasoning", "confidence"}
    missing = [key for key in required_keys if key not in output]
    if missing:
        raise AssertionError(f"Missing output keys: {missing}")

    if not isinstance(output["answer"], str) or not output["answer"].strip():
        raise AssertionError(f"Invalid answer field: {output['answer']!r}")

    if not isinstance(output["reasoning"], str) or not output["reasoning"].strip():
        raise AssertionError(f"Invalid reasoning field: {output['reasoning']!r}")

    if not isinstance(output["citations"], list):
        raise AssertionError(f"Citations must be list, got {type(output['citations'])}")

    invalid_citations = [cid for cid in output["citations"] if cid not in evidence_clip_ids]
    if invalid_citations:
        raise AssertionError(f"Citations not grounded in input evidence: {invalid_citations}")

    confidence = output["confidence"]
    if not isinstance(confidence, (int, float)):
        raise AssertionError(f"Confidence must be numeric, got {type(confidence)}")

    if confidence < 0.0 or confidence > 1.0:
        raise AssertionError(f"Confidence out of range [0,1]: {confidence}")


def test_real_answer_generator_live_api() -> None:
    load_and_validate_azure_openai_env()

    evidence = _load_real_traversal_results()
    evidence_clip_ids = {
        str(item.get("clip_id", ""))
        for item in evidence
        if isinstance(item, dict) and item.get("clip_id")
    }
    if not evidence_clip_ids:
        raise RuntimeError("No clip_id values found in traversal evidence")

    generator = AnswerGenerator.from_env(top_k=8)

    scenarios = [
        ("factual", "What is shown in the retrieved clips?"),
        ("temporal", "What happens before and after the key event in these clips?"),
        ("ambiguous", "What might be happening if interpretation is uncertain?"),
        ("conflicting", "Do the clips agree, or is there conflicting evidence?"),
    ]

    for scenario_name, query in scenarios:
        output = generator.generate(query, evidence)
        print(f"\\n[{scenario_name}] {json.dumps(output, indent=2)}")
        _assert_output_shape(output, evidence_clip_ids)

    # Mandatory no-result fallback case
    fallback_output = generator.generate("Any answer?", [])
    print(f"\\n[no-result] {json.dumps(fallback_output, indent=2)}")
    assert fallback_output == {
        "answer": "Insufficient evidence to answer the query.",
        "citations": [],
        "reasoning": "No reliable supporting clips found.",
        "confidence": 0.0,
    }, fallback_output


def run_tests() -> bool:
    print("\\n" + "=" * 72)
    print("ANSWER GENERATOR - REAL AZURE INTEGRATION TEST")
    print("=" * 72)

    try:
        test_real_answer_generator_live_api()
        print("PASS: real Azure answer generation")
        return True
    except Exception as exc:
        print(f"FAIL: {exc}")
        return False


if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
