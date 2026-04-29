# Deprecated – not used in final pipeline
import json
import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from video_rag_query.answer_generator import AnswerGenerationInput, AnswerGenerator
from video_rag_query.graph_api import GraphAPI
from video_rag_query.key_manager import GeminiKeyManager
from video_rag_query.models import FailureResponse
from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.run_query import _build_answer_payload, fetch_entity_corpus
from video_rag_query.traversal import TraversalConfig, TraversalExecutor

load_dotenv()

LIVE_QUERY = os.getenv("ANSWER_GENERATION_TEST_QUERY", "Which Airport was built on Swamp?")


def _require_live_env():
    assert os.path.exists(".env"), ".env is required for live Gemini tests."

    gemini_key_manager = GeminiKeyManager.from_env()
    assert gemini_key_manager.key_count > 0, "Gemini API keys are required for live Gemini tests."

    assert os.getenv("NEO4J_CLIP_URI"), "NEO4J_CLIP_URI is required."
    assert os.getenv("NEO4J_ENTITY_URI"), "NEO4J_ENTITY_URI is required."
    assert os.path.exists("outputs/mapping.db"), "outputs/mapping.db is required."

    cerebras_keys = [k.strip() for k in os.getenv("CEREBRAS_API_KEYS", "").split(",") if k.strip()]
    groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
    assert cerebras_keys or groq_keys, "CEREBRAS_API_KEYS or GROQ_API_KEYS are required for live ranked retrieval."

    return cerebras_keys, groq_keys


def _build_live_payload(query: str):
    cerebras_keys, groq_keys = _require_live_env()
    with GraphAPI(mapping_db_path="outputs/mapping.db") as api:
        entity_corpus = fetch_entity_corpus(api)
        decomposer = QueryDecomposer(
            cerebras_keys=cerebras_keys,
            groq_keys=groq_keys,
            entity_corpus=entity_corpus,
        )
        decomposition = decomposer.decompose(query)
        assert not isinstance(decomposition, FailureResponse), f"Live decomposition failed: {decomposition}"
        executor = TraversalExecutor(api, TraversalConfig(beam_width=15))
        results = executor.execute(decomposition, query)
        payload = _build_answer_payload(query, results, api)

    assert payload["results"], "Live traversal returned no clip results for answer generation."
    return payload


def _validate_answer(answer, allowed_clip_ids):
    assert set(answer.keys()) == {"answer", "citations", "reasoning", "confidence"}
    assert isinstance(answer["answer"], str)
    assert isinstance(answer["citations"], list)
    assert isinstance(answer["reasoning"], str)
    assert isinstance(answer["confidence"], float)
    assert set(answer["citations"]).issubset(allowed_clip_ids)


@pytest.mark.live_gemini
def test_live_gemini_answer_generation_prints_structured_output():
    payload = _build_live_payload(LIVE_QUERY)
    generator = AnswerGenerator()
    answer = generator.generate(payload)
    print(json.dumps(answer, indent=2, ensure_ascii=False))
    _validate_answer(answer, {row["clip_id"] for row in payload["results"]})


@pytest.mark.live_gemini
def test_live_gemini_prompt_preserves_rank_and_weak_middle_layout():
    payload = _build_live_payload(LIVE_QUERY)
    generator = AnswerGenerator()
    payload_obj = AnswerGenerationInput.from_dict(payload)
    normalized = generator._prepare_context(payload_obj.results)
    high = generator._presentation_order([clip for clip in normalized if clip.tier == "high"])
    mid = generator._presentation_order([clip for clip in normalized if clip.tier == "mid"])

    if len(high) >= 5:
        assert [clip.rank for clip in high[:5]] == [1, 2, 5, 4, 3]
    if len(mid) >= 5:
        assert [clip.rank for clip in mid[:5]] == [6, 7, 10, 9, 8]

    answer = generator.generate(payload)
    print(json.dumps(answer, indent=2, ensure_ascii=False))
    _validate_answer(answer, {row["clip_id"] for row in payload["results"]})


@pytest.mark.live_gemini
def test_live_gemini_missing_clip_path_degrades_to_text_only():
    payload = _build_live_payload(LIVE_QUERY)
    payload["results"][0]["clip_path"] = "/definitely/missing/clip.mp4"
    generator = AnswerGenerator()
    answer = generator.generate(payload)
    print(json.dumps(answer, indent=2, ensure_ascii=False))
    _validate_answer(answer, {row["clip_id"] for row in payload["results"]})


@pytest.mark.live_gemini
def test_live_gemini_multimodal_attachment_when_local_clips_exist():
    payload = _build_live_payload(LIVE_QUERY)
    if not any(os.path.isfile(str(row.get("clip_path", ""))) for row in payload["results"][:5]):
        pytest.skip("No readable Tier 1 clip_path found locally for live multimodal attachment.")

    generator = AnswerGenerator()
    answer = generator.generate(payload)
    print(json.dumps(answer, indent=2, ensure_ascii=False))
    _validate_answer(answer, {row["clip_id"] for row in payload["results"]})
