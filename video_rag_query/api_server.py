from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urljoin

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEDIA_ROOT = (PROJECT_ROOT / "outputs").resolve()
LEGACY_MEDIA_ROOT = (PROJECT_ROOT / "video_rag_preprocessing" / "outputs").resolve()
DEFAULT_MAPPING_DB = DEFAULT_MEDIA_ROOT / "mapping.db"

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=20)


app = FastAPI(title="VideoGraphRAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_media_root() -> Path:
    return DEFAULT_MEDIA_ROOT


def resolve_media_roots() -> List[Path]:
    return [DEFAULT_MEDIA_ROOT, LEGACY_MEDIA_ROOT]


def resolve_mapping_db_path() -> Path:
    return DEFAULT_MAPPING_DB


def _normalise_base_url(base_url: str) -> str:
    return base_url if base_url.endswith("/") else f"{base_url}/"


def _coerce_media_roots(media_root: Optional[Path] = None) -> List[Path]:
    if media_root is not None:
        return [Path(media_root).resolve()]
    return resolve_media_roots()


def _safe_media_file(clip_path: str, media_root: Optional[Path] = None) -> Optional[Path]:
    if not clip_path:
        return None

    roots = _coerce_media_roots(media_root)
    try:
        candidate = Path(clip_path).resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return None

    if not candidate.is_file():
        return None

    for root in roots:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue

    return None


def _safe_media_file_from_relative_path(relative_path: str) -> Optional[Path]:
    if not relative_path:
        return None

    for root in resolve_media_roots():
        try:
            candidate = (root / relative_path).resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            continue

        try:
            candidate.relative_to(root)
        except ValueError:
            continue

        if candidate.is_file():
            return candidate

    return None


def build_clip_url(
    clip_path: str,
    media_root: Optional[Path] = None,
    base_url: Optional[str] = None,
    request: Optional[Request] = None,
) -> Optional[str]:
    media_file = _safe_media_file(clip_path, media_root=media_root)
    if media_file is None:
        return None

    query_string = urlencode({"clip_path": str(media_file)})

    if request is not None:
        return f"{request.url_for('serve_media_by_path')}?{query_string}"

    if base_url is None:
        return f"/media?{query_string}"

    return urljoin(_normalise_base_url(base_url), f"media?{query_string}")


def _get_query_backend():
    from video_rag_query.answer_generator import AnswerGenerator, DEFAULT_FALLBACK_RESPONSE
    from video_rag_query.graph_api import GraphAPI
    from video_rag_query.models import FailureResponse
    from video_rag_query.query_decomposer import QueryDecomposer
    from video_rag_query.traversal import TraversalConfig, TraversalExecutor, TraversalState

    return {
        "AnswerGenerator": AnswerGenerator,
        "DEFAULT_FALLBACK_RESPONSE": DEFAULT_FALLBACK_RESPONSE,
        "FailureResponse": FailureResponse,
        "GraphAPI": GraphAPI,
        "QueryDecomposer": QueryDecomposer,
        "TraversalConfig": TraversalConfig,
        "TraversalExecutor": TraversalExecutor,
        "TraversalState": TraversalState,
    }


def _split_env_keys(env_name: str) -> List[str]:
    return [part.strip() for part in os.getenv(env_name, "").split(",") if part.strip()]


def _fetch_entity_corpus(api: Any) -> List[Dict[str, str]]:
    records = api._entity.execute_query("MATCH (e:Entity) RETURN e.id AS id, e.name AS name")
    return [{"id": str(row["id"]), "name": str(row["name"])} for row in records if row.get("id") and row.get("name")]


def _resolve_entity_labels(api: Any, entity_ids: List[str]) -> Dict[str, str]:
    if not entity_ids:
        return {}

    props = api.get_node_properties(entity_ids)
    labels: Dict[str, str] = {}
    for entity_id in entity_ids:
        entity_props = props.get(entity_id, {})
        labels[entity_id] = str(entity_props.get("name") or entity_id)
    return labels


def _normalise_results(
    results: List[Any],
    api: Any,
    top_k: int,
    request: Optional[Request] = None,
) -> List[Dict[str, Any]]:
    ranked = sorted(results, key=lambda item: float(item.score), reverse=True)
    playable_clip_ids: List[str] = []
    entity_ids: List[str] = []

    for result in ranked[:top_k]:
        clip_id = getattr(result, "best_clip_id", None) or result.clip_id
        playable_clip_ids.append(clip_id)
        entity_ids.extend([entity_id for entity_id in list(getattr(result, "entities", []) or []) if entity_id])

    clip_props_map = api.get_node_properties(playable_clip_ids)
    entity_labels = _resolve_entity_labels(api, entity_ids)

    normalized: List[Dict[str, Any]] = []
    seen_clip_ids = set()
    for rank, result in enumerate(ranked, start=1):
        if len(normalized) >= top_k:
            break

        clip_id = getattr(result, "best_clip_id", None) or result.clip_id
        if clip_id in seen_clip_ids:
            continue

        clip_props = clip_props_map.get(clip_id, {})
        clip_path = str(clip_props.get("clip_path", "") or "")
        clip_url = build_clip_url(clip_path, request=request)

        entity_values = []
        for entity_id in list(getattr(result, "entities", []) or []):
            entity_values.append(entity_labels.get(entity_id, str(entity_id)))

        normalized.append(
            {
                "clip_id": clip_id,
                "source_id": result.clip_id,
                "score": round(float(result.score), 4),
                "summary": str(clip_props.get("summary", "") or ""),
                "timestamp": {
                    "start": clip_props.get("start"),
                    "end": clip_props.get("end"),
                    "video_id": clip_props.get("video_id"),
                },
                "entities": entity_values,
                "clip_path": clip_path,
                "clip_url": clip_url,
                "transcript": str(clip_props.get("transcript", "") or ""),
                "ocr_text": str(clip_props.get("ocr", "") or ""),
                "rank": rank,
                "explanation": str(getattr(result, "explanation", "") or ""),
            }
        )
        seen_clip_ids.add(clip_id)

    return normalized


def _shape_public_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    public_results: List[Dict[str, Any]] = []
    for row in results:
        public_results.append(
            {
                "clip_id": row.get("clip_id"),
                "score": row.get("score"),
                "summary": row.get("summary"),
                "timestamp": row.get("timestamp"),
                "entities": row.get("entities", []),
                "clip_path": row.get("clip_path"),
            }
        )
    return public_results


def _shape_public_payload(payload: Dict[str, Any], answer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    public_payload: Dict[str, Any] = {
        "query": payload["query"],
        "timings": dict(payload["timings"]),
        "results": _shape_public_results(payload["results"]),
    }
    if answer is not None:
        public_payload["answer"] = answer
    return public_payload


def execute_retrieval_pipeline(
    query: str,
    top_k: int,
    request: Optional[Request] = None,
) -> Dict[str, Any]:
    backend = _get_query_backend()
    GraphAPI = backend["GraphAPI"]
    QueryDecomposer = backend["QueryDecomposer"]
    FailureResponse = backend["FailureResponse"]
    TraversalConfig = backend["TraversalConfig"]
    TraversalExecutor = backend["TraversalExecutor"]
    TraversalState = backend["TraversalState"]

    cerebras_keys = _split_env_keys("CEREBRAS_API_KEYS")
    groq_keys = _split_env_keys("GROQ_API_KEYS")
    if not cerebras_keys and not groq_keys:
        raise RuntimeError("CEREBRAS_API_KEYS or GROQ_API_KEYS must be configured for query decomposition.")

    mapping_db = resolve_mapping_db_path()
    if not mapping_db.exists():
        raise RuntimeError(f"Mapping database not found at {mapping_db}")

    total_start = time.perf_counter()
    with GraphAPI(mapping_db_path=str(mapping_db)) as api:
        entity_corpus = _fetch_entity_corpus(api)
        decomposer = QueryDecomposer(
            cerebras_keys=cerebras_keys,
            groq_keys=groq_keys,
            entity_corpus=entity_corpus,
        )

        decomposition_start = time.perf_counter()
        decomposition = decomposer.decompose(query)
        decomposition_ms = (time.perf_counter() - decomposition_start) * 1000.0

        if isinstance(decomposition, FailureResponse):
            raise RuntimeError(f"Query decomposition failed: {decomposition.reason}")

        executor = TraversalExecutor(
            api,
            TraversalConfig(
                beam_width=max(15, top_k * 2),
                rerank_top_n=max(50, top_k * 4),
            ),
        )

        traversal_start = time.perf_counter()
        typed_steps = decomposition.get_typed_execution_plan()
        states: List[Any] = [TraversalState()]
        for step in typed_steps:
            if not states:
                break
            states = executor._dispatch(step, states)
            states = executor._prune(states)
        candidates = executor._collect_candidates(states)
        traversal_ms = (time.perf_counter() - traversal_start) * 1000.0

        ranking_start = time.perf_counter()
        reranked = executor._rerank(candidates, decomposition, query)
        ranking_ms = (time.perf_counter() - ranking_start) * 1000.0

        results = _normalise_results(reranked, api=api, top_k=top_k, request=request)

    total_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "query": query,
        "timings": {
            "decomposition": round(decomposition_ms, 2),
            "traversal": round(traversal_ms, 2),
            "ranking": round(ranking_ms, 2),
            "total": round(total_ms, 2),
        },
        "results": results,
    }


def generate_answer_payload(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    backend = _get_query_backend()
    AnswerGenerator = backend["AnswerGenerator"]
    fallback_response = backend["DEFAULT_FALLBACK_RESPONSE"]

    generator = AnswerGenerator()
    answer = generator.generate({"query": query, "results": results})

    if results and answer == fallback_response:
        last_error = getattr(generator, "_last_error", None)
        no_keys = getattr(generator, "key_manager", None) is not None and generator.key_manager.key_count == 0
        if last_error or no_keys:
            raise RuntimeError(last_error or "No Gemini API keys available for answer generation.")

    return answer


@app.get("/health")
async def health() -> Dict[str, Any]:
    module_status = {}
    for module_name in ("fastapi", "uvicorn", "dotenv", "sentence_transformers", "numpy", "openai", "google.genai"):
        try:
            module_status[module_name] = importlib.util.find_spec(module_name) is not None
        except ModuleNotFoundError:
            module_status[module_name] = False

    return {
        "status": "ok",
        "mapping_db_exists": resolve_mapping_db_path().exists(),
        "media_root_exists": any(root.exists() for root in resolve_media_roots()),
        "configured": {
            "cerebras_keys": bool(_split_env_keys("CEREBRAS_API_KEYS")),
            "groq_keys": bool(_split_env_keys("GROQ_API_KEYS")),
            "gemini_keys": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEYS")),
        },
        "dependencies": module_status,
    }


@app.get("/media", name="serve_media_by_path")
async def serve_media_by_path(clip_path: str) -> FileResponse:
    media_file = _safe_media_file(clip_path)
    if media_file is None:
        raise HTTPException(status_code=404, detail={"stage": "media", "error": "Clip not found."})

    return FileResponse(media_file)


@app.get("/media/{relative_path:path}", name="serve_media")
async def serve_media(relative_path: str) -> FileResponse:
    media_file = _safe_media_file_from_relative_path(relative_path)
    if media_file is None:
        raise HTTPException(status_code=404, detail={"stage": "media", "error": "Clip not found."})

    return FileResponse(media_file)


@app.post("/query/retrieve")
async def query_retrieve(payload: QueryRequest, request: Request = None) -> Dict[str, Any]:
    retrieval_payload = execute_retrieval_pipeline(payload.query, payload.top_k, request=request)
    return _shape_public_payload(retrieval_payload)


@app.post("/query/answer")
async def query_answer(payload: QueryRequest, request: Request = None) -> Dict[str, Any]:
    retrieval_payload = execute_retrieval_pipeline(payload.query, payload.top_k, request=request)

    generation_start = time.perf_counter()
    try:
        answer = generate_answer_payload(payload.query, retrieval_payload["results"])
    except Exception as exc:
        generation_ms = round((time.perf_counter() - generation_start) * 1000.0, 2)
        timings = dict(retrieval_payload["timings"])
        timings["total"] = round(timings["total"] + generation_ms, 2)
        public_payload = _shape_public_payload({"query": retrieval_payload["query"], "timings": timings, "results": retrieval_payload["results"]})
        raise HTTPException(
            status_code=502,
            detail={
                "stage": "generation",
                "error": str(exc),
                "query": public_payload["query"],
                "timings": public_payload["timings"],
                "results": public_payload["results"],
            },
        ) from exc

    generation_ms = round((time.perf_counter() - generation_start) * 1000.0, 2)
    timings = dict(retrieval_payload["timings"])
    timings["total"] = round(timings["total"] + generation_ms, 2)

    return _shape_public_payload(
        {"query": retrieval_payload["query"], "timings": timings, "results": retrieval_payload["results"]},
        answer=answer,
    )
