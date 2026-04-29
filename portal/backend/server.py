"""
VideoGraphRAG Portal — FastAPI Backend
Connects directly to the LIVE pipeline: decomposition → traversal → ranking → generation.
NO mock data. NO placeholder responses.
"""
import os
import sys
import time
import json
import logging
import mimetypes
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Resolve project root ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from video_rag_query.graph_api import GraphAPI
from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.traversal import (
    TraversalExecutor, TraversalConfig, TraversalState, TraversalResult,
)
from video_rag_query.models import QueryDecomposition, FailureResponse
from video_rag_query.answer_generator import AnswerGenerator

import re

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("portal.backend")

# ── Stop words for keyword fallback ─────────────────────────────────────────
STOP_WORDS = {
    "a","an","the","and","or","but","if","then","else","when","at","from","by",
    "for","with","about","against","between","into","through","during","before",
    "after","above","below","to","up","down","in","out","on","off","over","under",
    "again","further","once","here","there","where","why","how","all","any","both",
    "each","few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","s","t","can","will","just","don","should",
    "now","is","was","were","be","been","being","have","has","had","having","do",
    "does","did","doing","what","which","who","whom","it","its","this","that",
    "these","those","are","would","could","may","might","shall","of",
}


def extract_kw(query: str) -> List[str]:
    text = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    return [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]


def kw_fallback(query: str, api: GraphAPI, top_k: int = 15) -> List[TraversalResult]:
    kws = extract_kw(query)
    if not kws:
        return []
    raw = api.keyword_fallback_search(kws, top_k=top_k)
    if not raw:
        return []
    mx = len(kws[:5]) * 1.2
    return [
        TraversalResult(
            clip_id=r["clip_id"],
            score=round(min(1.0, r["score"] / mx) * 0.6, 4) if mx > 0 else 0.0,
            path=[], entities=[],
            explanation=f"Keyword fallback: {kws}",
            best_clip_id=r["clip_id"],
        )
        for r in raw
    ]


def merge_results(trav, fb):
    seen, merged = set(), []
    for r in trav:
        cid = getattr(r, "best_clip_id", None) or r.clip_id
        if cid not in seen:
            seen.add(cid)
            merged.append(r)
    for r in fb:
        cid = getattr(r, "best_clip_id", None) or r.clip_id
        if cid not in seen:
            seen.add(cid)
            merged.append(r)
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


def build_answer_payload(query: str, results: List[TraversalResult], api: GraphAPI) -> Dict:
    payload: Dict[str, Any] = {"query": query, "results": []}
    if not results:
        return payload
    ranked = sorted(results, key=lambda r: float(r.score), reverse=True)
    rows, seen = [], set()
    for res in ranked:
        cid = getattr(res, "best_clip_id", None) or res.clip_id
        if cid in seen:
            continue
        props = api.get_node_properties([cid]).get(cid, {})
        if "video_id" not in props:
            continue
        seen.add(cid)
        rows.append({
            "clip_id": cid,
            "score": float(res.score),
            "summary": str(props.get("summary", "") or ""),
            "ocr_text": str(props.get("ocr", "") or ""),
            "transcript": str(props.get("transcript", "") or ""),
            "entities": list(getattr(res, "entities", []) or []),
            "timestamp": {
                "start": props.get("start"),
                "end": props.get("end"),
                "video_id": props.get("video_id"),
            },
            "clip_path": str(props.get("clip_path", "") or ""),
        })
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    payload["results"] = rows
    return payload


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="VideoGraphRAG Portal API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ───────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


# ── Global pipeline state ──────────────────────────────────────────────────
_pipeline = {
    "api": None,
    "decomposer": None,
    "answer_gen": None,
    "entity_corpus": None,
    "ready": False,
}


def _init_pipeline():
    """Initialize the pipeline once on first request (lazy init)."""
    if _pipeline["ready"]:
        return

    logger.info("Initializing pipeline...")
    t0 = time.perf_counter()

    CEREBRAS_KEYS = [k.strip() for k in os.getenv("CEREBRAS_API_KEYS", "").split(",") if k.strip()]
    GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]

    if not CEREBRAS_KEYS and not GROQ_KEYS:
        raise RuntimeError("No LLM API keys found in .env")

    mapping_db = os.path.join(PROJECT_ROOT, "outputs", "mapping.db")
    api = GraphAPI(mapping_db_path=mapping_db)
    api.connect()

    records = api._entity.execute_query("MATCH (e:Entity) RETURN e.id AS id, e.name AS name")
    entity_corpus = [{"id": r["id"], "name": r["name"]} for r in records]
    logger.info(f"Loaded {len(entity_corpus)} entities")

    decomposer = QueryDecomposer(
        cerebras_keys=CEREBRAS_KEYS,
        groq_keys=GROQ_KEYS,
        entity_corpus=entity_corpus,
    )

    answer_gen = AnswerGenerator()

    _pipeline["api"] = api
    _pipeline["decomposer"] = decomposer
    _pipeline["answer_gen"] = answer_gen
    _pipeline["entity_corpus"] = entity_corpus
    _pipeline["ready"] = True

    logger.info(f"Pipeline ready in {time.perf_counter() - t0:.2f}s")


def _run_retrieval(query: str, top_k: int = 10) -> Dict[str, Any]:
    """Run decomposition → traversal → ranking. Returns timings + results."""
    api: GraphAPI = _pipeline["api"]
    decomposer: QueryDecomposer = _pipeline["decomposer"]
    api.clear_cache()

    timings: Dict[str, float] = {}

    # Decomposition
    t0 = time.perf_counter()
    decomposition = decomposer.decompose(query)
    t1 = time.perf_counter()
    timings["decomposition"] = round((t1 - t0) * 1000, 2)

    if isinstance(decomposition, FailureResponse):
        raise HTTPException(
            status_code=422,
            detail={
                "stage": "decomposition",
                "error": f"Decomposition failed: {decomposition.reason}",
                "timings": timings,
            },
        )

    # Extract decomp metadata
    llm_logs = getattr(decomposition, "llm_logs", None) or {}
    provider_used = llm_logs.get("final_provider", "unknown")
    model_used = llm_logs.get("final_model", "unknown")

    # Traversal
    use_deep = decomposition.confidence >= 0.5
    cfg = TraversalConfig(beam_width=15) if use_deep else TraversalConfig(beam_width=10, max_depth=3)
    executor = TraversalExecutor(api, cfg)

    t2 = time.perf_counter()
    candidates = executor.execute(decomposition, original_query=query)
    t3 = time.perf_counter()
    timings["traversal"] = round((t3 - t2) * 1000, 2)

    # Keyword fallback
    if not candidates or len(candidates) < 3:
        fb = kw_fallback(query, api, top_k=top_k)
        candidates = merge_results(candidates or [], fb)

    # Ranking time is included in traversal (rerank is part of execute)
    timings["ranking"] = 0.0

    # Build enriched clip data
    ranked = sorted(candidates, key=lambda r: r.score, reverse=True)[:top_k]
    results = []
    seen = set()
    for res in ranked:
        cid = getattr(res, "best_clip_id", None) or res.clip_id
        if cid in seen:
            continue
        props = api.get_node_properties([cid]).get(cid, {})
        if "video_id" not in props:
            continue
        seen.add(cid)
        results.append({
            "clip_id": cid,
            "score": float(res.score),
            "summary": str(props.get("summary", "") or ""),
            "ocr_text": str(props.get("ocr", "") or ""),
            "transcript": str(props.get("transcript", "") or ""),
            "entities": list(getattr(res, "entities", []) or []),
            "timestamp": {
                "start": props.get("start"),
                "end": props.get("end"),
                "video_id": props.get("video_id"),
            },
            "clip_path": str(props.get("clip_path", "") or ""),
        })

    return {
        "timings": timings,
        "results": results,
        "decomposition": decomposition,
        "candidates": candidates,
        "provider_used": provider_used,
        "model_used": model_used,
    }


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": _pipeline["ready"]}


@app.post("/query/retrieve")
async def query_retrieve(req: QueryRequest):
    """Retrieval-only mode: decomposition → traversal → ranking."""
    _init_pipeline()

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    logger.info(f"[retrieve] query={query!r} top_k={req.top_k}")
    t_start = time.perf_counter()

    retrieval = _run_retrieval(query, top_k=req.top_k)

    t_end = time.perf_counter()
    retrieval["timings"]["total"] = round((t_end - t_start) * 1000, 2)

    logger.info(
        f"[retrieve] done: {len(retrieval['results'])} results, "
        f"total={retrieval['timings']['total']:.0f}ms"
    )

    return {
        "query": query,
        "timings": retrieval["timings"],
        "results": retrieval["results"],
        "debug": {
            "provider_used": retrieval["provider_used"],
            "model_used": retrieval["model_used"],
            "num_results": len(retrieval["results"]),
        },
    }


@app.post("/query/answer")
async def query_answer(req: QueryRequest):
    """Full pipeline: decomposition → traversal → ranking → generation."""
    _init_pipeline()

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    logger.info(f"[answer] query={query!r} top_k={req.top_k}")
    t_start = time.perf_counter()

    retrieval = _run_retrieval(query, top_k=req.top_k)

    # Build answer payload and generate
    answer_gen: AnswerGenerator = _pipeline["answer_gen"]
    api: GraphAPI = _pipeline["api"]

    t_gen_start = time.perf_counter()
    answer_payload = build_answer_payload(query, retrieval["candidates"], api)
    answer = answer_gen.generate(answer_payload)
    t_gen_end = time.perf_counter()

    gen_time = round((t_gen_end - t_gen_start) * 1000, 2)
    retrieval["timings"]["generation"] = gen_time

    t_end = time.perf_counter()
    retrieval["timings"]["total"] = round((t_end - t_start) * 1000, 2)

    # Extract gen logs
    gen_logs = answer.pop("_gen_logs", {})
    answer.pop("_latency_violation", None)

    gen_provider_attempts = gen_logs.get("provider_attempts", [])
    gen_model = (
        gen_provider_attempts[-1].get("model", "unknown")
        if gen_provider_attempts
        else "unknown"
    )

    logger.info(
        f"[answer] done: confidence={answer.get('confidence', 0):.2f}, "
        f"citations={len(answer.get('citations', []))}, "
        f"total={retrieval['timings']['total']:.0f}ms"
    )

    return {
        "query": query,
        "answer": answer,
        "timings": retrieval["timings"],
        "results": retrieval["results"],
        "debug": {
            "provider_used": retrieval["provider_used"],
            "model_used": retrieval["model_used"],
            "gen_model_used": gen_model,
            "num_results": len(retrieval["results"]),
            "media_used": gen_logs.get("media_used", False),
            "gen_retries": gen_logs.get("retry_count", 0),
        },
    }


@app.get("/media")
async def serve_media(clip_path: str):
    """Serve a clip file from disk for the video player."""
    if not clip_path:
        raise HTTPException(status_code=400, detail="Missing clip_path parameter")

    # Resolve relative paths against project root
    if not os.path.isabs(clip_path):
        clip_path = os.path.join(PROJECT_ROOT, clip_path)

    if not os.path.isfile(clip_path):
        raise HTTPException(status_code=404, detail=f"Clip file not found: {clip_path}")

    mime_type = mimetypes.guess_type(clip_path)[0] or "application/octet-stream"
    return FileResponse(clip_path, media_type=mime_type)


# ── Startup / shutdown ──────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("Portal backend starting...")


@app.on_event("shutdown")
async def shutdown():
    if _pipeline["api"]:
        _pipeline["api"].close()
        logger.info("Pipeline connections closed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
