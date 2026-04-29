"""
E2E Pipeline Verification — HARDENED
Strict success criteria, real execution, full observability.
"""
import os, sys, json, time, re, logging, statistics
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from video_rag_query.graph_api import GraphAPI
from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.traversal import TraversalExecutor, TraversalConfig, TraversalState, TraversalResult
from video_rag_query.models import QueryDecomposition, FailureResponse
from video_rag_query.answer_generator import AnswerGenerator

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_verify")
logger.setLevel(logging.INFO)

QNA_PATH = os.path.join(PROJECT_ROOT, "QNA.json")
SELECTED_INDICES = [0, 4, 6, 10, 18, 22, 25, 28]

# ── STRICT thresholds ────────────────────────────────────────────────────────
DECOMP_LATENCY_LIMIT = 5.0
TRAVERSAL_LATENCY_LIMIT = 5.0
GEN_LATENCY_LIMIT = 15.0
TOTAL_LATENCY_LIMIT = 20.0
MAX_RETRIES_OK = 2
MIN_RESULTS = 5
QUERY_GAP_SECONDS = 5  # minimal — no artificial inflation

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


def load_selected_queries(path: str) -> List[Dict]:
    with open(path) as f:
        all_q = json.load(f)
    return [all_q[i] for i in SELECTED_INDICES if i < len(all_q)]


def extract_kw(query: str) -> List[str]:
    text = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    return [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]


def kw_fallback(query, api, top_k=15):
    kws = extract_kw(query)
    if not kws:
        return []
    raw = api.keyword_fallback_search(kws, top_k=top_k)
    if not raw:
        return []
    mx = len(kws[:5]) * 1.2
    return [TraversalResult(
        clip_id=r["clip_id"], score=round(min(1.0, r["score"] / mx) * 0.6, 4) if mx > 0 else 0.0,
        path=[], entities=[], explanation=f"Keyword fallback: {kws}", best_clip_id=r["clip_id"],
    ) for r in raw]


def merge(trav, fb):
    seen, merged = set(), []
    for r in trav:
        cid = getattr(r, "best_clip_id", None) or r.clip_id
        if cid not in seen:
            seen.add(cid); merged.append(r)
    for r in fb:
        cid = getattr(r, "best_clip_id", None) or r.clip_id
        if cid not in seen:
            seen.add(cid); merged.append(r)
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


def build_answer_payload(query, results, api):
    payload = {"query": query, "results": []}
    if not results:
        return payload
    ranked = sorted(results, key=lambda r: float(r.score), reverse=True)
    rows, seen = [], set()
    for res in ranked:
        cid = getattr(res, 'best_clip_id', None) or res.clip_id
        if cid in seen:
            continue
        props = api.get_node_properties([cid]).get(cid, {})
        if "video_id" not in props:
            continue
        seen.add(cid)
        rows.append({
            "clip_id": cid, "score": float(res.score),
            "summary": str(props.get("summary", "") or ""),
            "ocr_text": str(props.get("ocr", "") or ""),
            "transcript": str(props.get("transcript", "") or ""),
            "entities": list(getattr(res, "entities", []) or []),
            "timestamp": {"start": props.get("start"), "end": props.get("end"), "video_id": props.get("video_id")},
            "clip_path": str(props.get("clip_path", "") or ""),
        })
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    payload["results"] = rows
    return payload


def get_decomp_llm_usage(decomposition) -> Dict:
    logs = getattr(decomposition, "llm_logs", None)
    if not logs:
        return {"provider_attempts": [], "final_provider": "unknown", "final_model": "unknown",
                "retry_count": 0, "failure_reasons": [], "total_llm_time": 0.0}
    attempts = logs.get("provider_attempts", [])
    failures = [a.get("error") for a in attempts if a.get("status") != "success" and a.get("error")]
    return {
        "provider_attempts": attempts,
        "final_provider": logs.get("final_provider", "unknown"),
        "final_model": logs.get("final_model", "unknown"),
        "retry_count": max(0, len(attempts) - 1),
        "failure_reasons": failures,
        "total_llm_time": logs.get("total_llm_time", 0.0),
    }


def determine_status(qr: Dict) -> str:
    """STRICT: success ONLY if all criteria met, else degraded."""
    if qr.get("_failed"):
        return "failed"
    t = qr.get("timings", {})
    decomp_usage = qr.get("llm_usage", {}).get("decomposition", {})
    answer = qr.get("answer_data", {})
    citations = answer.get("citations", [])
    retrieved = set(qr.get("retrieved_clip_ids", []))

    # Citations must be valid subset
    if citations and not set(citations).issubset(retrieved):
        return "degraded"
    # No schema errors in decomposition
    if decomp_usage.get("retry_count", 0) > MAX_RETRIES_OK:
        return "degraded"
    # Latency thresholds
    if t.get("decomposition", 0) > DECOMP_LATENCY_LIMIT:
        return "degraded"
    if t.get("traversal", 0) > TRAVERSAL_LATENCY_LIMIT:
        return "degraded"
    if t.get("generation", 0) > GEN_LATENCY_LIMIT:
        return "degraded"
    if t.get("total", 0) > TOTAL_LATENCY_LIMIT:
        return "degraded"
    return "success"


def validate_query(qr: Dict) -> List[Dict]:
    """Phase 3 validation checks per query."""
    checks = []
    qid = qr.get("query", "?")[:60]
    t = qr.get("timings", {})
    answer = qr.get("answer_data", {})
    citations = answer.get("citations", [])
    retrieved = set(qr.get("retrieved_clip_ids", []))
    decomp = qr.get("llm_usage", {}).get("decomposition", {})

    # Retrieval checks
    rc = qr.get("retrieval_results_count", 0)
    if rc < MIN_RESULTS:
        checks.append({"query": qid, "check": "R1_LOW_RESULTS", "detail": f"results={rc} < {MIN_RESULTS}"})
    clip_ids = qr.get("retrieved_clip_ids", [])
    if len(clip_ids) != len(set(clip_ids)):
        checks.append({"query": qid, "check": "R2_DUPLICATE_CLIPS", "detail": "duplicate clip_ids in retrieval"})

    # Decomposition checks
    if decomp.get("retry_count", 0) > MAX_RETRIES_OK:
        checks.append({"query": qid, "check": "D1_EXCESSIVE_RETRIES", "detail": f"retries={decomp['retry_count']}"})

    # Generation checks
    if citations and not set(citations).issubset(retrieved):
        hallucinated = set(citations) - retrieved
        checks.append({"query": qid, "check": "G1_HALLUCINATED_CITATIONS", "detail": f"{hallucinated}"})

    # Latency checks
    if t.get("decomposition", 0) > DECOMP_LATENCY_LIMIT:
        checks.append({"query": qid, "check": "L1_DECOMP_SLOW", "detail": f"{t['decomposition']:.2f}s > {DECOMP_LATENCY_LIMIT}s"})
    if t.get("traversal", 0) > TRAVERSAL_LATENCY_LIMIT:
        checks.append({"query": qid, "check": "L2_TRAVERSAL_SLOW", "detail": f"{t['traversal']:.2f}s > {TRAVERSAL_LATENCY_LIMIT}s"})
    if t.get("generation", 0) > GEN_LATENCY_LIMIT:
        checks.append({"query": qid, "check": "L3_GEN_SLOW", "detail": f"{t['generation']:.2f}s > {GEN_LATENCY_LIMIT}s"})
    if t.get("total", 0) > TOTAL_LATENCY_LIMIT:
        checks.append({"query": qid, "check": "L4_TOTAL_SLOW", "detail": f"{t['total']:.2f}s > {TOTAL_LATENCY_LIMIT}s"})

    return checks


def main():
    CEREBRAS_KEYS = [k.strip() for k in os.getenv("CEREBRAS_API_KEYS", "").split(",") if k.strip()]
    GROQ_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]

    if not CEREBRAS_KEYS and not GROQ_KEYS:
        print("❌ No API keys found."); sys.exit(1)

    questions = load_selected_queries(QNA_PATH)
    print(f"\n{'='*80}")
    print(f"  E2E PIPELINE VERIFICATION — HARDENED")
    print(f"  Queries: {len(questions)} | No artificial delays")
    print(f"{'='*80}\n")

    mapping_db = os.path.join(PROJECT_ROOT, "outputs", "mapping.db")
    per_query: List[Dict] = []
    all_timings: List[Dict] = []
    errors: List[Dict] = []
    all_checks: List[Dict] = []

    with GraphAPI(mapping_db_path=mapping_db) as api:
        print("⏳ Fetching entity corpus...")
        t0 = time.perf_counter()
        records = api._entity.execute_query("MATCH (e:Entity) RETURN e.id AS id, e.name AS name")
        entity_corpus = [{"id": r["id"], "name": r["name"]} for r in records]
        print(f"✅ {len(entity_corpus)} entities in {time.perf_counter()-t0:.2f}s")

        print("⏳ Initialising decomposer...")
        t0 = time.perf_counter()
        decomposer = QueryDecomposer(cerebras_keys=CEREBRAS_KEYS, groq_keys=GROQ_KEYS, entity_corpus=entity_corpus)
        print(f"✅ Decomposer ready in {time.perf_counter()-t0:.2f}s")

        answer_gen = AnswerGenerator()
        print(f"✅ AnswerGenerator ready (models: {answer_gen.model_priority})\n")

        for idx, q_obj in enumerate(questions, 1):
            query = q_obj["question"]
            q_id = q_obj.get("id", f"Q{idx}")
            q_level = q_obj.get("level", "unknown")

            if idx > 1:
                time.sleep(QUERY_GAP_SECONDS)

            print(f"\n{'─'*80}")
            print(f"  [{idx}/{len(questions)}] {q_id} ({q_level}) → {query}")
            print(f"{'─'*80}")

            api.clear_cache()
            qr: Dict[str, Any] = {
                "query": query, "query_id": q_id, "level": q_level,
                "status": "pending", "timings": {}, "retrieval_results_count": 0,
                "answer_data": {}, "retrieved_clip_ids": [],
                "llm_usage": {"decomposition": {}, "generation": {}},
            }

            try:
                # T0→T1: Decomposition
                t0 = time.perf_counter()
                decomposition = decomposer.decompose(query)
                t1 = time.perf_counter()

                if isinstance(decomposition, FailureResponse):
                    raise RuntimeError(f"Decomposition failed: {decomposition.reason}")

                decomp_time = t1 - t0
                decomp_usage = get_decomp_llm_usage(decomposition)
                qr["llm_usage"]["decomposition"] = decomp_usage

                print(f"  ✓ Decomposition: {decomp_time:.3f}s | "
                      f"entities={len(decomposition.entities)}, "
                      f"confidence={decomposition.confidence:.2f}, "
                      f"provider={decomp_usage['final_provider']}/{decomp_usage['final_model']}, "
                      f"retries={decomp_usage['retry_count']}")

                # T1→T2: Traversal (use the hardened executor directly)
                use_deep = decomposition.confidence >= 0.5
                cfg = TraversalConfig(beam_width=15) if use_deep else TraversalConfig(beam_width=10, max_depth=3)
                executor = TraversalExecutor(api, cfg)

                t2_start = time.perf_counter()
                candidates = executor.execute(decomposition, original_query=query)
                t2 = time.perf_counter()
                traversal_time = t2 - t2_start

                print(f"  ✓ Traversal:     {traversal_time:.3f}s | "
                      f"candidates={len(candidates)}"
                      f"{' [DEEP]' if use_deep else ' [SHALLOW]'}")

                # Fallback
                fallback_time = 0.0
                if not candidates or len(candidates) < 3:
                    tf_start = time.perf_counter()
                    fb = kw_fallback(query, api)
                    candidates = merge(candidates or [], fb)
                    fallback_time = time.perf_counter() - tf_start
                    print(f"  ✓ Fallback:      {fallback_time:.3f}s | added={len(fb)}")

                # T2→T3: Ranking (already done inside executor.execute)
                ranking_time = 0.0
                print(f"  ✓ Ranking:       {ranking_time:.3f}s | results={len(candidates)}")

                qr["retrieval_results_count"] = len(candidates)
                qr["retrieved_clip_ids"] = list(dict.fromkeys(
                    (getattr(r, "best_clip_id", None) or r.clip_id) for r in candidates[:20]
                ))

                # T3→T4: Answer Generation
                t4_start = time.perf_counter()
                payload = build_answer_payload(query, candidates, api)
                answer = answer_gen.generate(payload)
                t4 = time.perf_counter()
                gen_time = t4 - t4_start

                # Extract Gemini observability from answer
                gen_logs = answer.pop("_gen_logs", {})
                latency_violation = answer.pop("_latency_violation", False)
                qr["answer_data"] = answer
                qr["llm_usage"]["generation"] = {
                    "provider_attempts": gen_logs.get("provider_attempts", []),
                    "final_provider": "gemini",
                    "final_model": (gen_logs.get("provider_attempts", [{}])[-1].get("model", "unknown")
                                    if gen_logs.get("provider_attempts") else "unknown"),
                    "retry_count": gen_logs.get("retry_count", 0),
                    "media_used": gen_logs.get("media_used", False),
                    "token_estimate": gen_logs.get("token_estimate", 0),
                    "latency_violation": latency_violation,
                }

                total = decomp_time + traversal_time + fallback_time + gen_time
                qr["timings"] = {
                    "decomposition": round(decomp_time, 4),
                    "traversal": round(traversal_time, 4),
                    "ranking": round(ranking_time, 4),
                    "fallback": round(fallback_time, 4),
                    "generation": round(gen_time, 4),
                    "total": round(total, 4),
                }
                all_timings.append(qr["timings"])

                # Determine STRICT status
                qr["status"] = determine_status(qr)

                conf = answer.get("confidence", 0)
                ans_preview = (answer.get("answer", "") or "")[:120]
                cit_count = len(answer.get("citations", []))

                status_icon = "✅" if qr["status"] == "success" else "⚠️"
                print(f"  ✓ Generation:    {gen_time:.3f}s | "
                      f"confidence={conf:.2f}, citations={cit_count}, "
                      f"media={gen_logs.get('media_used', False)}")
                print(f"  {status_icon} TOTAL: {total:.3f}s | status={qr['status']}")
                print(f"\n  Answer: {ans_preview}...")

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                qr["status"] = "failed"
                qr["_failed"] = True
                qr["error"] = err
                errors.append({"query_id": q_id, "query": query, "error": err})
                print(f"  ❌ ERROR: {err}")
                logger.exception(f"Query {q_id} failed")

            # Run validation checks
            checks = validate_query(qr)
            all_checks.extend(checks)
            per_query.append(qr)

    # ═══ Aggregated Report ═══
    print(f"\n\n{'='*80}")
    print(f"  AGGREGATED REPORT")
    print(f"{'='*80}")

    succeeded = [q for q in per_query if q["status"] == "success"]
    degraded = [q for q in per_query if q["status"] == "degraded"]
    failed = [q for q in per_query if q["status"] == "failed"]

    decomp_providers = [q["llm_usage"]["decomposition"].get("final_provider", "?") for q in per_query if q["status"] != "failed"]
    total_q = max(1, len(decomp_providers))

    summary = {"num_queries": len(questions)}
    summary["success_count"] = len(succeeded)
    summary["degraded_count"] = len(degraded)
    summary["failed_count"] = len(failed)

    if all_timings:
        def avg(key): return round(statistics.mean([t[key] for t in all_timings]), 4)
        def mx(key): return round(max([t[key] for t in all_timings]), 4)
        def mn(key): return round(min([t[key] for t in all_timings]), 4)

        summary["avg_latency"] = {s: avg(s) for s in ["decomposition","traversal","generation","total"]}
        summary["max_latency"] = {s: mx(s) for s in ["decomposition","traversal","generation","total"]}
        summary["provider_usage"] = {
            "cerebras_pct": round(sum(1 for p in decomp_providers if p == "cerebras") / total_q * 100, 1),
            "groq_pct": round(sum(1 for p in decomp_providers if p == "groq") / total_q * 100, 1),
        }
        # Gemini latency distribution
        gen_latencies = [q["llm_usage"]["generation"].get("provider_attempts", [{}])[-1].get("latency", 0)
                         for q in per_query if q["status"] != "failed" and q["llm_usage"]["generation"].get("provider_attempts")]
        if gen_latencies:
            summary["gemini_latency"] = {
                "avg": round(statistics.mean(gen_latencies), 2),
                "min": round(min(gen_latencies), 2),
                "max": round(max(gen_latencies), 2),
            }

    summary["validation_failures"] = all_checks
    print(json.dumps(summary, indent=2))

    # Validation summary
    print(f"\n  STATUS: {len(succeeded)} success | {len(degraded)} degraded | {len(failed)} failed")
    print(f"  VALIDATIONS: {len(all_checks)} issues")
    for c in all_checks:
        print(f"    ⚠ [{c['check']}] {c['query'][:50]}: {c['detail']}")

    if all_timings:
        print(f"\n  {'Stage':<20} {'Avg':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'─'*50}")
        for stage in ["decomposition", "traversal", "generation", "total"]:
            vals = [t[stage] for t in all_timings]
            print(f"  {stage:<20} {statistics.mean(vals):>7.3f}s {min(vals):>7.3f}s {max(vals):>7.3f}s")

    output = {
        "metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "num_queries": len(questions), "version": "e2e_hardened_v2"},
        "per_query": per_query,
        "summary": summary,
        "validation_failures": all_checks,
        "errors": errors,
    }
    out_path = os.path.join(PROJECT_ROOT, "e2e_verification_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  📄 Full results: {out_path}")


if __name__ == "__main__":
    main()
