"""
Live Retrieval Pipeline Evaluation (Post-Refactor)
===================================================
Runs real queries from QNA.json through the refactored retrieval pipeline:
    query → decomposition → traversal → ranking → (fallback if needed) → results

Measures per-stage latency including fallback, validates output quality,
and compares against pre-fix baseline.

NO answer generation. NO mocking. Caches cleared between queries.
"""

import os
import sys
import json
import time
import re
import logging
import statistics
from typing import List, Dict, Any, Optional

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from video_rag_query.graph_api import GraphAPI
from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.traversal import TraversalExecutor, TraversalConfig, TraversalState, TraversalResult
from video_rag_query.models import QueryDecomposition, FailureResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("live_eval")
logger.setLevel(logging.INFO)

# ── Constants ────────────────────────────────────────────────────────────────
QNA_PATH = os.path.join(PROJECT_ROOT, "QNA.json")
NUM_QUERIES = 10
LATENCY_WARN_THRESHOLD = 5.0  # seconds
MIN_CONFIDENCE_FOR_DEEP_TRAVERSAL = 0.5
FALLBACK_SCORE_MULTIPLIER = 0.6
QUERY_GAP_SECONDS = 60  # gap between queries to avoid rate limiting

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at",
    "from", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "once",
    "here", "there", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "is", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "what", "which", "who",
    "whom", "it", "its", "this", "that", "these", "those", "are", "would",
    "could", "may", "might", "shall", "of",
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_questions(path: str, n: int) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        all_q = json.load(f)
    return all_q[:n]


def extract_query_keywords(query: str) -> List[str]:
    text = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    words = text.split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def keyword_fallback(query: str, api: GraphAPI, top_k: int = 15) -> List[TraversalResult]:
    """Fallback: keyword search across clip text fields."""
    keywords = extract_query_keywords(query)
    if not keywords:
        return []

    raw_results = api.keyword_fallback_search(keywords, top_k=top_k)
    if not raw_results:
        return []

    max_possible = len(keywords[:5]) * 1.2
    results = []
    for r in raw_results:
        normalised = min(1.0, r["score"] / max_possible) if max_possible > 0 else 0.0
        final_score = normalised * FALLBACK_SCORE_MULTIPLIER
        results.append(TraversalResult(
            clip_id=r["clip_id"],
            score=round(final_score, 4),
            path=[],
            entities=[],
            explanation=f"Keyword fallback: {keywords}",
            best_clip_id=r["clip_id"],
        ))
    return results


def merge_results(
    traversal: List[TraversalResult], fallback: List[TraversalResult]
) -> List[TraversalResult]:
    seen = set()
    merged = []
    for r in traversal:
        cid = getattr(r, "best_clip_id", None) or r.clip_id
        if cid not in seen:
            seen.add(cid)
            merged.append(r)
    for r in fallback:
        cid = getattr(r, "best_clip_id", None) or r.clip_id
        if cid not in seen:
            seen.add(cid)
            merged.append(r)
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


def validate_results(results, api: GraphAPI, query_text: str) -> List[str]:
    issues: List[str] = []
    if not results:
        issues.append("EMPTY_RESULTS: No results returned")
        return issues

    scores = [r.score for r in results]
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            issues.append(f"SORT_ORDER: not descending at {i}")
            break

    clip_ids = [r.clip_id for r in results]
    if len(clip_ids) != len(set(clip_ids)):
        dupes = [cid for cid in clip_ids if clip_ids.count(cid) > 1]
        issues.append(f"DUPLICATES: {set(dupes)}")

    sample_ids = clip_ids[:5]
    props = api.get_node_properties(sample_ids)
    for cid in sample_ids:
        node_props = props.get(cid, {})
        if "video_id" in node_props:
            start = node_props.get("start")
            end = node_props.get("end")
            if start is not None and end is not None:
                try:
                    if float(end) < float(start):
                        issues.append(f"INVALID_TIMESTAMP: {cid}")
                except (ValueError, TypeError):
                    issues.append(f"TIMESTAMP_TYPE: {cid}")
    return issues


def build_result_entry(result, api: GraphAPI) -> Dict[str, Any]:
    cid = getattr(result, "best_clip_id", None) or result.clip_id
    props = api.get_node_properties([cid]).get(cid, {})
    entry: Dict[str, Any] = {
        "clip_id": cid,
        "score": round(float(result.score), 4),
        "entities": list(getattr(result, "entities", []) or []),
    }
    if "video_id" in props:
        entry["timestamp"] = {
            "start": props.get("start"),
            "end": props.get("end"),
            "video_id": props.get("video_id"),
        }
    else:
        entry["timestamp"] = None
    return entry


# ═════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    CEREBRAS_API_KEYS = [k.strip() for k in os.getenv("CEREBRAS_API_KEYS", "").split(",") if k.strip()]
    GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]

    if not CEREBRAS_API_KEYS and not GROQ_API_KEYS:
        print("❌ No API keys.")
        sys.exit(1)

    questions = load_questions(QNA_PATH, NUM_QUERIES)
    print(f"\n{'='*80}")
    print(f"  LIVE RETRIEVAL PIPELINE EVALUATION (POST-REFACTOR)")
    print(f"  Queries: {len(questions)} | Threshold: {LATENCY_WARN_THRESHOLD}s")
    print(f"  Query gap: {QUERY_GAP_SECONDS}s between queries")
    print(f"{'='*80}\n")

    mapping_db = os.path.join(PROJECT_ROOT, "outputs", "mapping.db")
    if not os.path.exists(mapping_db):
        print(f"⚠️  mapping.db not found at {mapping_db}")

    all_query_results: List[Dict[str, Any]] = []
    all_timings: List[Dict[str, float]] = []
    errors: List[Dict[str, str]] = []

    with GraphAPI(mapping_db_path=mapping_db) as api:
        # One-time: entity corpus
        print("⏳ Fetching entity corpus...")
        t0 = time.perf_counter()
        cypher = "MATCH (e:Entity) RETURN e.id AS id, e.name AS name"
        records = api._entity.execute_query(cypher)
        entity_corpus = [{"id": r["id"], "name": r["name"]} for r in records]
        print(f"✅ Entity corpus: {len(entity_corpus)} entities in {time.perf_counter()-t0:.2f}s\n")

        # One-time: decomposer init
        print("⏳ Initialising QueryDecomposer...")
        t0 = time.perf_counter()
        decomposer = QueryDecomposer(
            cerebras_keys=CEREBRAS_API_KEYS,
            groq_keys=GROQ_API_KEYS,
            entity_corpus=entity_corpus,
        )
        print(f"✅ Decomposer ready in {time.perf_counter()-t0:.2f}s\n")

        for idx, q_obj in enumerate(questions, start=1):
            query = q_obj["question"]
            q_id = q_obj.get("id", f"Q{idx}")
            q_level = q_obj.get("level", "unknown")

            # Wait between queries to avoid rate limiting
            if idx > 1:
                print(f"\n  ⏳ Waiting {QUERY_GAP_SECONDS}s before next query (rate limit cooldown)...")
                time.sleep(QUERY_GAP_SECONDS)

            print(f"\n{'─'*80}")
            print(f"  [{idx}/{len(questions)}] {q_id} ({q_level}) → {query}")
            print(f"{'─'*80}")

            api.clear_cache()

            query_result: Dict[str, Any] = {
                "query_id": q_id, "level": q_level, "query": query,
                "timings": {}, "results": [], "validation": [],
                "error": None, "used_fallback": False,
                "decomp_confidence": 0.0, "fast_path": False,
            }

            try:
                # ── Stage 1: Decomposition ───────────────────────────────────
                t0 = time.perf_counter()
                decomposition = decomposer.decompose(query)
                t1 = time.perf_counter()

                if isinstance(decomposition, FailureResponse):
                    raise RuntimeError(f"Decomposition failed: {decomposition.reason}")

                decomp_time = t1 - t0
                is_fast_path = "fast_path_no_llm" in decomposition.ambiguity_flags
                query_result["fast_path"] = is_fast_path
                query_result["decomp_confidence"] = round(decomposition.confidence, 4)

                print(f"  ✓ Decomposition: {decomp_time:.3f}s | "
                      f"entities={len(decomposition.entities)}, "
                      f"steps={len(decomposition.execution_plan)}, "
                      f"confidence={decomposition.confidence:.2f}"
                      f"{' [FAST PATH]' if is_fast_path else ''}")

                # ── Stage 2: Traversal (with guard) ──────────────────────────
                use_deep = decomposition.confidence >= MIN_CONFIDENCE_FOR_DEEP_TRAVERSAL
                if use_deep:
                    cfg = TraversalConfig(beam_width=15)
                else:
                    cfg = TraversalConfig(beam_width=10, max_depth=3)
                    print(f"  ⚡ Shallow traversal (confidence {decomposition.confidence:.2f} < {MIN_CONFIDENCE_FOR_DEEP_TRAVERSAL})")

                executor = TraversalExecutor(api, cfg)

                typed_steps = decomposition.get_typed_execution_plan()
                states: List[Any] = [TraversalState()]

                t2_start = time.perf_counter()
                for step in typed_steps:
                    if not states:
                        break
                    states = executor._dispatch(step, states)
                    states = executor._prune(states)
                candidates = executor._collect_candidates(states)
                t2 = time.perf_counter()

                traversal_time = t2 - t2_start
                print(f"  ✓ Traversal:     {traversal_time:.3f}s | "
                      f"states={len(states)}, candidates={len(candidates)}"
                      f"{' [DEEP]' if use_deep else ' [SHALLOW]'}")

                # ── Stage 3: Ranking ─────────────────────────────────────────
                t3_start = time.perf_counter()
                ranked_results = executor._rerank(candidates, decomposition, query)
                t3 = time.perf_counter()

                ranking_time = t3 - t3_start
                print(f"  ✓ Ranking:       {ranking_time:.3f}s | "
                      f"results={len(ranked_results)}")

                # ── Stage 4: Keyword Fallback (if needed) ────────────────────
                fallback_time = 0.0
                fallback_count = 0
                if not ranked_results or len(ranked_results) < 3:
                    t4_start = time.perf_counter()
                    fallback = keyword_fallback(query, api, top_k=15)
                    ranked_results = merge_results(ranked_results or [], fallback)
                    t4 = time.perf_counter()
                    fallback_time = t4 - t4_start
                    fallback_count = len(fallback)
                    query_result["used_fallback"] = True
                    print(f"  ✓ Fallback:      {fallback_time:.3f}s | "
                          f"added={fallback_count}, total={len(ranked_results)}")

                total_time = decomp_time + traversal_time + ranking_time + fallback_time

                print(f"  ✓ Total:         {total_time:.3f}s")
                if total_time > LATENCY_WARN_THRESHOLD:
                    print(f"  ⚠️  LATENCY WARNING: {total_time:.3f}s > {LATENCY_WARN_THRESHOLD}s")

                timings = {
                    "decomposition": round(decomp_time, 4),
                    "traversal": round(traversal_time, 4),
                    "ranking": round(ranking_time, 4),
                    "fallback": round(fallback_time, 4),
                    "total": round(total_time, 4),
                }
                query_result["timings"] = timings
                all_timings.append(timings)

                # Results
                result_entries = []
                for r in ranked_results[:20]:
                    result_entries.append(build_result_entry(r, api))
                query_result["results"] = result_entries

                # Validation
                issues = validate_results(ranked_results, api, query)
                query_result["validation"] = issues
                if issues:
                    print(f"  ⚠️  Validation: {issues}")
                else:
                    print(f"  ✅ Validation: PASS")

                # Top-3
                print(f"\n  Top Results:")
                for rank, r in enumerate(ranked_results[:3], start=1):
                    cid = getattr(r, "best_clip_id", None) or r.clip_id
                    props = api.get_node_properties([cid]).get(cid, {})
                    if "video_id" in props:
                        start_t = props.get("start", 0)
                        end_t = props.get("end", 0)
                        try:
                            time_str = f"[{float(start_t):.1f}s – {float(end_t):.1f}s]"
                        except (ValueError, TypeError):
                            time_str = f"[{start_t} – {end_t}]"
                        print(f"    {rank}. {cid} | score={r.score:.4f} | {time_str}")
                    else:
                        print(f"    {rank}. {cid} | score={r.score:.4f} | (entity)")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                query_result["error"] = error_msg
                errors.append({"query_id": q_id, "query": query, "error": error_msg})
                print(f"  ❌ ERROR: {error_msg}")
                logger.exception(f"Query {q_id} failed")

            all_query_results.append(query_result)

    # ═════════════════════════════════════════════════════════════════════════
    # Aggregated summary
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print(f"  AGGREGATED SUMMARY (POST-REFACTOR)")
    print(f"{'='*80}")

    if all_timings:
        def safe_stats(key):
            vals = [t[key] for t in all_timings]
            return {
                "mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }

        totals = [t["total"] for t in all_timings]
        totals_sorted = sorted(totals)
        p95_idx = max(0, int(len(totals_sorted) * 0.95) - 1)

        queries_with_results = sum(1 for q in all_query_results if q.get("results"))
        queries_with_fallback = sum(1 for q in all_query_results if q.get("used_fallback"))
        queries_fast_path = sum(1 for q in all_query_results if q.get("fast_path"))

        summary = {
            "num_queries": len(questions),
            "num_successful": len(all_timings),
            "num_with_results": queries_with_results,
            "num_empty": len(questions) - queries_with_results,
            "num_used_fallback": queries_with_fallback,
            "num_fast_path": queries_fast_path,
            "num_errors": len(errors),
            "avg_times": {
                "decomposition": round(statistics.mean([t["decomposition"] for t in all_timings]), 4),
                "traversal": round(statistics.mean([t["traversal"] for t in all_timings]), 4),
                "ranking": round(statistics.mean([t["ranking"] for t in all_timings]), 4),
                "fallback": round(statistics.mean([t["fallback"] for t in all_timings]), 4),
                "total": round(statistics.mean(totals), 4),
            },
            "detailed_stats": {
                "decomposition": safe_stats("decomposition"),
                "traversal": safe_stats("traversal"),
                "ranking": safe_stats("ranking"),
                "fallback": safe_stats("fallback"),
                "total": safe_stats("total"),
            },
            "p95_total_latency": round(totals_sorted[p95_idx], 4),
            "max_total_latency": round(max(totals), 4),
            "min_total_latency": round(min(totals), 4),
        }

        print(json.dumps(summary, indent=2))

        # Comparison table
        print(f"\n  {'Stage':<20} {'Avg':>8} {'Med':>8} {'Min':>8} {'Max':>8} {'Stdev':>8}")
        print(f"  {'─'*60}")
        for stage in ["decomposition", "traversal", "ranking", "fallback", "total"]:
            s = summary["detailed_stats"][stage]
            print(f"  {stage:<20} {s['mean']:>7.3f}s {s['median']:>7.3f}s "
                  f"{s['min']:>7.3f}s {s['max']:>7.3f}s {s['stdev']:>7.3f}s")

        # Comparison with baseline
        avg_total = summary["avg_times"]["total"]
        max_total = summary["max_total_latency"]
        p95_total = summary["p95_total_latency"]
        empty_count = len(questions) - queries_with_results
        print(f"\n  {'Metric':<30} {'Before':>10} {'After':>10}")
        print(f"  {'─'*50}")
        print(f"  {'Queries with results':<30} {'3/10':>10} {queries_with_results}/{len(questions):>8}")
        print(f"  {'Empty results':<30} {'7/10':>10} {empty_count}/{len(questions):>8}")
        print(f"  {'Avg total latency':<30} {'6.145s':>10} {avg_total:.3f}s")
        print(f"  {'Max total latency':<30} {'16.770s':>10} {max_total:.3f}s")
        print(f"  {'P95 total latency':<30} {'12.824s':>10} {p95_total:.3f}s")

        # ── LLM Usage Stats ──────────────────────────────────────────────────────────
        llm_stats = {
            "cerebras_handled": 0,
            "groq_handled": 0,
            "fast_path_handled": 0,
            "total_queries": 0,
            "avg_retries": 0.0,
            "latency": {"cerebras": [], "groq": []}
        }
        
        try:
            log_path = os.path.join(PROJECT_ROOT, "logs", "llm_usage_logs.jsonl")
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    logs = [json.loads(line) for line in f if line.strip()]
                    
                # Take only the last N logs (matching this run)
                run_logs = logs[-len(questions):]
                llm_stats["total_queries"] = len(run_logs)
                
                total_retries = 0
                for log in run_logs:
                    prov = log.get("final_provider")
                    if prov == "cerebras": llm_stats["cerebras_handled"] += 1
                    elif prov == "groq": llm_stats["groq_handled"] += 1
                    elif prov == "fast_path": llm_stats["fast_path_handled"] += 1
                    
                    attempts = log.get("provider_attempts", [])
                    total_retries += max(0, len(attempts) - 1)
                    
                    for attempt in attempts:
                        p = attempt.get("provider")
                        if p in llm_stats["latency"] and attempt.get("latency", 0) > 0:
                            llm_stats["latency"][p].append(attempt["latency"])
                
                if llm_stats["total_queries"] > 0:
                    llm_stats["avg_retries"] = round(total_retries / llm_stats["total_queries"], 2)
                    
                for p in ["cerebras", "groq"]:
                    lats = llm_stats["latency"][p]
                    llm_stats["latency"][f"{p}_avg"] = round(sum(lats)/len(lats), 3) if lats else 0.0
                    
                print(f"\n  LLM Provider Metrics:")
                print(f"  {'─'*50}")
                tq = max(1, llm_stats["total_queries"])
                print(f"  Cerebras handled : {llm_stats['cerebras_handled']} ({(llm_stats['cerebras_handled']/tq)*100:.1f}%)")
                print(f"  Groq fallback    : {llm_stats['groq_handled']} ({(llm_stats['groq_handled']/tq)*100:.1f}%)")
                print(f"  Fast path        : {llm_stats['fast_path_handled']} ({(llm_stats['fast_path_handled']/tq)*100:.1f}%)")
                print(f"  Avg retries/query: {llm_stats['avg_retries']}")
                print(f"  Avg Latency      : Cerebras={llm_stats['latency'].get('cerebras_avg', 0):.2f}s | Groq={llm_stats['latency'].get('groq_avg', 0):.2f}s")
                
                summary["llm_stats"] = llm_stats
        except Exception as e:
            print(f"  ⚠️  Could not compute LLM stats from logs: {e}")

        if errors:
            print(f"\n  ⚠️  {len(errors)} queries failed:")
            for e in errors:
                print(f"    - {e['query_id']}: {e['error'][:100]}")
    else:
        print("  ❌ No successful queries.")

    # Save
    output_path = os.path.join(PROJECT_ROOT, "retrieval_eval_results_postfix.json")
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_queries": len(questions),
            "version": "post-refactor",
        },
        "per_query": all_query_results,
        "summary": summary if all_timings else None,
        "errors": errors,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  📄 Results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
