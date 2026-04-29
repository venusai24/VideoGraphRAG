import os
import logging
import json
import re
from typing import List, Dict

from dotenv import load_dotenv

# Ensure we can import from the project
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
parent_dir = os.path.dirname(project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from video_rag_query.graph_api import GraphAPI
from video_rag_query.query_decomposer import QueryDecomposer
from video_rag_query.traversal import TraversalExecutor, TraversalConfig, TraversalResult
from video_rag_query.models import QueryDecomposition, FailureResponse
from video_rag_query.answer_generator import AnswerGenerator
from video_rag_query.utils import extract_keywords

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Stop words for keyword extraction ────────────────────────────────────────
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

# ── Confidence threshold for traversal guard ─────────────────────────────────
MIN_CONFIDENCE_FOR_DEEP_TRAVERSAL = 0.5
FALLBACK_SCORE_MULTIPLIER = 0.6


def fetch_entity_corpus(api: GraphAPI) -> List[Dict[str, str]]:
    """Fetches all entities from the Entity graph to help with resolution."""
    logger.info("Fetching entity corpus from Neo4j...")
    cypher = "MATCH (e:Entity) RETURN e.id AS id, e.name AS name"
    records = api._entity.execute_query(cypher)
    return [{"id": r["id"], "name": r["name"]} for r in records]


def _extract_query_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from query text."""
    text = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
    words = text.split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def _keyword_fallback(query: str, api: GraphAPI, top_k: int = 15) -> List[TraversalResult]:
    """
    Fallback retrieval: extract keywords from the query and search across
    clip text fields (transcript, OCR, summary, keywords).

    Returns TraversalResult objects with a score multiplier applied (0.6×)
    so fallback results rank below confident graph-traversal results.
    """
    keywords = _extract_query_keywords(query)
    if not keywords:
        return []

    logger.info(f"Keyword fallback activated: keywords={keywords}")

    raw_results = api.keyword_fallback_search(keywords, top_k=top_k)
    if not raw_results:
        return []

    # Normalize scores: max possible per keyword is 1.2 (transcript) × 5 keywords = 6.0
    max_possible = len(keywords[:5]) * 1.2
    fallback_results = []

    for r in raw_results:
        normalised_score = min(1.0, r["score"] / max_possible) if max_possible > 0 else 0.0
        # Apply fallback multiplier so these rank below graph-traversal results
        final_score = normalised_score * FALLBACK_SCORE_MULTIPLIER

        fallback_results.append(TraversalResult(
            clip_id=r["clip_id"],
            score=round(final_score, 4),
            path=[],
            entities=[],
            explanation=f"Keyword fallback: matched {len(keywords)} keywords in clip text fields",
            best_clip_id=r["clip_id"],
        ))

    return fallback_results


def _merge_results(
    traversal_results: List[TraversalResult],
    fallback_results: List[TraversalResult],
) -> List[TraversalResult]:
    """
    Merge traversal results with fallback results.
    - Deduplicates by clip_id (traversal wins if both exist).
    - Sorts by score DESC.
    """
    seen = set()
    merged = []

    # Traversal results take priority
    for r in traversal_results:
        cid = getattr(r, "best_clip_id", None) or r.clip_id
        if cid not in seen:
            seen.add(cid)
            merged.append(r)

    # Add fallback results for clips not already in traversal results
    for r in fallback_results:
        cid = getattr(r, "best_clip_id", None) or r.clip_id
        if cid not in seen:
            seen.add(cid)
            merged.append(r)

    merged.sort(key=lambda x: x.score, reverse=True)
    return merged


def _build_answer_payload(query: str, results, api: GraphAPI) -> Dict[str, object]:
    """Build multimodal context payload for answer generation."""
    payload: Dict[str, object] = {"query": query, "results": []}
    if not results:
        return payload

    ranked_results = sorted(results, key=lambda res: float(res.score), reverse=True)
    rows: List[Dict[str, object]] = []
    seen = set()

    for res in ranked_results:
        candidate_clip_id = getattr(res, 'best_clip_id', None) or res.clip_id
        if candidate_clip_id in seen:
            continue

        props = api.get_node_properties([candidate_clip_id])
        clip_props = props.get(candidate_clip_id, {})

        # Keep only clip-like records for answer generation context.
        if "video_id" not in clip_props:
            continue

        seen.add(candidate_clip_id)
        rows.append(
            {
                "clip_id": candidate_clip_id,
                "score": float(res.score),
                "summary": str(clip_props.get("summary", "") or ""),
                "ocr_text": str(clip_props.get("ocr", "") or ""),
                "transcript": str(clip_props.get("transcript", "") or ""),
                "entities": list(getattr(res, "entities", []) or []),
                "timestamp": {
                    "start": clip_props.get("start"),
                    "end": clip_props.get("end"),
                    "video_id": clip_props.get("video_id"),
                },
                "clip_path": str(clip_props.get("clip_path", "") or ""),
            }
        )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    payload["results"] = rows
    return payload

def run_pipeline(query: str, cerebras_keys: List[str], groq_keys: List[str]):
    # 0. Resolve mapping database path
    # Look in project root / outputs / mapping.db
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mapping_db = os.path.join(base_dir, "outputs", "mapping.db")
    
    if not os.path.exists(mapping_db):
        logger.warning(f"Mapping database not found at {mapping_db}. Results may be incomplete.")

    # 1. Initialize GraphAPI
    with GraphAPI(mapping_db_path=mapping_db) as api:
        # 2. Fetch Entity Corpus (optional but recommended for better resolution)
        entity_corpus = fetch_entity_corpus(api)
        
        # 3. Initialize Decomposer
        decomposer = QueryDecomposer(
            cerebras_keys=cerebras_keys,
            groq_keys=groq_keys,
            entity_corpus=entity_corpus
        )
        
        # 4. Decompose Query
        logger.info(f"Processing query: {query}")
        decomposition = decomposer.decompose(query)
        
        if isinstance(decomposition, FailureResponse):
            logger.error(f"Decomposition failed: {decomposition.reason}")
            return
            
        # 5. Traversal guard: check entity confidence
        use_deep_traversal = decomposition.confidence >= MIN_CONFIDENCE_FOR_DEEP_TRAVERSAL
        
        if use_deep_traversal:
            # Full graph traversal
            executor = TraversalExecutor(api, TraversalConfig(beam_width=15))
            results = executor.execute(decomposition, query)
            logger.info(f"Deep traversal returned {len(results)} results (confidence={decomposition.confidence:.2f})")
        else:
            # Low confidence — skip deep traversal, use shallow resolve only
            logger.warning(
                f"Entity confidence {decomposition.confidence:.2f} < {MIN_CONFIDENCE_FOR_DEEP_TRAVERSAL} "
                f"— using shallow traversal + keyword fallback"
            )
            executor = TraversalExecutor(api, TraversalConfig(beam_width=10, max_depth=3))
            results = executor.execute(decomposition, query)

        # 6. Keyword fallback: if traversal returned nothing, search text fields
        if not results or len(results) < 3:
            logger.info("Activating keyword fallback (insufficient traversal results)")
            fallback = _keyword_fallback(query, api, top_k=15)
            results = _merge_results(results or [], fallback)
            if fallback:
                logger.info(f"Fallback added {len(fallback)} results, total now {len(results)}")

        # 7. Generate grounded answer from ranked multimodal evidence
        answer_generator = AnswerGenerator()
        answer_payload = _build_answer_payload(query, results, api)
        generated_answer = answer_generator.generate(answer_payload)
        
        # 8. Display Results
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        if not results:
            print("No matching results found in the graph.")
        else:
            print(f"Found {len(results)} potential results (Top 5 shown):")
            for i, res in enumerate(results[:5]):
                node_id = getattr(res, 'best_clip_id', None) or res.clip_id
                props = api.get_node_properties([node_id])
                node_props = props.get(node_id, {})
                
                # Determine if this is a Clip or an Entity
                is_clip = 'video_id' in node_props or "_" in node_id and any(c.isdigit() for c in node_id)
                
                print(f"\n[{i+1}] Result: {node_id} (Score: {res.score:.4f})")
                print(f"    Reasoning: {res.explanation}")
                
                if is_clip:
                    v_id = node_props.get('video_id', 'Unknown')
                    start = node_props.get('start', 0)
                    end = node_props.get('end', 0)
                    print(f"    🎬 VIDEO: {v_id} [{start:.2f}s - {end:.2f}s]")
                    print(f"    🔍 NEO4J (Clip Graph): MATCH (c:Clip {{id: '{node_id}'}}) RETURN c")
                else:
                    print(f"    👤 ENTITY Type: {node_props.get('type', 'Unknown')}")
                    print(f"    🔍 NEO4J (Entity Graph): MATCH (e:Entity {{id: '{node_id}'}}) RETURN e")
                    
                    # Show the clip that actually matched the keywords (if any)
                    source_clip_id = getattr(res, 'best_clip_id', None)
                    if source_clip_id:
                        print(f"    🎯 MATCH SOURCE: {source_clip_id} (Validated Match)")
                        print(f"       👉 NEO4J (Clip Graph): MATCH (c:Clip {{id: '{source_clip_id}'}}) RETURN c")
                    elif api._mapping_store:
                        # Fallback: Show sample clips where this entity appears (via SQLite)
                        mapping_clips = api._mapping_store.get_clips_for_entity(node_id)
                        if mapping_clips:
                            top_clip = mapping_clips[0]
                            print(f"    🔗 APPEARS IN (Sample): {top_clip['clip_id']} (Confidence: {top_clip['confidence']:.2f})")
                            print(f"       👉 NEO4J (Clip Graph): MATCH (c:Clip {{id: '{top_clip['clip_id']}'}}) RETURN c")
                
                if 'transcript' in node_props and node_props['transcript']:
                    snippet = node_props['transcript'][:150].replace('\n', ' ')
                    print(f"    📝 Snippet: {snippet}...")

            print("\n" + "="*80)
            print("GROUNDED ANSWER")
            print("="*80)
            print(json.dumps(generated_answer, indent=2, ensure_ascii=False))

            return generated_answer

            print("\n" + "="*80)
            print("GROUNDED ANSWER")
            print("="*80)
            print(json.dumps(generated_answer, indent=2, ensure_ascii=False))

            return generated_answer

if __name__ == "__main__":
    load_dotenv()
    
    # Get keys from environment
    CEREBRAS_API_KEYS = [k.strip() for k in os.getenv("CEREBRAS_API_KEYS", "").split(",") if k.strip()]
    GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
    
    if not CEREBRAS_API_KEYS and not GROQ_API_KEYS:
        print("Error: No API keys found. Please set CEREBRAS_API_KEYS or GROQ_API_KEYS in .env")
        sys.exit(1)
        
    questions = [
        "Which Airport was built on Swamp?"
    ]
    
    for q in questions:
        try:
            run_pipeline(q, CEREBRAS_API_KEYS, GROQ_API_KEYS)
        except Exception as e:
            logger.error(f"Error processing query '{q}': {e}")
            continue
