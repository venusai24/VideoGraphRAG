import json
import logging
import time
import signal
import threading
from typing import Optional, List
from pydantic import ValidationError
import openai

from .models import QueryDecomposition, FailureResponse
from .key_manager import KeyManager
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

VALID_OPERATIONS = {"resolve_entity", "traverse", "filter", "temporal_traverse", "extract"}
VALID_EDGES = {"APPEARS_IN", "NEXT", "SHARES_ENTITY", "RELATED_TO"}

# ── Temporal relation normalization map ──────────────────────────────────────
_TEMPORAL_RELATION_MAP = {
    "between": "during",
    "earlier than": "before",
    "later than": "after",
    "simultaneous": "during",
    "concurrent": "during",
    "prior": "before",
    "following": "after",
    "subsequent": "after",
}

# ── Edge normalization map ───────────────────────────────────────────────────
_EDGE_REMAP = {
    "SUBCLASS_OF": "RELATED_TO",
    "HAS_PART": "RELATED_TO",
    "PART_OF": "RELATED_TO",
    "IS_A": "RELATED_TO",
    "BELONGS_TO": "APPEARS_IN",
    "CONTAINS": "APPEARS_IN",
}


def _sanitize_raw_decomposition(parsed: dict) -> dict:
    """
    STRICT post-LLM sanitizer. Fixes known schema issues BEFORE Pydantic validation.
    Returns sanitized dict. Logs all corrections made.
    """
    corrections = []

    # 1. Fix temporal_constraints.relation
    tc = parsed.get("temporal_constraints")
    if isinstance(tc, dict):
        relation = tc.get("relation", "none")
        if relation not in ("before", "after", "during", "none"):
            mapped = _TEMPORAL_RELATION_MAP.get(relation.lower().strip(), "none")
            corrections.append(f"temporal_relation: '{relation}' -> '{mapped}'")
            tc["relation"] = mapped
        direction = tc.get("direction", "none")
        if direction not in ("forward", "backward", "neutral", "none"):
            corrections.append(f"temporal_direction: '{direction}' -> 'none'")
            tc["direction"] = "none"
    elif tc is None:
        parsed["temporal_constraints"] = {"relation": "none", "anchor_event": None, "direction": "none"}
        corrections.append("temporal_constraints: missing -> default")

    # 2. Fix execution_plan edges
    plan = parsed.get("execution_plan", [])
    if isinstance(plan, list):
        for step in plan:
            if not isinstance(step, dict):
                continue
            edge = step.get("edge")
            if edge and edge not in VALID_EDGES:
                remapped = _EDGE_REMAP.get(edge, "RELATED_TO")
                corrections.append(f"edge: '{edge}' -> '{remapped}' (step {step.get('step', '?')})")
                step["edge"] = remapped
            op = step.get("operation")
            if op and op not in VALID_OPERATIONS:
                corrections.append(f"operation: '{op}' removed (step {step.get('step', '?')})")
                step["operation"] = "extract"  # Safe no-op fallback

    # 3. Fill missing required top-level fields with safe defaults
    defaults = {
        "query_type": "direct",
        "entities": [],
        "actions": [],
        "sub_queries": [],
        "execution_plan": [],
        "confidence": 0.5,
        "ambiguity_flags": [],
    }
    for key, default_val in defaults.items():
        if key not in parsed:
            parsed[key] = default_val
            corrections.append(f"missing_field: '{key}' -> default")

    # 4. Fix sub_queries edge references
    sq_list = parsed.get("sub_queries", [])
    if isinstance(sq_list, list):
        for sq in sq_list:
            if isinstance(sq, dict):
                comps = sq.get("required_graph_components", [])
                if isinstance(comps, list):
                    fixed_comps = []
                    for c in comps:
                        if c in _EDGE_REMAP:
                            fixed_comps.append(_EDGE_REMAP[c])
                        else:
                            fixed_comps.append(c)
                    sq["required_graph_components"] = fixed_comps

    if corrections:
        logger.info(f"[sanitizer] Applied {len(corrections)} corrections: {corrections}")

    return parsed

# ── Hard timeout for LLM calls ──────────────────────────────────────────────
MAX_SINGLE_CALL_TIMEOUT = 8.0   # seconds per API call
MAX_TOTAL_DECOMP_TIMEOUT = 15.0  # seconds for the entire execute_with_failover

# ── Retry limits ─────────────────────────────────────────────────────────────
MAX_RETRIES_PER_KEY = 2   # Hard cap


def _validate_execution_plan(plan: list) -> List[str]:
    """
    Validates execution plan steps structurally. Returns list of violation messages.
    """
    violations = []
    for step in plan:
        if not isinstance(step, dict):
            violations.append(f"Step is not a dict: {step!r}")
            continue
        op = step.get("operation")
        if op not in VALID_OPERATIONS:
            violations.append(f"Step {step.get('step')}: unknown operation '{op}'")
        if op == "traverse":
            edge = step.get("edge")
            if edge not in VALID_EDGES:
                violations.append(f"Step {step.get('step')}: invalid edge '{edge}'")
            if not step.get("from") or not step.get("to"):
                violations.append(f"Step {step.get('step')}: traverse missing 'from' or 'to'")
        if op == "temporal_traverse":
            if step.get("edge") != "NEXT":
                violations.append(f"Step {step.get('step')}: temporal_traverse must use NEXT edge")
            if step.get("direction") not in {"forward", "backward", "neutral"}:
                violations.append(f"Step {step.get('step')}: invalid temporal direction")
        if op == "extract":
            if not step.get("fields"):
                violations.append(f"Step {step.get('step')}: extract missing 'fields'")
        if op == "resolve_entity":
            if not step.get("input") or not step.get("output"):
                violations.append(f"Step {step.get('step')}: resolve_entity missing 'input' or 'output'")
    return violations


class LLMClient:
    def __init__(self, key_manager: KeyManager, max_retries_per_key: int = 2):
        self.key_manager = key_manager
        self.max_retries_per_key = min(max_retries_per_key, MAX_RETRIES_PER_KEY)

        # STRICT MODEL POLICY — exhausted left-to-right, NO model downgrade within provider
        self.provider_configs = [
            {"provider": "cerebras", "base_url": "https://api.cerebras.ai/v1",      "model": "qwen-3-235b-a22b-instruct-2507"},
            {"provider": "groq",     "base_url": "https://api.groq.com/openai/v1",  "model": "qwen/qwen3-32b"},
            {"provider": "groq",     "base_url": "https://api.groq.com/openai/v1",  "model": "openai/gpt-oss-120b"},
        ]

    def _call_api(self, client: openai.OpenAI, model: str, query: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=MAX_SINGLE_CALL_TIMEOUT,
        )
        return response.choices[0].message.content

    def _try_key_for_config(self, config: dict, query: str, provider_attempts: List[dict]) -> Optional[QueryDecomposition]:
        """
        Try a SINGLE healthy key from the provider pool for this config.
        Returns validated QueryDecomposition on success, None otherwise.
        """
        provider_name = config["provider"]
        model = config["model"]
        base_url = config["base_url"]

        key_obj = self.key_manager.get_key(provider_name)
        if not key_obj:
            return None

        client = openai.OpenAI(
            api_key=key_obj.key,
            base_url=base_url,
            timeout=MAX_SINGLE_CALL_TIMEOUT,
        )

        for attempt in range(self.max_retries_per_key):
            attempt_log = {
                "provider": provider_name,
                "model": model,
                "status": "failed",
                "error": None,
                "latency": 0.0
            }
            start_time = time.time()
            try:
                logger.info(f"[{provider_name}/{model}] Key attempt {attempt + 1}/{self.max_retries_per_key}")
                raw = self._call_api(client, model, query)
                
                parsed = json.loads(raw)

                # STRICT: sanitize BEFORE any validation
                parsed = _sanitize_raw_decomposition(parsed)

                # Structural validation of execution_plan before Pydantic
                plan = parsed.get("execution_plan", [])
                violations = _validate_execution_plan(plan)
                if violations:
                    error_msg = f"Execution plan violations: {violations}"
                    logger.error(f"[{provider_name}] {error_msg}")
                    attempt_log["error"] = error_msg
                    attempt_log["latency"] = time.time() - start_time
                    provider_attempts.append(attempt_log)
                    continue

                validated = QueryDecomposition(**parsed)
                key_obj.mark_success()
                logger.info(f"[{provider_name}/{model}] Success.")
                
                attempt_log["status"] = "success"
                attempt_log["latency"] = time.time() - start_time
                provider_attempts.append(attempt_log)
                
                return validated

            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {e}"
                logger.error(f"[{provider_name}] {error_msg} (attempt {attempt+1})")
                attempt_log["error"] = error_msg
                attempt_log["latency"] = time.time() - start_time
                provider_attempts.append(attempt_log)
            except ValidationError as e:
                error_msg = f"Pydantic validation error: {e}"
                logger.error(f"[{provider_name}] {error_msg} (attempt {attempt+1})")
                attempt_log["error"] = error_msg
                attempt_log["latency"] = time.time() - start_time
                provider_attempts.append(attempt_log)
            except openai.RateLimitError as e:
                error_msg = f"Rate limit hit: {e}"
                logger.warning(f"[{provider_name}] {error_msg}")
                attempt_log["error"] = error_msg
                attempt_log["latency"] = time.time() - start_time
                provider_attempts.append(attempt_log)
                # Shorter cooldown: 30s instead of 90s so keys recover faster
                key_obj.mark_failure(cooldown_seconds=30)
                return None  # Stop retrying this key immediately
            except openai.APITimeoutError as e:
                error_msg = f"API timeout ({MAX_SINGLE_CALL_TIMEOUT}s): {e}"
                logger.warning(f"[{provider_name}] {error_msg}")
                attempt_log["error"] = error_msg
                attempt_log["latency"] = time.time() - start_time
                provider_attempts.append(attempt_log)
                key_obj.mark_failure(cooldown_seconds=15)
                return None  # Don't retry on timeout — move to next config
            except openai.APIStatusError as e:
                error_msg = f"API status error {e.status_code}: {e}"
                logger.error(f"[{provider_name}] {error_msg}")
                attempt_log["error"] = error_msg
                attempt_log["latency"] = time.time() - start_time
                provider_attempts.append(attempt_log)
                key_obj.mark_failure(cooldown_seconds=30)
                return None
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                logger.error(f"[{provider_name}] {error_msg} (attempt {attempt+1})")
                attempt_log["error"] = error_msg
                attempt_log["latency"] = time.time() - start_time
                provider_attempts.append(attempt_log)
                key_obj.mark_failure(cooldown_seconds=30)
                return None

        # Exhausted retries on JSON/validation errors — soft failure, apply short cooldown
        key_obj.mark_failure(cooldown_seconds=15)
        return None

    def execute_with_failover(self, query: str) -> "QueryDecomposition | FailureResponse":
        """
        Strict failover with hard total timeout:
          - Exhaust ALL keys for a given (provider, model) config before
            moving to the next config.
          - NEVER silently downgrade models.
          - Abort if total elapsed exceeds MAX_TOTAL_DECOMP_TIMEOUT.
        """
        t_start = time.perf_counter()
        
        llm_logs = {
            "query": query,
            "provider_attempts": [],
            "final_provider": None,
            "final_model": None,
            "total_llm_time": 0.0
        }

        for config in self.provider_configs:
            # Check total timeout
            elapsed = time.perf_counter() - t_start
            if elapsed > MAX_TOTAL_DECOMP_TIMEOUT:
                logger.warning(
                    f"Total decomposition timeout ({MAX_TOTAL_DECOMP_TIMEOUT}s) reached "
                    f"after {elapsed:.1f}s. Aborting LLM calls."
                )
                break

            provider_name = config["provider"]
            pool = self.key_manager.pools[provider_name]
            num_keys = len(pool.keys)

            logger.info(f"Trying config [{provider_name} / {config['model']}] — {num_keys} key(s) available")

            for _ in range(num_keys):
                # Re-check timeout before each key attempt
                if time.perf_counter() - t_start > MAX_TOTAL_DECOMP_TIMEOUT:
                    break

                if not pool.has_healthy_keys():
                    logger.warning(f"[{provider_name}] No healthy keys left, moving to next config.")
                    break

                result = self._try_key_for_config(config, query, llm_logs["provider_attempts"])
                if result is not None:
                    llm_logs["final_provider"] = config["provider"]
                    llm_logs["final_model"] = config["model"]
                    llm_logs["total_llm_time"] = time.perf_counter() - t_start
                    result.llm_logs = llm_logs
                    return result

        logger.error("All provider configs exhausted or timeout. Returning FailureResponse.")
        llm_logs["total_llm_time"] = time.perf_counter() - t_start
        return FailureResponse(
            status="failure",
            reason="llm_unavailable_or_timeout",
            fallback=None,
            llm_logs=llm_logs
        )
