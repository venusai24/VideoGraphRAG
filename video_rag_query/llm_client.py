import json
import logging
from typing import Optional, List
from pydantic import ValidationError
import openai

from .models import QueryDecomposition, FailureResponse
from .key_manager import KeyManager
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

VALID_OPERATIONS = {"resolve_entity", "traverse", "filter", "temporal_traverse", "extract"}
VALID_EDGES = {"APPEARS_IN", "NEXT", "SHARES_ENTITY", "RELATED_TO"}


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
    def __init__(self, key_manager: KeyManager, max_retries_per_key: int = 3):
        self.key_manager = key_manager
        self.max_retries_per_key = max_retries_per_key

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
        )
        return response.choices[0].message.content

    def _try_key_for_config(self, config: dict, query: str) -> Optional[QueryDecomposition]:
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

        client = openai.OpenAI(api_key=key_obj.key, base_url=base_url)

        for attempt in range(self.max_retries_per_key):
            try:
                logger.info(f"[{provider_name}/{model}] Key attempt {attempt + 1}/{self.max_retries_per_key}")
                raw = self._call_api(client, model, query)

                parsed = json.loads(raw)

                # Structural validation of execution_plan before Pydantic
                plan = parsed.get("execution_plan", [])
                violations = _validate_execution_plan(plan)
                if violations:
                    logger.error(f"[{provider_name}] Execution plan violations: {violations}")
                    # Do not break — retry may yield better output
                    continue

                validated = QueryDecomposition(**parsed)
                key_obj.mark_success()
                logger.info(f"[{provider_name}/{model}] Success.")
                return validated

            except json.JSONDecodeError as e:
                logger.error(f"[{provider_name}] JSON decode error (attempt {attempt+1}): {e}")
            except ValidationError as e:
                logger.error(f"[{provider_name}] Pydantic validation error (attempt {attempt+1}): {e}")
            except openai.RateLimitError as e:
                logger.warning(f"[{provider_name}] Rate limit hit: {e}")
                key_obj.mark_failure(cooldown_seconds=90)
                return None  # Stop retrying this key immediately
            except openai.APIStatusError as e:
                logger.error(f"[{provider_name}] API status error {e.status_code}: {e}")
                key_obj.mark_failure(cooldown_seconds=60)
                return None
            except Exception as e:
                logger.error(f"[{provider_name}] Unexpected error (attempt {attempt+1}): {e}")
                key_obj.mark_failure(cooldown_seconds=60)
                return None

        # Exhausted retries on JSON/validation errors — soft failure, apply short cooldown
        key_obj.mark_failure(cooldown_seconds=30)
        return None

    def execute_with_failover(self, query: str) -> QueryDecomposition | FailureResponse:
        """
        Strict failover: exhaust ALL keys for a given (provider, model) config before
        moving to the next config. NEVER silently downgrade models.
        """
        for config in self.provider_configs:
            provider_name = config["provider"]
            pool = self.key_manager.pools[provider_name]
            num_keys = len(pool.keys)

            logger.info(f"Trying config [{provider_name} / {config['model']}] — {num_keys} key(s) available")

            for _ in range(num_keys):
                if not pool.has_healthy_keys():
                    logger.warning(f"[{provider_name}] No healthy keys left, moving to next config.")
                    break

                result = self._try_key_for_config(config, query)
                if result is not None:
                    return result

        logger.error("All provider configs exhausted. Returning FailureResponse.")
        return FailureResponse(
            status="failure",
            reason="llm_unavailable_or_invalid_output",
            fallback=None,
        )
