import json
import logging
from typing import Optional
from pydantic import ValidationError
import openai

from .models import QueryDecomposition, FailureResponse
from .key_manager import KeyManager
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, key_manager: KeyManager, max_retries_per_key: int = 3):
        self.key_manager = key_manager
        self.max_retries_per_key = max_retries_per_key
        
        self.providers = {
            "cerebras": {
                "base_url": "https://api.cerebras.ai/v1",
                "model": "llama3.1-70b" # Fallback if qwen3 not there
            },
            "groq": {
                "base_url": "https://api.groq.com/openai/v1",
                "model": "qwen-2.5-32b" # Will fallback to llama if it fails
            }
        }

    def _call_api(self, client: openai.OpenAI, model: str, query: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except openai.NotFoundError:
            # Fallback models if the requested ones don't exist
            alt_model = "llama-3.3-70b-versatile" if "groq" in client.base_url.host else "llama3.1-8b"
            logger.warning(f"Model {model} not found, falling back to {alt_model}")
            response = client.chat.completions.create(
                model=alt_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

    def _try_provider(self, provider_name: str, query: str) -> Optional[QueryDecomposition]:
        key_obj = self.key_manager.get_key(provider_name)
        if not key_obj:
            logger.warning(f"No healthy keys available for provider {provider_name}")
            return None

        client = openai.OpenAI(
            api_key=key_obj.key,
            base_url=self.providers[provider_name]["base_url"]
        )
        model = self.providers[provider_name]["model"]

        for attempt in range(self.max_retries_per_key):
            try:
                logger.info(f"[{provider_name}] Attempt {attempt+1} for query...")
                raw_response = self._call_api(client, model, query)
                
                # Parse and validate with Pydantic
                parsed_json = json.loads(raw_response)
                validated_model = QueryDecomposition(**parsed_json)
                
                key_obj.mark_success()
                return validated_model
                
            except json.JSONDecodeError as e:
                logger.error(f"[{provider_name}] JSON parsing failed: {str(e)}")
            except ValidationError as e:
                logger.error(f"[{provider_name}] Schema validation failed: {str(e)}")
            except Exception as e:
                logger.error(f"[{provider_name}] API call failed: {str(e)}")
                # If it's an API error (timeout/rate limit), mark failure and break to try next key or provider
                key_obj.mark_failure(cooldown_seconds=60)
                break # Break out of retry loop for this specific key
                
        # If we exhausted retries for this key without an API exception, mark it as failed due to bad outputs
        key_obj.mark_failure(cooldown_seconds=30)
        return None

    def execute_with_failover(self, query: str) -> QueryDecomposition | FailureResponse:
        # 1. Try Cerebras
        logger.info("Attempting Cerebras primary...")
        # Try a few different keys if the first one fails
        for _ in range(len(self.key_manager.pools["cerebras"].keys)):
            result = self._try_provider("cerebras", query)
            if result:
                return result
            if not self.key_manager.pools["cerebras"].has_healthy_keys():
                break
                
        # 2. Try Groq (Fallback)
        logger.warning("Cerebras failed. Falling back to Groq...")
        for _ in range(len(self.key_manager.pools["groq"].keys)):
            result = self._try_provider("groq", query)
            if result:
                return result
            if not self.key_manager.pools["groq"].has_healthy_keys():
                break
                
        # 3. All failed
        logger.error("All providers and keys failed.")
        return FailureResponse(
            status="failure",
            reason="llm_unavailable_or_invalid_output",
            fallback=None
        )
