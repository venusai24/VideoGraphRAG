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
        
        # STRICT MODEL POLICY
        self.provider_configs = [
            {"provider": "cerebras", "base_url": "https://api.cerebras.ai/v1", "model": "qwen-3-235b-a22b-instruct-2507"},
            {"provider": "groq", "base_url": "https://api.groq.com/openai/v1", "model": "qwen/qwen3-32b"},
            {"provider": "groq", "base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b"}
        ]

    def _call_api(self, client: openai.OpenAI, model: str, query: str) -> str:
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

    def _try_config(self, config: dict, query: str) -> Optional[QueryDecomposition]:
        provider_name = config["provider"]
        model = config["model"]
        base_url = config["base_url"]
        
        key_obj = self.key_manager.get_key(provider_name)
        if not key_obj:
            logger.warning(f"No healthy keys available for provider {provider_name}")
            return None

        client = openai.OpenAI(
            api_key=key_obj.key,
            base_url=base_url
        )

        for attempt in range(self.max_retries_per_key):
            try:
                logger.info(f"[{provider_name} - {model}] Attempt {attempt+1}...")
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
                # If it's an API error (timeout/rate limit or NotFound), mark failure and break
                key_obj.mark_failure(cooldown_seconds=60)
                break # Break out of retry loop for this specific key
                
        # If we exhausted retries without API exception (e.g., bad JSON), mark failed
        key_obj.mark_failure(cooldown_seconds=30)
        return None

    def execute_with_failover(self, query: str) -> QueryDecomposition | FailureResponse:
        for config in self.provider_configs:
            provider_name = config["provider"]
            logger.info(f"Attempting config: {provider_name} with {config['model']}...")
            
            # Try available keys for this provider
            num_keys = len(self.key_manager.pools[provider_name].keys)
            for _ in range(num_keys):
                result = self._try_config(config, query)
                if result:
                    return result
                if not self.key_manager.pools[provider_name].has_healthy_keys():
                    break
                    
        # All failed
        logger.error("All providers and configurations failed.")
        return FailureResponse(
            status="failure",
            reason="llm_unavailable_or_invalid_output",
            fallback=None
        )
