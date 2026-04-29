import time
import os
import re
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

class APIKey:
    def __init__(self, key: str, provider: str):
        self.key = key
        self.provider = provider
        self.failures = 0
        self.cooldown_until = 0.0

    def is_healthy(self) -> bool:
        return time.time() > self.cooldown_until

    def mark_failure(self, cooldown_seconds: float = 30.0):
        self.failures += 1
        self.cooldown_until = time.time() + cooldown_seconds
        logger.warning(f"[{self.provider}] Key marked as failed. Cooldown for {cooldown_seconds}s. Total failures: {self.failures}")

    def mark_success(self):
        if self.failures > 0:
            logger.info(f"[{self.provider}] Key recovered.")
        self.failures = 0
        self.cooldown_until = 0.0

class ProviderPool:
    def __init__(self, provider_name: str, keys: List[str]):
        self.provider_name = provider_name
        self.keys = [APIKey(k, provider_name) for k in keys if k.strip()]
        self.current_idx = 0

    def get_next_key(self) -> Optional[APIKey]:
        if not self.keys:
            return None
        
        # Try to find a healthy key starting from current_idx (round-robin)
        start_idx = self.current_idx
        for _ in range(len(self.keys)):
            key_obj = self.keys[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.keys)
            if key_obj.is_healthy():
                return key_obj
                
        return None

    def has_healthy_keys(self) -> bool:
        return any(k.is_healthy() for k in self.keys)

class KeyManager:
    def __init__(self, cerebras_keys: List[str], groq_keys: List[str]):
        self.pools = {
            "cerebras": ProviderPool("cerebras", cerebras_keys),
            "groq": ProviderPool("groq", groq_keys)
        }

    def get_key(self, provider: str) -> Optional[APIKey]:
        if provider not in self.pools:
            return None
        return self.pools[provider].get_next_key()


class GeminiKeyManager:
    """
    Round-robin Gemini API key manager with cooldown handling.

    Key discovery order:
      1) GEMINI_API_KEYS / GOOGLE_API_KEYS (comma-separated)
      2) GEMINI_API_KEY / GOOGLE_API_KEY (single key)
      3) Parse .env curl snippets containing X-goog-api-key headers
    """

    _CURL_KEY_PATTERN = re.compile(r"X-goog-api-key:\s*([A-Za-z0-9_-]+)")

    def __init__(self, gemini_keys: List[str], cooldown_seconds: float = 60.0):
        self.cooldown_seconds = cooldown_seconds
        self.pool = ProviderPool("gemini", gemini_keys)

    @classmethod
    def from_env(cls, env_path: str = ".env", cooldown_seconds: float = 60.0) -> "GeminiKeyManager":
        keys: List[str] = []

        for var in ("GEMINI_API_KEYS", "GOOGLE_API_KEYS"):
            value = os.getenv(var, "")
            if value:
                keys.extend([k.strip() for k in value.split(",") if k.strip()])

        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            value = os.getenv(var, "").strip()
            if value:
                keys.append(value)

        # If no explicit env vars are present, parse curl snippets in .env.
        if not keys and env_path and os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                keys.extend(cls._CURL_KEY_PATTERN.findall(content))
            except Exception as exc:
                logger.warning("Failed to parse Gemini keys from %s: %s", env_path, exc)

        # Deduplicate while preserving order.
        unique_keys: List[str] = []
        seen = set()
        for key in keys:
            if key and key not in seen:
                seen.add(key)
                unique_keys.append(key)

        return cls(unique_keys, cooldown_seconds=cooldown_seconds)

    @property
    def key_count(self) -> int:
        return len(self.pool.keys)

    def get_next_key(self) -> Optional[APIKey]:
        return self.pool.get_next_key()

    def has_healthy_keys(self) -> bool:
        return self.pool.has_healthy_keys()

    def mark_failure(self, key_obj: Optional[APIKey], cooldown_seconds: Optional[float] = None):
        if key_obj is None:
            return
        key_obj.mark_failure(cooldown_seconds or self.cooldown_seconds)

    def mark_success(self, key_obj: Optional[APIKey]):
        if key_obj is None:
            return
        key_obj.mark_success()

    def reset_cooldowns(self):
        """Allow a fresh key-rotation pass (used when switching model tiers)."""
        for key_obj in self.pool.keys:
            key_obj.cooldown_until = 0.0
