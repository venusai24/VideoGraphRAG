from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

class APIKey:
    def __init__(self, key: str, provider: str):
        self.key = key
        self.provider = provider
        self.failures = 0
        self.cooldown_until = 0.0

    def is_healthy(self) -> bool:
        return time.time() > self.cooldown_until

    def mark_failure(self, cooldown_seconds: float = 60.0):
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
    Round-robin Gemini API key manager with per-key cooldown.

    Keys are loaded from environment variables and can also be extracted from
    curl snippets in the .env file ("X-goog-api-key: ...") to support existing
    local workflows.
    """

    _CSV_ENV_VARS = ("GEMINI_API_KEYS", "GOOGLE_API_KEYS")
    _SINGLE_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    _INDEXED_PREFIXES = ("GEMINI_API_KEY_", "GOOGLE_API_KEY_")
    _CURL_KEY_PATTERN = re.compile(r"X-goog-api-key:\s*([A-Za-z0-9_-]{20,})", re.IGNORECASE)

    def __init__(self, keys: Sequence[str], cooldown_seconds: float = 60.0):
        cleaned_keys = self._clean_keys(keys)
        if not cleaned_keys:
            raise RuntimeError(
                "Missing Gemini API keys. Set GEMINI_API_KEYS (comma-separated) "
                "or GEMINI_API_KEY in .env."
            )

        self.pool = ProviderPool("gemini", cleaned_keys)
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))

    @property
    def key_count(self) -> int:
        return len(self.pool.keys)

    def has_healthy_keys(self) -> bool:
        return self.pool.has_healthy_keys()

    def get_next_key(self) -> Optional[APIKey]:
        return self.pool.get_next_key()

    def mark_failure(self, key_obj: APIKey, cooldown_seconds: Optional[float] = None) -> None:
        cooldown = self.cooldown_seconds if cooldown_seconds is None else max(1.0, cooldown_seconds)
        key_obj.mark_failure(cooldown_seconds=cooldown)

    def mark_success(self, key_obj: APIKey) -> None:
        key_obj.mark_success()

    @classmethod
    def from_env(
        cls,
        *,
        cooldown_seconds: float = 60.0,
        dotenv_path: Optional[str] = None,
    ) -> "GeminiKeyManager":
        resolved_dotenv = dotenv_path or find_dotenv(usecwd=True)
        if resolved_dotenv:
            load_dotenv(resolved_dotenv, override=False)
        else:
            load_dotenv(override=False)

        keys = cls._collect_env_keys()
        if not keys:
            keys = cls._extract_keys_from_dotenv_text(resolved_dotenv)

        return cls(keys, cooldown_seconds=cooldown_seconds)

    @classmethod
    def _collect_env_keys(cls) -> List[str]:
        keys: List[str] = []

        for var_name in cls._CSV_ENV_VARS:
            raw = os.getenv(var_name, "")
            if raw:
                keys.extend(part.strip() for part in raw.split(","))

        for var_name in cls._SINGLE_ENV_VARS:
            raw = os.getenv(var_name, "")
            if raw:
                keys.append(raw.strip())

        for name, value in os.environ.items():
            if not value:
                continue
            if any(name.startswith(prefix) for prefix in cls._INDEXED_PREFIXES):
                keys.append(value.strip())

        return cls._clean_keys(keys)

    @classmethod
    def _extract_keys_from_dotenv_text(cls, dotenv_path: Optional[str]) -> List[str]:
        if not dotenv_path:
            return []

        path = Path(dotenv_path)
        if not path.exists() or not path.is_file():
            return []

        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []

        matches = cls._CURL_KEY_PATTERN.findall(text)
        return cls._clean_keys(matches)

    @staticmethod
    def _clean_keys(keys: Sequence[str]) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        for key in keys:
            candidate = str(key or "").strip().strip('"').strip("'")
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            cleaned.append(candidate)
        return cleaned
