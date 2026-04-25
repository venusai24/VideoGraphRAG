import time
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
