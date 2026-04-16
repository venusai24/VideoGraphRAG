from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Sequence

from ..contracts import VisionClipInput, VisionExtraction
from ..vision import (
    build_qwen_prompt,
    build_repair_prompt,
    file_to_data_uri,
    validate_vision_response,
)


class OpenAICompatibleVisionClient:
    """
    Minimal OpenAI-compatible client suitable for Qwen/VLM endpoints served by vLLM.

    The client validates model output with Pydantic and retries once with a repair
    prompt if the initial response is not valid JSON.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout_sec: int = 120,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def infer(self, batch: Sequence[VisionClipInput]) -> Sequence[VisionExtraction]:
        outputs: List[VisionExtraction] = []
        for item in batch:
            prompt = build_qwen_prompt(item)
            response_text = self._chat_completion(
                self._build_multimodal_messages(item, prompt)
            )
            try:
                outputs.append(validate_vision_response(response_text))
            except Exception:
                repair_text = self._chat_completion(
                    [
                        {
                            "role": "user",
                            "content": build_repair_prompt(response_text),
                        }
                    ]
                )
                outputs.append(
                    validate_vision_response(
                        response_text,
                        repaired_response=repair_text,
                    )
                )
        return outputs

    def _build_multimodal_messages(
        self,
        clip_input: VisionClipInput,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in clip_input.sampled_frames:
            if not frame.image_path:
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": file_to_data_uri(frame.image_path)},
                }
            )
        return [{"role": "user", "content": content}]

    def _chat_completion(self, messages: Sequence[Dict[str, Any]]) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": list(messages),
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else {}
                ),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
        message = body["choices"][0]["message"]["content"]
        if isinstance(message, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in message
            )
        return str(message)
