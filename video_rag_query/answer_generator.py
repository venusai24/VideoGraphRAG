import json
import logging
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set

from .key_manager import GeminiKeyManager

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - handled at runtime when package is absent
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)

MODEL_PRIORITY = [
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-3-flash-lite",
]

SYSTEM_MESSAGE = (
    "You are a grounded video reasoning system.\n\n"
    "Use ONLY the provided clips and context.\n"
    "Do NOT hallucinate.\n"
    "If evidence is insufficient, say so.\n"
    "Always cite clip_ids."
)

STRICT_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "citations", "reasoning", "confidence"],
}

DEFAULT_FALLBACK_RESPONSE: Dict[str, Any] = {
    "answer": "Insufficient evidence to answer the query.",
    "citations": [],
    "reasoning": "No reliable supporting clips found.",
    "confidence": 0.0,
}

MAX_CONTEXT_TOKENS = 250_000
RESPONSE_RESERVE_TOKENS = 4_000

_CLIP_ID_PATTERN = re.compile(r"\bclip_[A-Za-z0-9_-]+\b")
_ENTITY_ID_PATTERN = re.compile(r"\b(?:person|location|topic|object|event)_[A-Za-z0-9_-]+\b")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with", "which", "who", "whom",
    "what", "where", "when", "how", "why", "has", "been", "somehow", "them", "from",
}


@dataclass
class EvidenceClip:
    clip_id: str
    score: float
    summary: str = ""
    ocr_text: str = ""
    transcript: str = ""
    entities: List[str] = field(default_factory=list)
    timestamp: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerGenerationInput:
    query: str
    results: List[EvidenceClip] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnswerGenerationInput":
        query = str(payload.get("query", "")).strip()
        raw_results = payload.get("results", []) or []
        clips: List[EvidenceClip] = []

        for row in raw_results:
            if not isinstance(row, dict):
                continue
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                continue

            entities = row.get("entities", []) or []
            if not isinstance(entities, list):
                entities = []

            timestamp = row.get("timestamp", {}) or {}
            if not isinstance(timestamp, dict):
                timestamp = {"value": str(timestamp)}

            clips.append(
                EvidenceClip(
                    clip_id=clip_id,
                    score=float(row.get("score", 0.0) or 0.0),
                    summary=str(row.get("summary", "") or ""),
                    ocr_text=str(row.get("ocr_text", "") or ""),
                    transcript=str(row.get("transcript", "") or ""),
                    entities=[str(e) for e in entities if str(e).strip()],
                    timestamp=timestamp,
                )
            )

        return cls(query=query, results=clips)


@dataclass
class _ContextClip:
    clip_id: str
    score: float
    tier: str
    summary: str
    ocr_text: str
    transcript: str
    entities: List[str]
    timestamp: Dict[str, Any]


class AnswerGenerator:
    """Gemini-powered grounded answer generator with strict context control."""

    def __init__(
        self,
        key_manager: Optional[GeminiKeyManager] = None,
        model_priority: Optional[Sequence[str]] = None,
        context_limit_tokens: int = MAX_CONTEXT_TOKENS,
        temperature: float = 0.25,
    ):
        self.key_manager = key_manager or GeminiKeyManager.from_env()
        self.model_priority = list(model_priority or MODEL_PRIORITY)
        self.context_limit_tokens = context_limit_tokens
        self.temperature = temperature

    def generate(self, payload: Dict[str, Any] | AnswerGenerationInput) -> Dict[str, Any]:
        if isinstance(payload, dict):
            payload_obj = AnswerGenerationInput.from_dict(payload)
        else:
            payload_obj = payload

        if not payload_obj.query.strip() or not payload_obj.results:
            return dict(DEFAULT_FALLBACK_RESPONSE)

        context_clips = self._prepare_context(payload_obj.results, payload_obj.query)
        if not context_clips:
            return dict(DEFAULT_FALLBACK_RESPONSE)

        context_budget = max(
            2_000,
            self.context_limit_tokens
            - RESPONSE_RESERVE_TOKENS
            - self._estimate_tokens(SYSTEM_MESSAGE)
            - self._estimate_tokens(payload_obj.query)
            - 1_200,
        )
        self._trim_to_budget(context_clips, context_budget)

        allowed_clip_ids = {c.clip_id for c in context_clips}
        allowed_entities = self._collect_allowed_entities(context_clips)
        user_prompt = self._build_user_prompt(payload_obj.query, context_clips)

        estimated_total = (
            self._estimate_tokens(SYSTEM_MESSAGE)
            + self._estimate_tokens(user_prompt)
            + RESPONSE_RESERVE_TOKENS
        )
        if estimated_total > self.context_limit_tokens:
            overflow = estimated_total - self.context_limit_tokens
            tighter_budget = max(1_200, context_budget - overflow - 500)
            self._trim_to_budget(context_clips, tighter_budget)
            user_prompt = self._build_user_prompt(payload_obj.query, context_clips)

            estimated_total = (
                self._estimate_tokens(SYSTEM_MESSAGE)
                + self._estimate_tokens(user_prompt)
                + RESPONSE_RESERVE_TOKENS
            )

        if estimated_total > self.context_limit_tokens:
            logger.error(
                "Context budget enforcement failed: estimated total tokens=%s exceeds limit=%s",
                estimated_total,
                self.context_limit_tokens,
            )
            return dict(DEFAULT_FALLBACK_RESPONSE)

        if self.key_manager.key_count == 0:
            logger.error("No Gemini API keys available for answer generation.")
            return dict(DEFAULT_FALLBACK_RESPONSE)

        for model_name in self.model_priority:
            model_result = self._run_model_tier(
                model_name=model_name,
                prompt=user_prompt,
                allowed_clip_ids=allowed_clip_ids,
                allowed_entities=allowed_entities,
                context_clips=context_clips,
                query=payload_obj.query,
            )
            if model_result is not None:
                return model_result

        logger.error("All Gemini model tiers exhausted. Returning fallback response.")
        return dict(DEFAULT_FALLBACK_RESPONSE)

    def _run_model_tier(
        self,
        model_name: str,
        prompt: str,
        allowed_clip_ids: Set[str],
        allowed_entities: Set[str],
        context_clips: List[_ContextClip],
        query: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.key_manager.has_healthy_keys():
            # Repeat key rotation per model tier even when prior tier cooldowns were applied.
            self.key_manager.reset_cooldowns()

        attempts = 0
        max_attempts = max(self.key_manager.key_count, 1)

        while attempts < max_attempts and self.key_manager.has_healthy_keys():
            key_obj = self.key_manager.get_next_key()
            attempts += 1
            if key_obj is None:
                break

            try:
                raw = self._invoke_gemini(api_key=key_obj.key, model_name=model_name, user_prompt=prompt)
                parsed = self._parse_json(raw)
                validated = self._validate_output(parsed, allowed_clip_ids, allowed_entities)

                if validated is None:
                    corrected = self._retry_with_correction(
                        api_key=key_obj.key,
                        model_name=model_name,
                        base_prompt=prompt,
                        raw_output=raw,
                        allowed_clip_ids=allowed_clip_ids,
                        allowed_entities=allowed_entities,
                    )
                    validated = corrected

                if validated is None:
                    logger.warning("Invalid model output on %s; rotating key.", model_name)
                    self.key_manager.mark_failure(key_obj, cooldown_seconds=30.0)
                    continue

                self.key_manager.mark_success(key_obj)
                validated["confidence"] = self._compute_confidence(
                    validated, query=query, context_clips=context_clips
                )
                return validated

            except Exception as exc:
                cooldown_seconds = 60.0
                if self._is_rate_limit_error(exc):
                    cooldown_seconds = 60.0
                self.key_manager.mark_failure(key_obj, cooldown_seconds=cooldown_seconds)
                logger.warning("Model call failed for %s: %s", model_name, exc)

        return None

    def _invoke_gemini(self, api_key: str, model_name: str, user_prompt: str) -> str:
        if genai is None or genai_types is None:
            raise RuntimeError(
                "google-genai package is not installed. Install with: pip install google-genai"
            )

        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=STRICT_OUTPUT_SCHEMA,
            system_instruction=SYSTEM_MESSAGE,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=config,
        )

        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if not parts:
                    continue
                chunks = []
                for part in parts:
                    value = getattr(part, "text", None)
                    if value:
                        chunks.append(value)
                merged = "\n".join(chunks).strip()
                if merged:
                    return merged

        raise ValueError("Gemini returned an empty response body")

    def _retry_with_correction(
        self,
        api_key: str,
        model_name: str,
        base_prompt: str,
        raw_output: str,
        allowed_clip_ids: Set[str],
        allowed_entities: Set[str],
    ) -> Optional[Dict[str, Any]]:
        correction_prompt = (
            base_prompt
            + "\n\n[CORRECTION REQUIRED]\n"
            + "Your previous output was invalid. Return ONLY a valid JSON object matching the schema.\n"
            + f"Allowed clip_ids: {sorted(allowed_clip_ids)}\n"
            + "Do not invent clip_ids or entities.\n"
            + "Previous invalid output:\n"
            + raw_output[:6_000]
        )

        try:
            corrected_raw = self._invoke_gemini(
                api_key=api_key,
                model_name=model_name,
                user_prompt=correction_prompt,
            )
            parsed = self._parse_json(corrected_raw)
            return self._validate_output(parsed, allowed_clip_ids, allowed_entities)
        except Exception as exc:
            logger.warning("Correction retry failed for %s: %s", model_name, exc)
            return None

    def _prepare_context(self, clips: List[EvidenceClip], query: str) -> List[_ContextClip]:
        ranked = sorted(clips, key=lambda c: c.score, reverse=True)
        total = len(ranked)

        high_n = min(5, total)
        mid_n = min(10, max(0, total - high_n))

        high = ranked[:high_n]
        mid = ranked[high_n: high_n + mid_n]
        low = ranked[high_n + mid_n:]

        context: List[_ContextClip] = []

        for clip in high:
            context.append(
                _ContextClip(
                    clip_id=clip.clip_id,
                    score=clip.score,
                    tier="high",
                    summary=self._truncate_to_tokens(clip.summary, 2_000),
                    ocr_text=self._truncate_to_tokens(clip.ocr_text, 2_500),
                    transcript=self._truncate_to_tokens(clip.transcript, 6_000),
                    entities=clip.entities[:25],
                    timestamp=clip.timestamp,
                )
            )

        for clip in mid:
            context.append(
                _ContextClip(
                    clip_id=clip.clip_id,
                    score=clip.score,
                    tier="mid",
                    summary=self._truncate_to_tokens(clip.summary, 1_000),
                    ocr_text=self._truncate_to_tokens(clip.ocr_text, 450),
                    transcript=self._truncate_to_tokens(clip.transcript, 450),
                    entities=clip.entities[:12],
                    timestamp=clip.timestamp,
                )
            )

        for clip in low:
            context.append(
                _ContextClip(
                    clip_id=clip.clip_id,
                    score=clip.score,
                    tier="low",
                    summary=self._truncate_to_tokens(clip.summary, 260),
                    ocr_text="",
                    transcript="",
                    entities=clip.entities[:8],
                    timestamp=clip.timestamp,
                )
            )

        fixed_prompt_tokens = self._estimate_tokens(SYSTEM_MESSAGE) + self._estimate_tokens(query) + 1_400
        budget = max(3_000, self.context_limit_tokens - RESPONSE_RESERVE_TOKENS - fixed_prompt_tokens)
        self._trim_to_budget(context, budget)
        return context

    def _trim_to_budget(self, context: List[_ContextClip], budget_tokens: int):
        if not context:
            return

        def total_tokens() -> int:
            return sum(
                self._estimate_tokens(c.summary)
                + self._estimate_tokens(c.ocr_text)
                + self._estimate_tokens(c.transcript)
                + self._estimate_tokens(" ".join(c.entities))
                for c in context
            )

        max_iterations = 240
        iteration = 0

        while total_tokens() > budget_tokens and iteration < max_iterations:
            iteration += 1

            if self._reduce_field(context, field_name="transcript", reduction_ratio=0.72):
                continue
            if self._reduce_field(context, field_name="ocr_text", reduction_ratio=0.72):
                continue
            if self._reduce_low_summaries(context, reduction_ratio=0.75):
                continue
            if self._drop_one_tier_clip(context, "low"):
                continue
            if self._drop_one_tier_clip(context, "mid"):
                continue
            if self._reduce_field(context, field_name="summary", reduction_ratio=0.8):
                continue
            break

    def _reduce_field(self, context: List[_ContextClip], field_name: str, reduction_ratio: float) -> bool:
        order = {"low": 0, "mid": 1, "high": 2}
        reduced = False

        for clip in sorted(
            context,
            key=lambda c: (order.get(c.tier, 99), -self._estimate_tokens(getattr(c, field_name, ""))),
        ):
            value = getattr(clip, field_name, "")
            tokens = self._estimate_tokens(value)
            if tokens <= 0:
                continue

            min_tokens = 0
            if field_name == "summary" and clip.tier == "high":
                min_tokens = 120

            new_tokens = max(min_tokens, int(tokens * reduction_ratio))
            if new_tokens >= tokens:
                new_tokens = tokens - 1
            if new_tokens < 0:
                new_tokens = 0

            setattr(clip, field_name, self._truncate_to_tokens(value, new_tokens))
            reduced = True
            if new_tokens > 0:
                break

        return reduced

    def _reduce_low_summaries(self, context: List[_ContextClip], reduction_ratio: float) -> bool:
        reduced = False
        for clip in context:
            if clip.tier != "low" or not clip.summary:
                continue
            tokens = self._estimate_tokens(clip.summary)
            if tokens <= 0:
                continue
            new_tokens = max(0, int(tokens * reduction_ratio))
            if new_tokens >= tokens:
                new_tokens = tokens - 1
            clip.summary = self._truncate_to_tokens(clip.summary, new_tokens)
            reduced = True
            break
        return reduced

    def _drop_one_tier_clip(self, context: List[_ContextClip], tier: str) -> bool:
        for idx in range(len(context) - 1, -1, -1):
            if context[idx].tier == tier:
                context.pop(idx)
                return True
        return False

    def _build_user_prompt(self, query: str, context_clips: List[_ContextClip]) -> str:
        high = [c for c in context_clips if c.tier == "high"]
        mid = [c for c in context_clips if c.tier == "mid"]
        low = [c for c in context_clips if c.tier == "low"]

        allowed_clip_ids = [c.clip_id for c in context_clips]

        lines: List[str] = []
        lines.append("1. Query")
        lines.append(f"User Query: {query}")
        lines.append("")

        lines.append("2. High Priority Evidence")
        lines.append("[HIGH PRIORITY CLIPS]")
        lines.extend(self._render_clip_block(high, include_all_modalities=True))
        lines.append("")

        lines.append("3. Mid Priority Evidence")
        lines.append("[ADDITIONAL CONTEXT]")
        lines.extend(self._render_clip_block(mid, include_all_modalities=False))

        if low:
            lines.append("")
            lines.append("4. Low Priority Evidence")
            lines.append("[LOW PRIORITY CONTEXT]")
            for clip in low:
                lines.append(f"Clip: {clip.clip_id}")
                lines.append(f"Score: {clip.score:.4f}")
                lines.append(f"Summary: {clip.summary or '[empty]'}")
                lines.append("---")

        lines.append("")
        lines.append("5. Instructions")
        lines.append("Answer using ONLY this data. Cite clip_ids. Handle temporal relationships correctly.")
        lines.append("If conflicting evidence exists, mention uncertainty.")
        lines.append("If evidence is insufficient, explicitly state that evidence is insufficient.")
        lines.append(f"Allowed clip_ids for citations: {allowed_clip_ids}")
        lines.append(
            "Return strict JSON only in this shape: "
            '{"answer":"...","citations":["clip_..."],"reasoning":"...","confidence":0.0}'
        )

        return "\n".join(lines)

    def _render_clip_block(self, clips: List[_ContextClip], include_all_modalities: bool) -> List[str]:
        lines: List[str] = []
        for clip in clips:
            lines.append(f"Clip: {clip.clip_id}")
            lines.append(f"Score: {clip.score:.4f}")
            if clip.timestamp:
                lines.append(f"Timestamp: {json.dumps(clip.timestamp, ensure_ascii=True)}")
            lines.append(f"Summary: {clip.summary or '[empty]'}")

            if include_all_modalities:
                lines.append(f"OCR: {clip.ocr_text or '[empty]'}")
                lines.append(f"Transcript: {clip.transcript or '[empty]'}")
                lines.append(f"Entities: {clip.entities or []}")
            else:
                if clip.ocr_text:
                    lines.append(f"OCR (truncated): {clip.ocr_text}")
                if clip.transcript:
                    lines.append(f"Transcript (truncated): {clip.transcript}")
                if clip.entities:
                    lines.append(f"Entities: {clip.entities}")

            lines.append("---")

        if not clips:
            lines.append("[none]")
        return lines

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()

        return json.loads(text)

    def _validate_output(
        self,
        parsed: Dict[str, Any],
        allowed_clip_ids: Set[str],
        allowed_entities: Set[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return None

        if not all(k in parsed for k in ("answer", "citations", "reasoning", "confidence")):
            return None

        answer = str(parsed.get("answer", "")).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()

        citations_raw = parsed.get("citations", [])
        if not isinstance(citations_raw, list):
            return None

        citations: List[str] = []
        for c in citations_raw:
            cid = str(c).strip()
            if cid and cid not in citations:
                citations.append(cid)

        if not set(citations).issubset(allowed_clip_ids):
            return None

        mentioned_clip_ids = set(_CLIP_ID_PATTERN.findall(f"{answer}\n{reasoning}"))
        if not mentioned_clip_ids.issubset(allowed_clip_ids):
            return None

        if allowed_entities:
            mentioned_entities = set(_ENTITY_ID_PATTERN.findall(f"{answer}\n{reasoning}"))
            if not mentioned_entities.issubset(allowed_entities):
                return None

        if not answer:
            answer = DEFAULT_FALLBACK_RESPONSE["answer"]
        if not reasoning:
            reasoning = DEFAULT_FALLBACK_RESPONSE["reasoning"]

        return {
            "answer": answer,
            "citations": citations,
            "reasoning": reasoning,
            "confidence": 0.0,
        }

    def _compute_confidence(
        self,
        output: Dict[str, Any],
        query: str,
        context_clips: List[_ContextClip],
    ) -> float:
        citations = output.get("citations", []) or []
        if not citations:
            return 0.0

        clip_map = {c.clip_id: c for c in context_clips}
        cited = [clip_map[c] for c in citations if c in clip_map]
        if not cited:
            return 0.0

        avg_clip_score = sum(c.score for c in cited) / len(cited)
        agreement = self._compute_clip_agreement(cited)
        query_coverage = self._compute_query_coverage(query, cited)

        confidence = 0.5 * avg_clip_score + 0.3 * agreement + 0.2 * query_coverage
        confidence = max(0.0, min(1.0, confidence))
        return round(confidence, 4)

    def _compute_clip_agreement(self, clips: List[_ContextClip]) -> float:
        if len(clips) == 1:
            return 0.75

        token_sets: List[Set[str]] = []
        for clip in clips:
            text = " ".join([clip.summary, clip.transcript, clip.ocr_text])
            token_sets.append(self._tokenize(text))

        sims: List[float] = []
        for left, right in combinations(token_sets, 2):
            if not left and not right:
                sims.append(0.0)
                continue
            union = left | right
            sims.append(len(left & right) / max(1, len(union)))

        if not sims:
            return 0.0
        return max(0.0, min(1.0, sum(sims) / len(sims)))

    def _compute_query_coverage(self, query: str, clips: List[_ContextClip]) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        evidence_tokens: Set[str] = set()
        for clip in clips:
            evidence_tokens |= self._tokenize(" ".join([clip.summary, clip.transcript, clip.ocr_text]))

        if not evidence_tokens:
            return 0.0

        covered = query_tokens & evidence_tokens
        return max(0.0, min(1.0, len(covered) / len(query_tokens)))

    def _collect_allowed_entities(self, context_clips: List[_ContextClip]) -> Set[str]:
        allowed: Set[str] = set()
        for clip in context_clips:
            for entity in clip.entities:
                entity_value = str(entity).strip()
                if entity_value:
                    allowed.add(entity_value)
        return allowed

    def _tokenize(self, text: str) -> Set[str]:
        words = _WORD_PATTERN.findall((text or "").lower())
        return {w for w in words if len(w) > 2 and w not in _STOPWORDS}

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""

        approx_chars = max_tokens * 4
        if len(text) <= approx_chars:
            return text

        return text[:approx_chars].rstrip() + " ...[truncated]"

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "rate limit" in msg or "429" in msg or "quota" in msg
