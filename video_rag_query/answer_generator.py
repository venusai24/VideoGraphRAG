import json
import logging
import mimetypes
import os
import re
import time
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

DEFAULT_MODEL_PRIORITY = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
]

SYSTEM_MESSAGE = (
    "You are a grounded video reasoning system.\n\n"
    "Use ONLY the provided evidence.\n"
    "Do NOT hallucinate.\n"
    "If evidence is insufficient, say so.\n"
    "Cite only the provided clip_ids.\n"
    "Do not make unsupported temporal inferences."
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

MAX_CONTEXT_TOKENS = 200_000
MAX_TEXT_TOKENS = 160_000  # HARD context budget for text content
RESPONSE_RESERVE_TOKENS = 4_000
MAX_MEDIA_ATTACHMENTS = 2  # DISABLED: media uploads add 30-200s latency; set to 1 to re-enable
VIDEO_UPLOAD_POLL_SECONDS = 3
VIDEO_UPLOAD_TIMEOUT_SECONDS = 10  # HARD: if media > 10s, skip to text-only
GENERATION_LATENCY_LIMIT = 15.0  # seconds — violations logged

_ENTITY_ID_PATTERN = re.compile(r"\b(?:person|location|topic|object|event)_[A-Za-z0-9_-]+\b")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_POTENTIAL_CLIP_REF_PATTERN = re.compile(
    r"\b(?:clip_[A-Za-z0-9_-]+|[A-Za-z0-9-]+_\d+(?:\.\d+)?_\d+(?:\.\d+)?)\b"
)

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
    clip_path: str = ""
    rank: int = 0


@dataclass
class AnswerGenerationInput:
    query: str
    results: List[EvidenceClip] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnswerGenerationInput":
        query = str(payload.get("query", "")).strip()
        raw_results = payload.get("results", []) or []
        clips: List[EvidenceClip] = []
        required_fields = {
            "clip_id",
            "score",
            "summary",
            "ocr_text",
            "transcript",
            "entities",
            "timestamp",
            "clip_path",
            "rank",
        }

        for idx, row in enumerate(raw_results):
            if not isinstance(row, dict):
                raise ValueError(f"Answer-generation result at index {idx} is not a dict.")

            missing = sorted(required_fields - set(row.keys()))
            if missing:
                raise ValueError(
                    f"Answer-generation result for index {idx} is missing required fields: {missing}"
                )

            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                raise ValueError(f"Answer-generation result at index {idx} is missing clip_id.")

            entities = row.get("entities", []) or []
            if not isinstance(entities, list):
                entities = []

            timestamp = row.get("timestamp", {}) or {}
            if not isinstance(timestamp, dict):
                timestamp = {"value": str(timestamp)}

            rank = int(row.get("rank", 0) or 0)
            if rank <= 0:
                raise ValueError(f"Answer-generation result for clip {clip_id} has invalid rank={rank}.")

            clips.append(
                EvidenceClip(
                    clip_id=clip_id,
                    score=float(row.get("score", 0.0) or 0.0),
                    summary=str(row.get("summary", "") or ""),
                    ocr_text=str(row.get("ocr_text", "") or ""),
                    transcript=str(row.get("transcript", "") or ""),
                    entities=[str(e) for e in entities if str(e).strip()],
                    timestamp=timestamp,
                    clip_path=str(row.get("clip_path", "") or ""),
                    rank=rank,
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
    clip_path: str
    rank: int
    presentation_slot: int = 0
    media_eligible: bool = False


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
        self.model_priority = list(model_priority or self._load_model_priority())
        self.context_limit_tokens = context_limit_tokens
        self.temperature = temperature
        self._last_error: Optional[str] = None

    def generate(self, payload: Dict[str, Any] | AnswerGenerationInput) -> Dict[str, Any]:
        if isinstance(payload, dict):
            payload_obj = AnswerGenerationInput.from_dict(payload)
        else:
            payload_obj = payload

        if not payload_obj.query.strip() or not payload_obj.results:
            return dict(DEFAULT_FALLBACK_RESPONSE)

        self._last_error = None
        self._gen_logs = {"provider_attempts": [], "retry_count": 0, "media_used": False, "token_estimate": 0}
        context_clips = self._prepare_context(payload_obj.results)
        if not context_clips:
            return dict(DEFAULT_FALLBACK_RESPONSE)

        context_budget = max(
            2_000,
            min(MAX_TEXT_TOKENS, self.context_limit_tokens)
            - RESPONSE_RESERVE_TOKENS
            - self._estimate_tokens(SYSTEM_MESSAGE)
            - self._estimate_tokens(payload_obj.query)
            - 1_200,
        )
        self._trim_to_budget(context_clips, context_budget)
        self._apply_prompt_metadata(context_clips)
        self._gen_logs["token_estimate"] = sum(
            self._estimate_tokens(c.summary) + self._estimate_tokens(c.ocr_text)
            + self._estimate_tokens(c.transcript) for c in context_clips
        )

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
            self._apply_prompt_metadata(context_clips)
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

        logger.error("All Gemini model tiers exhausted. Last error: %s", self._last_error or "unknown")
        result = dict(DEFAULT_FALLBACK_RESPONSE)
        result["_gen_logs"] = self._gen_logs
        return result

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
            self.key_manager.reset_cooldowns()

        attempts = 0
        max_attempts = max(self.key_manager.key_count, 1)

        while attempts < max_attempts and self.key_manager.has_healthy_keys():
            key_obj = self.key_manager.get_next_key()
            attempts += 1
            if key_obj is None:
                break

            attempt_log = {
                "provider": "gemini",
                "model": model_name,
                "status": "failed",
                "error": None,
                "latency": 0.0,
                "media_used": False,
            }
            gen_start = time.time()

            try:
                raw = self._invoke_gemini(
                    api_key=key_obj.key,
                    model_name=model_name,
                    user_prompt=prompt,
                    context_clips=context_clips,
                )
                attempt_log["media_used"] = self._gen_logs.get("media_used", False)
                parsed = self._parse_json(raw)
                validated = self._validate_output(parsed, allowed_clip_ids, allowed_entities)

                if validated is None:
                    self._gen_logs["retry_count"] += 1
                    corrected = self._retry_with_correction(
                        api_key=key_obj.key,
                        model_name=model_name,
                        base_prompt=prompt,
                        raw_output=raw,
                        allowed_clip_ids=allowed_clip_ids,
                        allowed_entities=allowed_entities,
                        context_clips=context_clips,
                    )
                    validated = corrected

                if validated is None:
                    self._last_error = f"Invalid JSON output after correction retry on model={model_name}"
                    logger.warning("Invalid model output on %s; rotating key.", model_name)
                    attempt_log["error"] = self._last_error
                    attempt_log["latency"] = time.time() - gen_start
                    self._gen_logs["provider_attempts"].append(attempt_log)
                    self.key_manager.mark_failure(key_obj, cooldown_seconds=30.0)
                    continue

                self.key_manager.mark_success(key_obj)
                validated["confidence"] = self._compute_confidence(
                    validated, query=query, context_clips=context_clips
                )
                attempt_log["status"] = "success"
                attempt_log["latency"] = time.time() - gen_start
                self._gen_logs["provider_attempts"].append(attempt_log)

                # Timing guard
                if attempt_log["latency"] > GENERATION_LATENCY_LIMIT:
                    logger.warning(
                        "LATENCY_VIOLATION: generation took %.2fs > %.1fs limit (model=%s)",
                        attempt_log["latency"], GENERATION_LATENCY_LIMIT, model_name,
                    )
                    validated["_latency_violation"] = True

                validated["_gen_logs"] = self._gen_logs
                return validated

            except Exception as exc:
                self._last_error = f"{model_name}: {exc}"
                cooldown_seconds = 60.0
                if self._is_rate_limit_error(exc):
                    cooldown_seconds = 60.0
                self.key_manager.mark_failure(key_obj, cooldown_seconds=cooldown_seconds)
                attempt_log["error"] = str(exc)[:200]
                attempt_log["latency"] = time.time() - gen_start
                self._gen_logs["provider_attempts"].append(attempt_log)
                logger.warning("Model call failed for %s: %s", model_name, exc)

        return None

    def _invoke_gemini(
        self,
        api_key: str,
        model_name: str,
        user_prompt: str,
        context_clips: List[_ContextClip],
    ) -> str:
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

        contents: List[Any] = []
        for clip in self._select_media_candidates(context_clips):
            try:
                uploaded = self._upload_media_file(client, clip)
                contents.append(
                    f"Attached media for Rank #{clip.rank} | Clip ID {clip.clip_id} | "
                    f"basename={os.path.basename(clip.clip_path)}"
                )
                contents.append(uploaded)
            except Exception as exc:
                logger.warning(
                    "Degrading clip %s to text-only because media attachment failed: %s",
                    clip.clip_id,
                    exc,
                )

        contents.append(user_prompt)
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
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
        context_clips: List[_ContextClip],
    ) -> Optional[Dict[str, Any]]:
        correction_prompt = (
            base_prompt
            + "\n\n[CORRECTION REQUIRED]\n"
            + "Your previous output was invalid.\n"
            + "Return ONLY a valid JSON object matching the required schema.\n"
            + f"Allowed clip_ids: {sorted(allowed_clip_ids)}\n"
            + "Do not invent clip_ids or entities.\n"
            + "If evidence is weak, keep the answer conservative but still return valid JSON.\n"
            + "Previous invalid output:\n"
            + raw_output[:6_000]
        )

        try:
            corrected_raw = self._invoke_gemini(
                api_key=api_key,
                model_name=model_name,
                user_prompt=correction_prompt,
                context_clips=context_clips,
            )
            parsed = self._parse_json(corrected_raw)
            return self._validate_output(parsed, allowed_clip_ids, allowed_entities)
        except Exception as exc:
            logger.warning("Correction retry failed for %s: %s", model_name, exc)
            return None

    def _prepare_context(self, clips: List[EvidenceClip]) -> List[_ContextClip]:
        ranked = sorted(clips, key=lambda c: (c.rank, -c.score, c.clip_id))
        context: List[_ContextClip] = []

        # STRICT tier policy
        tier1_count = 0
        tier2_count = 0

        for clip in ranked:
            if clip.rank <= 2 and tier1_count < 2:
                tier = "high"
                tier1_count += 1
                summary_tokens = 2_000
                ocr_tokens = 2_500
                transcript_tokens = 6_000
                entities = clip.entities[:25]
            elif clip.rank <= 5 and tier2_count < 3:
                tier = "mid"
                tier2_count += 1
                summary_tokens = 200  # ≤200 chars
                ocr_tokens = 200
                transcript_tokens = 200  # trimmed transcript ≤200 chars
                entities = clip.entities[:12]
            else:
                tier = "low"
                summary_tokens = 100  # summaries ONLY ≤100 chars
                ocr_tokens = 0
                transcript_tokens = 0
                entities = clip.entities[:8]

            context.append(
                _ContextClip(
                    clip_id=clip.clip_id,
                    score=clip.score,
                    tier=tier,
                    summary=self._truncate_to_tokens(clip.summary, summary_tokens),
                    ocr_text=self._truncate_to_tokens(clip.ocr_text, ocr_tokens),
                    transcript=self._truncate_to_tokens(clip.transcript, transcript_tokens),
                    entities=entities,
                    timestamp=clip.timestamp,
                    clip_path=clip.clip_path,
                    rank=clip.rank,
                )
            )

        self._apply_prompt_metadata(context)
        return context

    def _apply_prompt_metadata(self, context: List[_ContextClip]):
        if not context:
            return

        for clip in context:
            clip.presentation_slot = 0
            clip.media_eligible = False

        for tier_name in ("high", "mid", "low"):
            tier_clips = [c for c in context if c.tier == tier_name]
            ordered = self._presentation_order(tier_clips)
            for slot, clip in enumerate(ordered, start=1):
                clip.presentation_slot = slot

        media_candidates = [c for c in sorted(context, key=lambda clip: clip.rank) if c.tier == "high"]
        attached = 0
        for clip in media_candidates:
            if attached >= MAX_MEDIA_ATTACHMENTS:
                break
            if self._is_media_readable(clip.clip_path):
                clip.media_eligible = True
                attached += 1

    def _presentation_order(self, clips: List[_ContextClip]) -> List[_ContextClip]:
        ranked = sorted(clips, key=lambda clip: clip.rank)
        if len(ranked) <= 1:
            return ranked

        split_idx = max(1, len(ranked) // 2)
        front_half = ranked[:split_idx]
        back_half = ranked[split_idx:]
        return front_half + list(reversed(back_half))

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

        for clip in sorted(
            context,
            key=lambda c: (order.get(c.tier, 99), -self._estimate_tokens(getattr(c, field_name, "")), -c.rank),
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
            return True

        return False

    def _reduce_low_summaries(self, context: List[_ContextClip], reduction_ratio: float) -> bool:
        for clip in sorted([c for c in context if c.tier == "low"], key=lambda c: c.rank, reverse=True):
            if not clip.summary:
                continue
            tokens = self._estimate_tokens(clip.summary)
            if tokens <= 0:
                continue
            new_tokens = max(0, int(tokens * reduction_ratio))
            if new_tokens >= tokens:
                new_tokens = tokens - 1
            clip.summary = self._truncate_to_tokens(clip.summary, new_tokens)
            return True
        return False

    def _drop_one_tier_clip(self, context: List[_ContextClip], tier: str) -> bool:
        candidates = [c for c in context if c.tier == tier]
        if not candidates:
            return False
        worst = max(candidates, key=lambda clip: clip.rank)
        context.remove(worst)
        return True

    def _build_user_prompt(self, query: str, context_clips: List[_ContextClip]) -> str:
        high = self._presentation_order([c for c in context_clips if c.tier == "high"])
        mid = self._presentation_order([c for c in context_clips if c.tier == "mid"])
        low = self._presentation_order([c for c in context_clips if c.tier == "low"])

        allowed_clip_ids = [c.clip_id for c in sorted(context_clips, key=lambda clip: clip.rank)]
        rank_map = [f"Rank #{clip.rank} -> {clip.clip_id}" for clip in sorted(context_clips, key=lambda clip: clip.rank)]
        recap = [clip for clip in sorted(context_clips, key=lambda clip: clip.rank) if clip.rank <= 3]

        lines: List[str] = []
        lines.append("1. Query")
        lines.append(f"User Query: {query}")
        lines.append("")

        lines.append("2. Allowed clip_ids and canonical rank map")
        lines.append("Canonical rank numbers encode retrieval priority and do not change even if prompt placement differs.")
        lines.append("Rank #1 is the highest-relevance clip.")
        lines.extend(rank_map or ["[none]"])
        lines.append("")

        lines.append("3. Tier 1 detailed evidence")
        lines.append("Presentation order uses strong-edge / weak-middle layout to reduce lost-in-the-middle effects.")
        lines.append("[HIGH PRIORITY CLIPS]")
        lines.extend(self._render_clip_block(high, include_all_modalities=True))
        lines.append("")

        lines.append("4. Tier 2 compressed evidence")
        lines.append("[ADDITIONAL CONTEXT]")
        lines.extend(self._render_clip_block(mid, include_all_modalities=False))

        if low:
            lines.append("")
            lines.append("5. Tier 3 minimal evidence")
            lines.append("[LOW PRIORITY CONTEXT]")
            lines.extend(self._render_clip_block(low, include_all_modalities=False, summary_only=True))

        lines.append("")
        lines.append("6. High-priority recap")
        if recap:
            lines.append("Highest-priority clips: Rank #1, Rank #2, Rank #3")
            for clip in recap:
                lines.append(
                    f"Rank #{clip.rank} | Clip ID {clip.clip_id} | Summary: {self._truncate_to_tokens(clip.summary, 80) or '[empty]'}"
                )
        else:
            lines.append("[none]")

        lines.append("")
        lines.append("7. Reasoning and output rules")
        lines.append("Use ONLY the provided evidence.")
        lines.append("Lower rank numbers are higher priority. Prefer them when evidence conflicts.")
        lines.append("Higher rank numbers may support or disambiguate, but should not override stronger evidence without explicit justification.")
        lines.append("If clips conflict, mention the conflict and lower certainty.")
        lines.append("Do not infer before/after unless timestamps or text support it.")
        lines.append("Every factual claim must be attributable to one or more provided clip_ids.")
        lines.append(f"Allowed clip_ids for citations: {allowed_clip_ids}")
        lines.append(
            "Return strict JSON only in this shape: "
            '{"answer":"...","citations":["clip_id"],"reasoning":"...","confidence":0.0}'
        )

        return "\n".join(lines)

    def _render_clip_block(
        self,
        clips: List[_ContextClip],
        include_all_modalities: bool,
        summary_only: bool = False,
    ) -> List[str]:
        lines: List[str] = []
        for clip in clips:
            lines.append(
                f"Rank #{clip.rank} | Clip ID: {clip.clip_id} | Score: {clip.score:.4f} | "
                f"Tier: {clip.tier} | Media attached: {'yes' if clip.media_eligible else 'no'}"
            )
            if clip.timestamp:
                lines.append(f"Timestamp: {json.dumps(clip.timestamp, ensure_ascii=True)}")
            lines.append(f"Summary: {clip.summary or '[empty]'}")

            if not summary_only:
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

    def _select_media_candidates(self, context_clips: List[_ContextClip]) -> List[_ContextClip]:
        selected = [clip for clip in sorted(context_clips, key=lambda clip: clip.rank) if clip.media_eligible]
        return selected[:MAX_MEDIA_ATTACHMENTS]

    def _upload_media_file(self, client: Any, clip: _ContextClip) -> Any:
        mime_type = mimetypes.guess_type(clip.clip_path)[0] or "application/octet-stream"
        upload_start = time.time()
        uploaded = client.files.upload(file=clip.clip_path)

        if mime_type.startswith("video/"):
            deadline = time.time() + VIDEO_UPLOAD_TIMEOUT_SECONDS
            current = uploaded
            while getattr(current, "state", None) is None or getattr(current.state, "name", None) != "ACTIVE":
                state_name = getattr(getattr(current, "state", None), "name", None)
                if state_name in {"FAILED", "CANCELLED"}:
                    raise RuntimeError(f"Uploaded video file entered terminal state {state_name}")
                if time.time() >= deadline:
                    elapsed = time.time() - upload_start
                    logger.warning("Media upload timeout after %.1fs — skipping to text-only", elapsed)
                    raise TimeoutError(f"Media upload exceeded {VIDEO_UPLOAD_TIMEOUT_SECONDS}s limit")
                time.sleep(VIDEO_UPLOAD_POLL_SECONDS)
                current = client.files.get(name=current.name)
            uploaded = current

        self._gen_logs["media_used"] = True
        return uploaded

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        return json.loads((raw or "").strip())

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

        mentioned_clip_ids = set(_POTENTIAL_CLIP_REF_PATTERN.findall(f"{answer}\n{reasoning}"))
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

    def _load_model_priority(self) -> List[str]:
        configured = [
            os.getenv("GEMINI_MODEL_PRIMARY", "").strip(),
            os.getenv("GEMINI_MODEL_FALLBACK_1", "").strip(),
            os.getenv("GEMINI_MODEL_FALLBACK_2", "").strip(),
        ]
        models = [model for model in configured if model]
        return models or list(DEFAULT_MODEL_PRIORITY)

    def _is_media_readable(self, clip_path: str) -> bool:
        return bool(clip_path and os.path.isfile(clip_path) and os.access(clip_path, os.R_OK))

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
