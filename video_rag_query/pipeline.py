from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from .answer_generator import AnswerGenerator
from .models import FailureResponse

if TYPE_CHECKING:
    from .graph_api import GraphAPI
    from .query_decomposer import QueryDecomposer
    from .traversal import TraversalExecutor

logger = logging.getLogger(__name__)


class QueryAnswerPipeline:
    """
    End-to-end query flow for VideoGraphRAG:
    decompose -> traversal -> evidence normalization -> answer generation.
    """

    def __init__(
        self,
        decomposer: "QueryDecomposer",
        traversal_executor: "TraversalExecutor",
        answer_generator: AnswerGenerator,
        graph_api: Optional["GraphAPI"] = None,
    ):
        self.decomposer = decomposer
        self.traversal_executor = traversal_executor
        self.answer_generator = answer_generator
        self.graph_api = graph_api or getattr(traversal_executor, "api", None)

    def run(self, query: str) -> Dict[str, Any]:
        decomposition = self.decomposer.decompose(query)
        if isinstance(decomposition, FailureResponse):
            logger.warning("Query decomposition failed: %s", decomposition.reason)
            return self.answer_generator.fallback_response(
                "No reliable supporting clips found."
            )

        traversal_results = self.traversal_executor.execute(decomposition)
        enriched_results = self._enrich_results(traversal_results)

        normalized = self.answer_generator.normalize_evidence(enriched_results)
        return self.answer_generator.generate(
            query,
            normalized,
            already_normalized=True,
        )

    def _enrich_results(self, results: Sequence[Any]) -> List[Dict[str, Any]]:
        items = [self._to_result_dict(result) for result in results]
        if not items:
            return []

        clip_ids = [item["clip_id"] for item in items if item.get("clip_id")]
        props_by_id: Dict[str, Dict[str, Any]] = {}

        if self.graph_api is not None and clip_ids:
            try:
                props_by_id = self.graph_api.get_node_properties(clip_ids)
            except Exception as exc:
                logger.warning("Failed to enrich traversal results from GraphAPI: %s", exc)

        enriched: List[Dict[str, Any]] = []
        for item in items:
            clip_id = item.get("clip_id", "")
            props = props_by_id.get(clip_id, {})

            summary = item.get("summary") or props.get("summary") or item.get("explanation", "")

            timestamp = item.get("timestamp")
            if timestamp is None:
                start = props.get("start")
                end = props.get("end")
                if start is not None or end is not None:
                    timestamp = {"start": start, "end": end}

            enriched.append(
                {
                    "clip_id": clip_id,
                    "score": item.get("score", 0.0),
                    "summary": summary,
                    "timestamp": timestamp,
                    "entities": item.get("entities", []),
                    "path": item.get("path", []),
                }
            )

        return enriched

    def _to_result_dict(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return {
                "clip_id": result.get("clip_id", ""),
                "score": result.get("score", 0.0),
                "summary": result.get("summary", ""),
                "timestamp": result.get("timestamp"),
                "entities": result.get("entities", []),
                "path": result.get("path", []),
                "explanation": result.get("explanation", ""),
            }

        as_dict = result.to_dict() if hasattr(result, "to_dict") else {}
        return {
            "clip_id": as_dict.get("clip_id", getattr(result, "clip_id", "")),
            "score": as_dict.get("score", getattr(result, "score", 0.0)),
            "summary": as_dict.get("summary", getattr(result, "summary", "")),
            "timestamp": as_dict.get("timestamp", getattr(result, "timestamp", None)),
            "entities": as_dict.get("entities", getattr(result, "entities", [])),
            "path": as_dict.get("path", getattr(result, "path", [])),
            "explanation": as_dict.get("explanation", getattr(result, "explanation", "")),
        }
