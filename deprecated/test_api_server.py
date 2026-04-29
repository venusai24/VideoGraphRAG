# Deprecated – not used in final pipeline
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote


class QueryApiServerTests(unittest.TestCase):
    def test_graph_api_import_exposes_multigraph_manager(self):
        import video_rag_query.graph_api as graph_api

        self.assertTrue(hasattr(graph_api, "MultiGraphManager"))

    def test_build_clip_url_returns_none_for_paths_outside_media_root(self):
        from video_rag_query.api_server import build_clip_url

        with tempfile.TemporaryDirectory() as tmpdir:
            media_root = Path(tmpdir) / "outputs"
            media_root.mkdir()
            outside_file = Path(tmpdir) / "elsewhere" / "clip.mp4"
            outside_file.parent.mkdir()
            outside_file.write_bytes(b"video")

            clip_url = build_clip_url(
                clip_path=str(outside_file),
                media_root=media_root,
                base_url="http://localhost:8000/",
            )

        self.assertIsNone(clip_url)

    def test_build_clip_url_returns_media_path_for_valid_clip(self):
        from video_rag_query.api_server import build_clip_url

        with tempfile.TemporaryDirectory() as tmpdir:
            media_root = Path(tmpdir) / "outputs"
            clip_file = media_root / "abc123" / "clips" / "clip_0001.mp4"
            clip_file.parent.mkdir(parents=True)
            clip_file.write_bytes(b"video")

            clip_url = build_clip_url(
                clip_path=str(clip_file),
                media_root=media_root,
                base_url="http://localhost:8000/",
            )

        self.assertEqual(
            clip_url,
            f"http://localhost:8000/media?clip_path={quote(str(clip_file.resolve()), safe='')}",
        )

    def test_retrieve_endpoint_returns_pipeline_payload_with_strict_shape(self):
        from video_rag_query.api_server import QueryRequest, query_retrieve

        payload = {
            "query": "find the airport on a swamp",
            "timings": {
                "decomposition": 101.2,
                "traversal": 222.0,
                "ranking": 18.8,
                "total": 342.0,
            },
            "results": [
                {
                    "clip_id": "clip-1",
                    "score": 0.91,
                    "summary": "Airport construction clip",
                    "timestamp": {"start": 10.0, "end": 18.0, "video_id": "abc123"},
                    "entities": ["airport", "swamp"],
                    "clip_path": "/tmp/outputs/abc123/clips/clip_0001.mp4",
                    "clip_url": "http://localhost:8000/media/abc123/clips/clip_0001.mp4",
                    "rank": 1,
                    "transcript": "Airport construction clip transcript",
                    "ocr_text": "",
                    "explanation": "Matched airport and swamp references",
                }
            ],
        }

        with patch("video_rag_query.api_server.execute_retrieval_pipeline", return_value=payload):
            response = asyncio.run(query_retrieve(QueryRequest(query=payload["query"], top_k=10)))

        self.assertEqual(set(response.keys()), {"query", "timings", "results"})
        self.assertEqual(response["query"], payload["query"])
        self.assertEqual(response["results"][0]["clip_id"], "clip-1")
        self.assertEqual(set(response["timings"].keys()), {"decomposition", "traversal", "ranking", "total"})
        self.assertEqual(
            set(response["results"][0].keys()),
            {"clip_id", "score", "summary", "timestamp", "entities", "clip_path"},
        )

    def test_answer_endpoint_returns_structured_answer_on_success(self):
        from video_rag_query.api_server import QueryRequest, query_answer

        retrieval_payload = {
            "query": "Which airport was built on a swamp?",
            "timings": {
                "decomposition": 101.2,
                "traversal": 222.0,
                "ranking": 18.8,
                "total": 342.0,
            },
            "results": [
                {
                    "clip_id": "clip-1",
                    "score": 0.91,
                    "summary": "Airport construction clip",
                    "timestamp": {"start": 10.0, "end": 18.0, "video_id": "abc123"},
                    "entities": ["airport", "swamp"],
                    "clip_path": "/tmp/outputs/abc123/clips/clip_0001.mp4",
                    "clip_url": "http://localhost:8000/media/abc123/clips/clip_0001.mp4",
                    "transcript": "The airport is being built on swamp land.",
                    "ocr_text": "",
                    "rank": 1,
                }
            ],
        }
        answer_payload = {
            "answer": "The airport was built on swamp land.",
            "citations": ["clip-1"],
            "reasoning": "The retrieved clip states that directly.",
            "confidence": 0.88,
        }

        with patch("video_rag_query.api_server.execute_retrieval_pipeline", return_value=retrieval_payload), patch(
            "video_rag_query.api_server.generate_answer_payload", return_value=answer_payload
        ):
            response = asyncio.run(query_answer(QueryRequest(query=retrieval_payload["query"], top_k=10)))

        self.assertEqual(set(response.keys()), {"query", "timings", "results", "answer"})
        self.assertEqual(response["answer"], answer_payload)
        self.assertEqual(set(response["timings"].keys()), {"decomposition", "traversal", "ranking", "total"})
        self.assertEqual(
            set(response["results"][0].keys()),
            {"clip_id", "score", "summary", "timestamp", "entities", "clip_path"},
        )

    def test_answer_endpoint_raises_http_error_with_retrieval_payload_on_generation_failure(self):
        from fastapi import HTTPException

        from video_rag_query.api_server import QueryRequest, query_answer

        retrieval_payload = {
            "query": "Which airport was built on a swamp?",
            "timings": {
                "decomposition": 101.2,
                "traversal": 222.0,
                "ranking": 18.8,
                "total": 342.0,
            },
            "results": [
                {
                    "clip_id": "clip-1",
                    "score": 0.91,
                    "summary": "Airport construction clip",
                    "timestamp": {"start": 10.0, "end": 18.0, "video_id": "abc123"},
                    "entities": ["airport", "swamp"],
                    "clip_path": "/tmp/outputs/abc123/clips/clip_0001.mp4",
                    "clip_url": "http://localhost:8000/media/abc123/clips/clip_0001.mp4",
                    "transcript": "The airport is being built on swamp land.",
                    "ocr_text": "",
                    "rank": 1,
                }
            ],
        }

        with patch("video_rag_query.api_server.execute_retrieval_pipeline", return_value=retrieval_payload), patch(
            "video_rag_query.api_server.generate_answer_payload", side_effect=RuntimeError("gemini unavailable")
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(query_answer(QueryRequest(query=retrieval_payload["query"], top_k=10)))

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(ctx.exception.detail["stage"], "generation")
        self.assertEqual(
            ctx.exception.detail["results"],
            [
                {
                    "clip_id": "clip-1",
                    "score": 0.91,
                    "summary": "Airport construction clip",
                    "timestamp": {"start": 10.0, "end": 18.0, "video_id": "abc123"},
                    "entities": ["airport", "swamp"],
                    "clip_path": "/tmp/outputs/abc123/clips/clip_0001.mp4",
                }
            ],
        )
        self.assertEqual(set(ctx.exception.detail["timings"].keys()), {"decomposition", "traversal", "ranking", "total"})


if __name__ == "__main__":
    unittest.main()
