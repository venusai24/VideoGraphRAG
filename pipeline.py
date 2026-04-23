import os
import json

from normalization.evidence_builder import parse_transcript, parse_ocr, parse_keywords
from clip_builder.clip_builder import build_clips
from semantic.entity_extractor import extract_entities
import semantic.canonicalizer as canonicalizer
from graph.cross_layer import link_entities_to_clips
from graph.clip_graph import build_temporal_edges, build_similarity_edges
from semantic.relation_graph import build_cooccurrence_edges, build_similarity_edges as build_semantic_similarity
from graph.graph_builder import build_graph
from retrieval.engine import query_graph


def canonicalize_entities(entity_map):
    for fn_name in ("canonicalize_entities", "canonicalize", "build_canonical_entities"):
        fn = getattr(canonicalizer, fn_name, None)
        if callable(fn):
            return fn(entity_map)

    # Fallback: class-based API
    canonicalizer_cls = getattr(canonicalizer, "EntityCanonicalizer", None)
    if canonicalizer_cls is not None:
        instance = canonicalizer_cls()
        process_fn = getattr(instance, "process", None)
        if callable(process_fn):
            return process_fn(entity_map)

    raise ImportError(
        "No supported canonicalizer interface found in semantic.canonicalizer. "
        "Expected function API (canonicalize_entities/canonicalize/build_canonical_entities) "
        "or class API (EntityCanonicalizer.process)."
    )


def load_clip_folder(path, video_id):
    transcript = parse_transcript(f"{path}/transcript.json", video_id)
    ocr = parse_ocr(f"{path}/ocr.json", video_id)
    keywords = parse_keywords(f"{path}/keywords.json", video_id)

    return transcript + ocr + keywords


def run_pipeline(root_dir="outputs"):
    all_evidence = []
    all_clips = []

    # 🔹 STEP 1–2: Load + build clips
    for folder in sorted(os.listdir(root_dir)):
        clip_path = os.path.join(root_dir, folder)

        if not os.path.isdir(clip_path):
            continue

        video_id = folder
        evidence = load_clip_folder(clip_path, video_id)

        with open(os.path.join(clip_path, "rag_chunks.json"), "r", encoding="utf-8") as f:
            rag_chunks = json.load(f)

        clips = build_clips(rag_chunks, evidence)

        all_evidence.extend(evidence)
        all_clips.extend(clips)

    print(f"[INFO] Total clips: {len(all_clips)}")

    # 🔹 STEP 4–5: Entities
    entity_map = extract_entities(all_clips)
    semantic_nodes, clip_to_concepts = canonicalize_entities(entity_map)

    print(f"[INFO] Semantic nodes: {len(semantic_nodes)}")

    # 🔹 STEP 6: Cross-layer
    mention_edges = link_entities_to_clips(all_clips, semantic_nodes, clip_to_concepts)

    # 🔹 STEP 7: Clip graph
    temporal_edges = build_temporal_edges(all_clips)
    similarity_edges = build_similarity_edges(all_clips)

    clip_edges = temporal_edges + similarity_edges

    # 🔹 STEP 8: Semantic graph
    co_edges = build_cooccurrence_edges(clip_to_concepts)
    sem_sim_edges = build_semantic_similarity(semantic_nodes)

    semantic_edges = co_edges + sem_sim_edges

    # 🔹 STEP 9: Assemble graph
    graph = build_graph(
        all_clips,
        semantic_nodes,
        clip_edges,
        mention_edges,
        semantic_edges
    )

    print(f"[INFO] Graph nodes: {len(graph.nodes)}")
    print(f"[INFO] Graph edges: {len(graph.edges)}")

    return graph


if __name__ == "__main__":
    graph = run_pipeline("outputs")

    # 🔹 STEP 10: Query test
    queries = [
        "Obama policies",
        "Economic growth",
        "President of US"
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = query_graph(q, graph)

        for r in results:
            print(r)