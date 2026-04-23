from video_rag_preprocessing.data_loader import VideoDataLoader
from video_rag_preprocessing.temporal_clip_graph import build_temporal_clip_graph
from video_rag_preprocessing.semantic_graph import build_semantic_graph, build_bipartite_mapping, build_bipartite_mapping_dict
from video_rag_preprocessing.retrieval import VideoGraphRetriever

loader = VideoDataLoader("outputs/")
data = loader.load_data()
G = build_temporal_clip_graph(data)
G = build_semantic_graph(G, data)
G = build_bipartite_mapping(G, data)
bipartite_dict = build_bipartite_mapping_dict(G, data)

retriever = VideoGraphRetriever(G, bipartite_dict)

queries = ["Who is President of USA?", "Obama is President of Which Country?"]
for q in queries:
    print(f"--- Query: {q} ---")
    res = retriever.retrieve(q, top_n_clips=3)
    for i, c in enumerate(res.clips):
        print(f"Result {i+1}: Score={c.bm25_score:.4f}, Final={c.final_score:.4f}, Video={c.video_id}, {c.start}-{c.end}")
        print(c.summary[:100])
