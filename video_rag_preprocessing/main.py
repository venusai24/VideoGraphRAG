import sys
from data_loader import VideoDataLoader
from temporal_clip_graph import build_temporal_clip_graph, print_temporal_graph_samples
from semantic_graph import build_semantic_graph, build_bipartite_mapping, build_bipartite_mapping_dict, print_semantic_graph_samples
from retrieval import VideoGraphRetriever, _tokenize

# --- Configuration ---
OUTPUTS_DIR = "outputs/"
CHUNK_A = 4.0
CHUNK_B = 6.0

def main():
    print(f"Loading data from {OUTPUTS_DIR}...")
    loader = VideoDataLoader(OUTPUTS_DIR)
    data = loader.load_data()
    
    if not data:
        print("No data loaded. Please check the outputs directory.")
        return

    print("Building Layer 1: Temporal Clip Graph...")
    graph = build_temporal_clip_graph(data, a=CHUNK_A, b=CHUNK_B)
    print(f"  -> Layer 1 completed. Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print_temporal_graph_samples(graph)

    print("Building Layer 2: Semantic Graph...")
    graph = build_semantic_graph(graph, data)
    print(f"  -> Layer 2 completed. Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    print("Building Bipartite Mapping Layer...")
    graph = build_bipartite_mapping(graph, data)
    bipartite_dict = build_bipartite_mapping_dict(graph, data)
    print(f"  -> Bipartite mapping completed. Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print_semantic_graph_samples(graph)

    print("Initializing Retrieval Pipeline (pre-computing entity embeddings)...")
    retriever = VideoGraphRetriever(graph, bipartite_dict)
    
    print("\n" + "="*70)
    print("GraphRAG Pipeline Ready!")
    print("="*70 + "\n")

    while True:
        try:
            query = input("Enter your search query (or type exit to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if query.lower() in ['exit', 'quit']:
            break
            
        if not query:
            continue
            
        print(f"\nSearching for: '{query}'...")
        result = retriever.retrieve(query, top_n_clips=5)
        
        if not result.clips:
            print("No matching clips found.")
            print("-" * 70 + "\n")
            continue
            
        print("\n--- Diagnostic Info ---")
        print(f"Seed Entities Mapped   : {[(eid, round(score, 4)) for eid, score in result.seed_entities]}")
        print(f"Candidate Clips Checked: {result.candidate_clip_count}")
        query_tokens = _tokenize(result.query)
        print(f"BM25 Search Tokens     : {query_tokens}")
        print("-----------------------\n")
            
        print(f"\nTop {len(result.clips)} Results:")
        for i, clip in enumerate(result.clips, 1):
            snippet = clip.summary[:150] + "..." if len(clip.summary) > 150 else clip.summary
            
            # Fallback to transcript snippet if summary is empty
            if not snippet.strip():
                snippet = clip.transcript[:150] + "..." if len(clip.transcript) > 150 else clip.transcript

            print(f"  #{i}")
            print(f"     Video ID       : {clip.video_id}")
            print(f"     Time Range     : {clip.start:.2f}s - {clip.end:.2f}s")
            print(f"     Relevance Score: {clip.final_score:.4f} (Hybrid: BM25 {clip.bm25_score:.2f} + Semantic)")
            print(f"     Summary Snippet: {snippet}")
            print()
            
        print("-" * 70 + "\n")

if __name__ == "__main__":
    main()
