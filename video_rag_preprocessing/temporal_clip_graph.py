import networkx as nx
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

def parse_time(time_str: str) -> float:
    """
    Parses 'H:MM:SS.ffffff' or 'M:SS' into total seconds robustly.
    Handles varying lengths of fractional seconds.
    """
    if not time_str:
        return 0.0
        
    parts = time_str.split(':')
    try:
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = "0"
            m, s = parts
        else:
            return float(time_str)
            
        sec_parts = s.split('.')
        sec = float(sec_parts[0])
        frac = 0.0
        if len(sec_parts) == 2:
            frac = float('.' + sec_parts[1])
            
        return int(h) * 3600 + int(m) * 60 + sec + frac
    except ValueError:
        logger.warning(f"Could not parse time string: {time_str}")
        return 0.0

def get_overlapping_text(data_list: List[Dict[str, Any]], seg_start: float, seg_end: float, text_key: str = 'text') -> str:
    """
    Extracts and concatenates text from items whose timeframes overlap with the segment.
    """
    texts = []
    if not data_list:
        return ""
        
    for item in data_list:
        if not isinstance(item, dict):
            continue
            
        item_text = item.get(text_key, "")
        if not item_text:
            continue
            
        # Check instances array (used in transcript.json, ocr.json)
        instances = item.get('instances', [])
        added = False
        
        if instances:
            for inst in instances:
                i_start = parse_time(inst.get('start', ''))
                i_end = parse_time(inst.get('end', ''))
                # Overlap condition: max(start1, start2) < min(end1, end2)
                if max(seg_start, i_start) < min(seg_end, i_end):
                    texts.append(str(item_text))
                    added = True
                    break # Add text only once per item even if multiple instances overlap
        else:
            # Fallback for formats like rag_chunks.json which use start_time/end_time directly
            start_str = item.get('start_time')
            end_str = item.get('end_time')
            if start_str and end_str:
                i_start = parse_time(start_str)
                i_end = parse_time(end_str)
                if max(seg_start, i_start) < min(seg_end, i_end):
                    texts.append(str(item_text))
                    
    return " ".join(texts)

def calculate_average_sentiment(sentiments: List[Dict[str, Any]], seg_start: float, seg_end: float) -> float:
    """
    Calculates the duration-weighted average sentiment score for a given timeframe.
    Maps Positive -> 1.0, Neutral -> 0.0, Negative -> -1.0.
    """
    score_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    total_score = 0.0
    total_duration = 0.0
    if not sentiments:
        return 0.0
        
    for sent in sentiments:
        if not isinstance(sent, dict): continue
        key = sent.get('sentimentKey')
        if not key or key not in score_map:
            continue
        val = score_map[key]
        for app in sent.get('appearances', []):
            if not isinstance(app, dict): continue
            i_start = app.get('startSeconds')
            i_end = app.get('endSeconds')
            if i_start is None:
                i_start = parse_time(app.get('startTime', ''))
            if i_end is None:
                i_end = parse_time(app.get('endTime', ''))
                
            overlap_start = max(seg_start, float(i_start))
            overlap_end = min(seg_end, float(i_end))
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                total_score += val * duration
                total_duration += duration
                
    if total_duration > 0:
        return round(total_score / total_duration, 3)
    return 0.0

def get_most_frequent_emotion(emotions: List[Dict[str, Any]], seg_start: float, seg_end: float) -> str:
    """
    Finds the most frequent emotion type based on total overlap duration.
    """
    emotion_durations = {}
    if not emotions:
        return ""
        
    for em in emotions:
        if not isinstance(em, dict): continue
        e_type = em.get('type')
        if not e_type:
            continue
        for app in em.get('appearances', []):
            if not isinstance(app, dict): continue
            i_start = app.get('startSeconds')
            i_end = app.get('endSeconds')
            if i_start is None:
                i_start = parse_time(app.get('startTime', ''))
            if i_end is None:
                i_end = parse_time(app.get('endTime', ''))
                
            overlap_start = max(seg_start, float(i_start))
            overlap_end = min(seg_end, float(i_end))
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                emotion_durations[e_type] = emotion_durations.get(e_type, 0.0) + duration
                
    if not emotion_durations:
        return ""
    return max(emotion_durations.items(), key=lambda x: x[1])[0]

def get_overlapping_speakers(transcript: List[Dict[str, Any]], seg_start: float, seg_end: float) -> List[int]:
    """
    Extracts a list of unique speakerIds that overlap with the timeframe.
    """
    speakers = set()
    if not transcript:
        return []
        
    for item in transcript:
        if not isinstance(item, dict): continue
        speaker_id = item.get('speakerId')
        if speaker_id is None:
            continue
            
        instances = item.get('instances', [])
        for inst in instances:
            if not isinstance(inst, dict): continue
            i_start = parse_time(inst.get('start', ''))
            i_end = parse_time(inst.get('end', ''))
            if max(seg_start, i_start) < min(seg_end, i_end):
                speakers.add(speaker_id)
                break # Only need to add the speaker once if any instance overlaps
    return list(speakers)

def build_temporal_clip_graph(clip_data: Dict[str, Dict[str, Any]], a: float = 10.0, b: float = 12.0) -> nx.DiGraph:
    """
    Processes loaded clip data to build a temporal clip graph (Layer 1).
    
    Args:
        clip_data: Dictionary structured as { folder_name: { payload_name: payload_data } }
        a: Join consecutive scenes if their combined duration is < 'a' seconds.
        b: If a scene or combined segment > 'b' seconds, partition into chunks of ~ (a+b)/2 seconds.
        
    Returns:
        A NetworkX DiGraph representing chronological synthetic clips and their attributes.
    """
    G = nx.DiGraph()
    
    for folder_name, payloads in clip_data.items():
        scenes = payloads.get('scenes')
        transcript = payloads.get('transcript')
        ocr = payloads.get('ocr')
        rag_chunks = payloads.get('rag_chunks')
        keywords = payloads.get('keywords')
        raw_insights = payloads.get('raw_insights')
        
        sentiments = []
        emotions = []
        if raw_insights and isinstance(raw_insights, dict):
            summarized = raw_insights.get('summarizedInsights', {})
            sentiments = summarized.get('sentiments', [])
            emotions = summarized.get('emotions', [])
        
        if not scenes:
            logger.info(f"No scenes found for {folder_name}. Skipping graph generation.")
            continue
            
        # 1. Extract and sort all scene instances
        all_scene_instances: List[Tuple[float, float]] = []
        for scene in scenes:
            for inst in scene.get('instances', []):
                start_sec = parse_time(inst.get('start', ''))
                end_sec = parse_time(inst.get('end', ''))
                if end_sec > start_sec:
                    all_scene_instances.append((start_sec, end_sec))
                    
        all_scene_instances.sort(key=lambda x: x[0])
        
        # 2. Temporal Slicing Algorithm
        merged_segments: List[List[float]] = []
        current_segment = None
        
        # Step 2a: Join consecutive scenes if combined duration < a
        for start, end in all_scene_instances:
            if current_segment is None:
                current_segment = [start, end]
            else:
                combined_duration = end - current_segment[0]
                # Note: Assuming 'consecutive' implies we just extend the current segment
                # If there are gaps between scenes, this merges across the gap.
                if combined_duration < a:
                    current_segment[1] = max(current_segment[1], end)
                else:
                    merged_segments.append(current_segment)
                    current_segment = [start, end]
                    
        if current_segment is not None:
            merged_segments.append(current_segment)
            
        # Step 2b: Partition segments longer than b
        final_segments: List[Tuple[float, float]] = []
        target_chunk_duration = (a + b) / 2.0
        
        for start, end in merged_segments:
            duration = end - start
            if duration > b:
                # Partition into equal chunks
                num_chunks = max(1, round(duration / target_chunk_duration))
                chunk_duration = duration / num_chunks
                for i in range(num_chunks):
                    chunk_start = start + i * chunk_duration
                    chunk_end = start + (i + 1) * chunk_duration if i < num_chunks - 1 else end
                    final_segments.append((chunk_start, chunk_end))
            else:
                final_segments.append((start, end))
                
        # Determine Video ID (fallback to folder name if missing)
        video_id = folder_name
        if rag_chunks and isinstance(rag_chunks, list) and len(rag_chunks) > 0:
            video_id = rag_chunks[0].get('video_id', folder_name)
            
        # 3. Create nodes and edges
        prev_node_id = None
        for start, end in final_segments:
            # Node ID Format: VideoID_Start_End
            node_id = f"{video_id}_{start:.2f}_{end:.2f}"
            
            transcript_text = get_overlapping_text(transcript, start, end, 'text')
            ocr_text = get_overlapping_text(ocr, start, end, 'text')
            rag_summary = get_overlapping_text(rag_chunks, start, end, 'text')
            keywords_text = get_overlapping_text(keywords, start, end, 'text')
            
            avg_sentiment = calculate_average_sentiment(sentiments, start, end)
            freq_emotion = get_most_frequent_emotion(emotions, start, end)
            speaker_ids = get_overlapping_speakers(transcript, start, end)
            
            G.add_node(
                node_id,
                node_class='Clip',
                video_id=video_id,
                start=start,
                end=end,
                transcript=transcript_text,
                ocr=ocr_text,
                keywords=keywords_text,
                summary=rag_summary,
                average_sentiment=avg_sentiment,
                emotion=freq_emotion,
                speaker_ids=speaker_ids
            )
            
            if prev_node_id is not None:
                # 5. Create directed edges connecting chronological clips
                G.add_edge(prev_node_id, node_id, type='NEXT')
                
            prev_node_id = node_id
            
    return G

def print_temporal_graph_samples(G: nx.DiGraph, num_samples: int = 5):
    """
    Prints a few sample nodes and edges from the Layer 1 graph.
    """
    print(f"\n--- Layer 1 (Temporal) Graph Samples ---")
    
    # Sample Nodes
    nodes = list(G.nodes(data=True))
    sample_nodes = nodes[:num_samples]
    print(f"Sample Nodes ({len(sample_nodes)}):")
    for node_id, attrs in sample_nodes:
        print(f"  Node ID: {node_id}")
        for k, v in attrs.items():
            val = str(v)
            if len(val) > 80: val = val[:77] + "..."
            print(f"    {k}: {val}")
            
    # Sample Edges
    edges = list(G.edges(data=True))
    sample_edges = edges[:num_samples]
    print(f"\nSample Edges ({len(sample_edges)}):")
    for u, v, attrs in sample_edges:
        print(f"  Edge: {u} -> {v}")
        for ak, av in attrs.items():
            print(f"    {ak}: {av}")

if __name__ == "__main__":
    # Test script for rapid verification
    from data_loader import VideoDataLoader
    import sys
    
    logging.basicConfig(level=logging.INFO)
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    
    loader = VideoDataLoader(target_dir)
    data = loader.load_data()
    
    graph = build_temporal_clip_graph(data, a=10.0, b=12.0)
    
    print(f"Generated NetworkX DiGraph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    if graph.number_of_nodes() > 0:
        print_temporal_graph_samples(graph)
