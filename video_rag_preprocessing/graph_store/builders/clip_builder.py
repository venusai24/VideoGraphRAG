import logging
from typing import Dict, Any, List, Tuple
from temporal_clip_graph import parse_time, get_overlapping_text, calculate_average_sentiment, get_most_frequent_emotion, get_overlapping_speakers

logger = logging.getLogger(__name__)

class ClipGraphBuilder:
    """
    Builds the Layer 1 Temporal Clip Graph directly in Neo4j.
    """
    def __init__(self, neo4j_connection):
        self.conn = neo4j_connection

    def create_constraints(self):
        """Create constraints and indexes for the Clip Graph."""
        try:
            self.conn.execute_write("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clip) REQUIRE c.id IS UNIQUE")
            self.conn.execute_write("CREATE INDEX IF NOT EXISTS FOR (c:Clip) ON (c.video_id)")
        except Exception as e:
            logger.warning(f"Could not create constraints for Clip Graph: {e}")

    def build_graph(self, clip_data: Dict[str, Dict[str, Any]], a: float = 10.0, b: float = 12.0):
        """
        Processes loaded clip data, slices it, and builds nodes and NEXT edges in Neo4j.
        """
        self.create_constraints()
        
        all_clips = []
        all_edges = []
        
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
            
            for start, end in all_scene_instances:
                if current_segment is None:
                    current_segment = [start, end]
                else:
                    combined_duration = end - current_segment[0]
                    if combined_duration < a:
                        current_segment[1] = max(current_segment[1], end)
                    else:
                        merged_segments.append(current_segment)
                        current_segment = [start, end]
                        
            if current_segment is not None:
                merged_segments.append(current_segment)
                
            # Partition segments longer than b
            final_segments: List[Tuple[float, float]] = []
            target_chunk_duration = (a + b) / 2.0
            
            for start, end in merged_segments:
                duration = end - start
                if duration > b:
                    num_chunks = max(1, round(duration / target_chunk_duration))
                    chunk_duration = duration / num_chunks
                    for i in range(num_chunks):
                        chunk_start = start + i * chunk_duration
                        chunk_end = start + (i + 1) * chunk_duration if i < num_chunks - 1 else end
                        final_segments.append((chunk_start, chunk_end))
                else:
                    final_segments.append((start, end))
                    
            # Determine Video ID
            video_id = folder_name
            if rag_chunks and isinstance(rag_chunks, list) and len(rag_chunks) > 0:
                video_id = rag_chunks[0].get('video_id', folder_name)
                
            # 3. Create nodes and edges locally, then batch write to Neo4j
            prev_node_id = None
            for start, end in final_segments:
                node_id = f"{video_id}_{start:.2f}_{end:.2f}"
                
                transcript_text = get_overlapping_text(transcript, start, end, 'text')
                ocr_text = get_overlapping_text(ocr, start, end, 'text')
                rag_summary = get_overlapping_text(rag_chunks, start, end, 'text')
                keywords_text = get_overlapping_text(keywords, start, end, 'text')
                
                avg_sentiment = calculate_average_sentiment(sentiments, start, end)
                freq_emotion = get_most_frequent_emotion(emotions, start, end)
                speaker_ids = get_overlapping_speakers(transcript, start, end)
                
                all_clips.append({
                    "id": node_id,
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "transcript": transcript_text,
                    "ocr": ocr_text,
                    "keywords": keywords_text,
                    "summary": rag_summary,
                    "average_sentiment": avg_sentiment,
                    "emotion": freq_emotion,
                    "speaker_ids": speaker_ids
                })
                
                if prev_node_id is not None:
                    all_edges.append({
                        "source": prev_node_id,
                        "target": node_id,
                        "type": "NEXT"
                    })
                    
                prev_node_id = node_id

        # Batch insert into Neo4j
        self._batch_insert_clips(all_clips)
        self._batch_insert_edges(all_edges)
        logger.info(f"Clip Graph Builder pushed {len(all_clips)} clips and {len(all_edges)} NEXT edges to Neo4j.")

    def _batch_insert_clips(self, clips: List[Dict[str, Any]]):
        if not clips:
            return
            
        query = """
        UNWIND $clips AS clip
        MERGE (c:Clip {id: clip.id})
        SET c.video_id = clip.video_id,
            c.start = clip.start,
            c.end = clip.end,
            c.transcript = clip.transcript,
            c.ocr = clip.ocr,
            c.keywords = clip.keywords,
            c.summary = clip.summary,
            c.average_sentiment = clip.average_sentiment,
            c.emotion = clip.emotion,
            c.speaker_ids = clip.speaker_ids
        """
        self.conn.execute_write(query, {"clips": clips})
        
    def _batch_insert_edges(self, edges: List[Dict[str, Any]]):
        if not edges:
            return
            
        query = """
        UNWIND $edges AS edge
        MATCH (source:Clip {id: edge.source})
        MATCH (target:Clip {id: edge.target})
        MERGE (source)-[r:NEXT]->(target)
        """
        self.conn.execute_write(query, {"edges": edges})
