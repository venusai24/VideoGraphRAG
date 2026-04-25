import logging
from typing import Dict, Any, List
import math
import itertools
from semantic_graph import get_entity_id, filter_instances, normalize_name, get_normalized_mapping

logger = logging.getLogger(__name__)

class MappingGraphBuilder:
    """
    Builds the Cross-Graph Mapping Layer in Neo4j.
    Connects EntityRef nodes to ClipRef nodes via APPEARS_IN relationships.
    """
    def __init__(self, neo4j_connection):
        self.conn = neo4j_connection

    def create_constraints(self):
        """Create constraints and indexes for the Mapping Graph."""
        try:
            self.conn.execute_write("CREATE CONSTRAINT IF NOT EXISTS FOR (e:EntityRef) REQUIRE e.id IS UNIQUE")
            self.conn.execute_write("CREATE CONSTRAINT IF NOT EXISTS FOR (c:ClipRef) REQUIRE c.id IS UNIQUE")
        except Exception as e:
            logger.warning(f"Could not create constraints for Mapping Graph: {e}")

    def _batch_insert_mappings(self, mappings: List[Dict[str, Any]]):
        if not mappings: return
        query = """
        UNWIND $mappings AS mapping
        MERGE (e:EntityRef {id: mapping.entity_id})
        MERGE (c:ClipRef {id: mapping.clip_id})
        MERGE (e)-[r:APPEARS_IN]->(c)
        SET r.confidence = mapping.confidence,
            r.source = mapping.source,
            r.timestamp = mapping.timestamp
        """
        self.conn.execute_write(query, {"mappings": mappings})

    def _batch_insert_shares_entity(self, edges: List[Dict[str, Any]]):
        if not edges: return
        query = """
        UNWIND $edges AS edge
        MATCH (c1:ClipRef {id: edge.c1})
        MATCH (c2:ClipRef {id: edge.c2})
        MERGE (c1)-[r:SHARES_ENTITY]->(c2)
        SET r.weight = edge.weight
        """
        self.conn.execute_write(query, {"edges": edges})

    def build_graph(self, clip_data: Dict[str, Dict[str, Any]], clip_graph_conn=None, merge_threshold: float = 0.85, significance_threshold: float = 0.5):
        """
        Calculates overlaps between Entities and Clips, and builds the mapping graph.
        Also calculates and pushes SHARES_ENTITY edges.
        """
        self.create_constraints()
        
        # To calculate overlaps, we need the clip boundaries. 
        clip_intervals = []
        if clip_graph_conn:
            try:
                records = clip_graph_conn.execute_query("MATCH (c:Clip) RETURN c.id AS id, c.video_id AS video_id, c.start AS start, c.end AS end")
                for r in records:
                    clip_intervals.append({
                        'node_id': r['id'],
                        'video_id': r['video_id'],
                        'start': r['start'],
                        'end': r['end']
                    })
            except Exception as e:
                logger.error(f"Failed to fetch clip intervals from Clip Graph: {e}")
        
        if not clip_intervals:
            logger.warning("No clip intervals found. Mapping graph will be empty.")
            return

        all_mappings = []

        entity_categories = {
            'namedPeople': 'person', 'brands': 'brand', 'namedLocations': 'location',
            'topics': 'topic', 'labels': 'label', 'detectedObjects': 'detected_object'
        }

        canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=merge_threshold)

        def get_canonical(entity, e_type):
            raw_id = get_entity_id(entity, e_type)
            return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

        # Track for TF-IDF
        clip_entities = {} # clip_id -> {entity_id: confidence}
        for c in clip_intervals:
            clip_entities[c['node_id']] = {}

        for folder_name, payloads in clip_data.items():
            raw_insights = payloads.get('raw_insights')
            ocr = payloads.get('ocr')
            
            video_id = folder_name
            rag_chunks = payloads.get('rag_chunks')
            if rag_chunks and isinstance(rag_chunks, list) and len(rag_chunks) > 0:
                video_id = rag_chunks[0].get('video_id', folder_name)

            def extract_mappings(entity_node_id, instances_array, source_name, default_conf=1.0):
                valid_instances = filter_instances(instances_array, default_conf)
                for inst in valid_instances:
                    i_start = inst.get('startSeconds') or inst.get('start') or inst.get('startTime', '')
                    if i_start == '': continue
                    
                    i_end = inst.get('endSeconds') or inst.get('end') or inst.get('endTime', '')
                    if i_end == '': continue

                    try:
                        i_start = float(i_start)
                        i_end = float(i_end)
                    except ValueError:
                        continue
                    
                    confidence = float(inst.get('confidence', default_conf))
                    
                    for clip in clip_intervals:
                        if clip['video_id'] != video_id: continue
                            
                        if max(i_start, clip['start']) < min(i_end, clip['end']):
                            all_mappings.append({
                                'entity_id': entity_node_id,
                                'clip_id': clip['node_id'],
                                'confidence': confidence,
                                'source': source_name,
                                'timestamp': float(i_start)
                            })
                            # Keep highest confidence for clip
                            existing_conf = clip_entities[clip['node_id']].get(entity_node_id, 0.0)
                            if confidence > existing_conf:
                                clip_entities[clip['node_id']][entity_node_id] = confidence

            if raw_insights:
                for category, entity_type in entity_categories.items():
                    for entity in raw_insights.get(category, []):
                        node_id = get_canonical(entity, entity_type)
                        if node_id:
                            arrays = entity.get('instances', []) + entity.get('appearances', [])
                            extract_mappings(node_id, arrays, source_name=category)
                
                summarized = raw_insights.get('summarizedInsights', {})
                for sent in summarized.get('sentiments', []):
                    node_id = get_canonical(sent, 'sentiment')
                    if node_id: extract_mappings(node_id, sent.get('appearances', []), source_name='sentiments', default_conf=1.0)
                for em in summarized.get('emotions', []):
                    node_id = get_canonical(em, 'emotion')
                    if node_id: extract_mappings(node_id, em.get('appearances', []), source_name='emotions', default_conf=1.0)

            if ocr and isinstance(ocr, list):
                for item in ocr:
                    if not isinstance(item, dict): continue
                    text = item.get('text')
                    if not text: continue
                    node_id_str = f"text_{normalize_name(text)}"
                    arrays = item.get('instances', []) + item.get('appearances', [])
                    extract_mappings(node_id_str, arrays, source_name='OCR')

        self._batch_insert_mappings(all_mappings)
        logger.info(f"Mapping Graph Builder pushed {len(all_mappings)} APPEARS_IN mappings to Neo4j.")

        # Calculate SHARES_ENTITY edges
        N = len(clip_intervals)
        entity_df = {}
        for c_id, ents in clip_entities.items():
            for e in ents:
                entity_df[e] = entity_df.get(e, 0) + 1
        
        entity_idf = {}
        for e, df in entity_df.items():
            entity_idf[e] = math.log(N / max(1, df))

        shares_entity_edges = []
        clip_ids = list(clip_entities.keys())
        for i in range(len(clip_ids)):
            c1 = clip_ids[i]
            for j in range(i + 1, len(clip_ids)):
                c2 = clip_ids[j]
                shared = set(clip_entities[c1].keys()).intersection(set(clip_entities[c2].keys()))
                if not shared: continue

                score = 0.0
                for e in shared:
                    avg_conf = (clip_entities[c1][e] + clip_entities[c2][e]) / 2.0
                    score += avg_conf * entity_idf[e]
                
                if score >= significance_threshold:
                    shares_entity_edges.append({'c1': c1, 'c2': c2, 'weight': score})
                    shares_entity_edges.append({'c1': c2, 'c2': c1, 'weight': score})

        self._batch_insert_shares_entity(shares_entity_edges)
        logger.info(f"Mapping Graph Builder pushed {len(shares_entity_edges)} SHARES_ENTITY edges to Neo4j (threshold={significance_threshold}).")
