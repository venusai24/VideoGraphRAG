import logging
from typing import Dict, Any, List
from temporal_clip_graph import parse_time
from semantic_graph import get_entity_id, filter_instances, normalize_name

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

    def build_graph(self, clip_data: Dict[str, Dict[str, Any]], clip_graph_conn=None):
        """
        Calculates overlaps between Entities and Clips, and builds the mapping graph.
        In order to do temporal overlap checks, we need to know the clip intervals.
        We can pull the Clip nodes from the Clip Graph or calculate them exactly the same way.
        """
        self.create_constraints()
        
        # To calculate overlaps, we need the clip boundaries. 
        # If we have access to the clip graph connection, we can fetch them.
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
            logger.warning("No clip intervals found. Mapping graph will be empty. Ensure Clip Graph is built first and connection is passed.")
            return

        all_mappings = []

        entity_categories = {
            'namedPeople': 'person',
            'brands': 'brand',
            'namedLocations': 'location',
            'topics': 'topic',
            'labels': 'label',
            'detectedObjects': 'detected_object'
        }

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
                    i_start = parse_time(str(i_start))
                    
                    i_end = inst.get('endSeconds') or inst.get('end') or inst.get('endTime', '')
                    i_end = parse_time(str(i_end))
                    
                    confidence = float(inst.get('confidence', default_conf))
                    
                    for clip in clip_intervals:
                        if clip['video_id'] != video_id:
                            continue
                            
                        if max(float(i_start), clip['start']) < min(float(i_end), clip['end']):
                            all_mappings.append({
                                'entity_id': entity_node_id,
                                'clip_id': clip['node_id'],
                                'confidence': confidence,
                                'source': source_name,
                                'timestamp': float(i_start)
                            })

            if raw_insights:
                for category, entity_type in entity_categories.items():
                    for entity in raw_insights.get(category, []):
                        node_id = get_entity_id(entity, entity_type)
                        if node_id:
                            arrays = entity.get('instances', []) + entity.get('appearances', [])
                            extract_mappings(node_id, arrays, source_name=category)
                
                summarized = raw_insights.get('summarizedInsights', {})
                for sent in summarized.get('sentiments', []):
                    node_id = get_entity_id(sent, 'sentiment')
                    if node_id:
                        extract_mappings(node_id, sent.get('appearances', []), source_name='sentiments', default_conf=1.0)
                for em in summarized.get('emotions', []):
                    node_id = get_entity_id(em, 'emotion')
                    if node_id:
                        extract_mappings(node_id, em.get('appearances', []), source_name='emotions', default_conf=1.0)

            if ocr and isinstance(ocr, list):
                for item in ocr:
                    if not isinstance(item, dict): continue
                    text = item.get('text')
                    if not text: continue
                    node_id_str = f"text_{normalize_name(text)}"
                    arrays = item.get('instances', []) + item.get('appearances', [])
                    extract_mappings(node_id_str, arrays, source_name='OCR')

        # Batch insert into Neo4j Mapping Graph
        self._batch_insert_mappings(all_mappings)
        logger.info(f"Mapping Graph Builder pushed {len(all_mappings)} APPEARS_IN mappings to Neo4j.")

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
