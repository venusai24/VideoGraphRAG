import logging
import itertools
from typing import Dict, Any, List
from semantic_graph import get_entity_id, normalize_name, get_relationship_type, get_normalized_mapping

logger = logging.getLogger(__name__)

class EntityGraphBuilder:
    """
    Builds the Layer 2 Semantic Graph directly in Neo4j.
    """
    def __init__(self, neo4j_connection):
        self.conn = neo4j_connection

    def create_constraints(self):
        """Create constraints and indexes for the Entity Graph."""
        try:
            self.conn.execute_write("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            self.conn.execute_write("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
        except Exception as e:
            logger.warning(f"Could not create constraints for Entity Graph: {e}")

    def build_graph(self, clip_data: Dict[str, Dict[str, Any]]):
        """
        Extracts entities from clip_data and pushes nodes and semantic edges to Neo4j.
        """
        self.create_constraints()
        
        all_entities = {}
        subclass_edges = []
        assoc_edges = []
        expressed_edges = []
        cooccurrence_dict = {}  # (e1, e2) -> weight
        
        entity_categories = {
            'namedPeople': 'person',
            'brands': 'brand',
            'namedLocations': 'location',
            'topics': 'topic',
            'labels': 'label',
            'detectedObjects': 'detected_object'
        }

        canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=0.85)

        def get_canonical(entity, e_type):
            raw_id = get_entity_id(entity, e_type)
            return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

        for folder_name, payloads in clip_data.items():
            raw_insights = payloads.get('raw_insights')
            ocr = payloads.get('ocr')
            
            if not raw_insights and not ocr:
                continue
                
            entities_in_current_clip = set()
            
            if raw_insights:
                # 1. Add normal entities
                for category, entity_type in entity_categories.items():
                    entities = raw_insights.get(category, [])
                    if not isinstance(entities, list): continue
                        
                    for entity in entities:
                        if not isinstance(entity, dict): continue
                        
                        node_id = get_canonical(entity, entity_type)
                        if not node_id: continue
                        
                        entities_in_current_clip.add(node_id)
                        
                        if node_id not in all_entities:
                            attrs = {
                                'id': node_id,
                                'type': entity_type,
                                'name': entity.get('name', ''),
                                'description': entity.get('description', '')
                            }
                            if entity_type == 'topic':
                                attrs['iabName'] = entity.get('iabName', '')
                                attrs['iptcName'] = entity.get('iptcName', '')
                            all_entities[node_id] = attrs
                        else:
                            if not all_entities[node_id].get('description') and entity.get('description'):
                                all_entities[node_id]['description'] = entity.get('description')
                                
                        # Hierarchy edges for topics
                        if entity_type == 'topic':
                            name = entity.get('name', '')
                            if '/' in name:
                                parts = name.split('/')
                                for i in range(1, len(parts)):
                                    parent_name = '/'.join(parts[:i])
                                    child_name = '/'.join(parts[:i+1])
                                    parent_id = f"topic_{normalize_name(parent_name)}"
                                    child_id = f"topic_{normalize_name(child_name)}"
                                    
                                    if parent_id not in all_entities:
                                        all_entities[parent_id] = {'id': parent_id, 'type': 'topic', 'name': parent_name, 'description': ''}
                                    if child_id not in all_entities:
                                        all_entities[child_id] = {'id': child_id, 'type': 'topic', 'name': child_name, 'description': ''}
                                        
                                    subclass_edges.append((child_id, parent_id))
                                    
                # 2. Add Sentiments and Emotions
                summarized = raw_insights.get('summarizedInsights', {})
                
                sentiments = summarized.get('sentiments', [])
                for sent in sentiments:
                    node_id = get_canonical(sent, 'sentiment')
                    if node_id:
                        entities_in_current_clip.add(node_id)
                        if node_id not in all_entities:
                            all_entities[node_id] = {'id': node_id, 'type': 'sentiment', 'name': sent.get('sentimentKey', ''), 'description': ''}
                        
                emotions = summarized.get('emotions', [])
                for em in emotions:
                    node_id = get_canonical(em, 'emotion')
                    if node_id:
                        entities_in_current_clip.add(node_id)
                        if node_id not in all_entities:
                            all_entities[node_id] = {'id': node_id, 'type': 'emotion', 'name': em.get('type', ''), 'description': ''}
                        
                # 3. Person -> Emotion/Sentiment edges
                # (Skipped full temporal overlap check for brevity in entity-only context, 
                # but we will connect them if they co-exist in this clip's payload)
                people = raw_insights.get('namedPeople', [])
                for person in people:
                    p_id = get_canonical(person, 'person')
                    if not p_id: continue
                    
                    for sent in sentiments:
                        s_id = get_canonical(sent, 'sentiment')
                        if s_id: assoc_edges.append((p_id, s_id))
                        
                    for em in emotions:
                        e_id = get_canonical(em, 'emotion')
                        if e_id: expressed_edges.append((p_id, e_id))

            # 4. Add OCR Text
            if ocr and isinstance(ocr, list):
                for item in ocr:
                    if not isinstance(item, dict): continue
                    text = item.get('text')
                    if not text: continue
                    
                    node_id_str = f"text_{normalize_name(text)}"
                    entities_in_current_clip.add(node_id_str)
                    
                    if node_id_str not in all_entities:
                        all_entities[node_id_str] = {'id': node_id_str, 'type': 'text', 'name': text, 'description': 'Visual text from OCR'}

            # 5. Build Clip-Based Co-occurrence Edges
            entities_list = list(entities_in_current_clip)
            for e1, e2 in itertools.combinations(entities_list, 2):
                # Ensure consistent ordering so we don't duplicate undirected edges as two directed edges unless desired
                # We'll just store them in one direction and Neo4j can query undirected.
                pair = tuple(sorted([e1, e2]))
                cooccurrence_dict[pair] = cooccurrence_dict.get(pair, 0) + 1

        # Batch insert into Neo4j
        self._batch_insert_entities(list(all_entities.values()))
        
        # Insert edges
        subclass = [{"source": u, "target": v} for u, v in set(subclass_edges)]
        assoc = [{"source": u, "target": v} for u, v in set(assoc_edges)]
        expr = [{"source": u, "target": v} for u, v in set(expressed_edges)]
        
        related = []
        for (u, v), weight in cooccurrence_dict.items():
            t1 = all_entities[u]['type']
            t2 = all_entities[v]['type']
            rel_type = get_relationship_type(t1, t2)
            related.append({
                "source": u,
                "target": v,
                "weight": weight,
                "relationship_type": rel_type
            })

        self._batch_insert_subclass(subclass)
        self._batch_insert_assoc(assoc)
        self._batch_insert_expr(expr)
        self._batch_insert_related(related)
        
        logger.info(f"Entity Graph Builder pushed {len(all_entities)} entities to Neo4j.")
        logger.info(f"Pushed edges: {len(subclass)} SUBCLASS_OF, {len(assoc)} ASSOCIATED_WITH, {len(expr)} EXPRESSED, {len(related)} RELATED_TO.")

    def _batch_insert_entities(self, entities: List[Dict[str, Any]]):
        if not entities: return
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {id: entity.id})
        SET e.type = entity.type,
            e.name = entity.name,
            e.description = entity.description,
            e.iabName = entity.iabName,
            e.iptcName = entity.iptcName
        """
        self.conn.execute_write(query, {"entities": entities})

    def _batch_insert_subclass(self, edges: List[Dict[str, Any]]):
        if not edges: return
        query = """
        UNWIND $edges AS edge
        MATCH (source:Entity {id: edge.source})
        MATCH (target:Entity {id: edge.target})
        MERGE (source)-[r:SUBCLASS_OF]->(target)
        """
        self.conn.execute_write(query, {"edges": edges})

    def _batch_insert_assoc(self, edges: List[Dict[str, Any]]):
        if not edges: return
        query = """
        UNWIND $edges AS edge
        MATCH (source:Entity {id: edge.source})
        MATCH (target:Entity {id: edge.target})
        MERGE (source)-[r:ASSOCIATED_WITH]->(target)
        """
        self.conn.execute_write(query, {"edges": edges})

    def _batch_insert_expr(self, edges: List[Dict[str, Any]]):
        if not edges: return
        query = """
        UNWIND $edges AS edge
        MATCH (source:Entity {id: edge.source})
        MATCH (target:Entity {id: edge.target})
        MERGE (source)-[r:EXPRESSED]->(target)
        """
        self.conn.execute_write(query, {"edges": edges})

    def _batch_insert_related(self, edges: List[Dict[str, Any]]):
        if not edges: return
        query = """
        UNWIND $edges AS edge
        MATCH (source:Entity {id: edge.source})
        MATCH (target:Entity {id: edge.target})
        MERGE (source)-[r:RELATED_TO]->(target)
        SET r.weight = edge.weight, r.relationship_type = edge.relationship_type
        """
        self.conn.execute_write(query, {"edges": edges})
