import logging
import itertools
from typing import Dict, Any, List
from semantic_graph import (
    get_entity_id, normalize_name, get_relationship_type,
    get_normalized_mapping, passes_seen_duration,
    ENTITY_CATEGORIES, MIN_SEEN_DURATION,
    _get_summarized_insights,
)

logger = logging.getLogger(__name__)


class EntityGraphBuilder:
    """
    Builds the Layer 2 Semantic Graph directly in Neo4j.

    Entity types ingested:
        person, brand, location, topic, label, detected_object,
        keyword, face, text (OCR)

    Excluded (stored as Clip scalar attributes instead):
        sentiments, emotions, audioEffects, framePatterns
    """

    def __init__(self, neo4j_connection):
        self.conn = neo4j_connection

    def create_constraints(self):
        try:
            self.conn.execute_write("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            self.conn.execute_write("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
        except Exception as e:
            logger.warning(f"Could not create constraints for Entity Graph: {e}")

    def build_graph(self, clip_data: Dict[str, Dict[str, Any]]):
        """
        Extracts entities from clip_data and pushes nodes + semantic edges to Neo4j.
        Uses MERGE so existing nodes (e.g. OCR text entities) are preserved.
        """
        self.create_constraints()

        all_entities   = {}
        subclass_edges = []
        cooccurrence_dict: Dict[tuple, int] = {}

        canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=0.85)

        def get_canonical(entity, e_type):
            raw_id = get_entity_id(entity, e_type)
            return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

        for folder_name, payloads in clip_data.items():
            si  = _get_summarized_insights(payloads)
            ocr = payloads.get('ocr')

            if not si and not ocr:
                continue

            entities_in_current_clip = set()

            # ── summarizedInsights entities ───────────────────────────────
            if si:
                for category, entity_type in ENTITY_CATEGORIES.items():
                    entities = si.get(category, [])
                    if not isinstance(entities, list):
                        continue

                    for entity in entities:
                        if not isinstance(entity, dict):
                            continue
                        if not passes_seen_duration(entity):
                            continue

                        node_id = get_canonical(entity, entity_type)
                        if not node_id:
                            continue

                        entities_in_current_clip.add(node_id)

                        if node_id not in all_entities:
                            attrs = {
                                'id':          node_id,
                                'type':        entity_type,
                                'name':        entity.get('name', ''),
                                'description': entity.get('description', '')
                            }
                            if entity_type == 'topic':
                                attrs['iabName']  = entity.get('iabName', '')
                                attrs['iptcName'] = entity.get('iptcName', '')
                            if entity_type == 'keyword':
                                attrs['isTranscript'] = entity.get('isTranscript', False)
                            all_entities[node_id] = attrs
                        else:
                            if not all_entities[node_id].get('description') and entity.get('description'):
                                all_entities[node_id]['description'] = entity.get('description')

                        # Topic hierarchy edges
                        if entity_type == 'topic':
                            name = entity.get('name', '')
                            if '/' in name:
                                parts = name.split('/')
                                for i in range(1, len(parts)):
                                    parent_name = '/'.join(parts[:i])
                                    child_name  = '/'.join(parts[:i + 1])
                                    parent_id   = f"topic_{normalize_name(parent_name)}"
                                    child_id    = f"topic_{normalize_name(child_name)}"

                                    if parent_id not in all_entities:
                                        all_entities[parent_id] = {
                                            'id': parent_id, 'type': 'topic',
                                            'name': parent_name, 'description': ''
                                        }
                                    if child_id not in all_entities:
                                        all_entities[child_id] = {
                                            'id': child_id, 'type': 'topic',
                                            'name': child_name, 'description': ''
                                        }
                                    subclass_edges.append((child_id, parent_id))

            # ── OCR text entities ─────────────────────────────────────────
            if ocr and isinstance(ocr, list):
                for item in ocr:
                    if not isinstance(item, dict):
                        continue
                    text = item.get('text')
                    if not text:
                        continue
                    node_id_str = f"text_{normalize_name(text)}"
                    entities_in_current_clip.add(node_id_str)
                    if node_id_str not in all_entities:
                        all_entities[node_id_str] = {
                            'id':          node_id_str,
                            'type':        'text',
                            'name':        text,
                            'description': 'Visual text from OCR'
                        }

            # ── Co-occurrence edges (typed entities only, NOT OCR text) ───────────
            # Excluding text entities keeps RELATED_TO edge count manageable.
            typed_entities = [
                e for e in entities_in_current_clip
                if not e.startswith('text_') and not e.startswith('face_')
            ]
            entities_list = list(typed_entities)
            for e1, e2 in itertools.combinations(entities_list, 2):
                pair = tuple(sorted([e1, e2]))
                cooccurrence_dict[pair] = cooccurrence_dict.get(pair, 0) + 1

        # ── Batch insert ──────────────────────────────────────────────────
        self._batch_insert_entities(list(all_entities.values()))

        subclass = [{"source": u, "target": v} for u, v in set(subclass_edges)]

        related = []
        for (u, v), weight in cooccurrence_dict.items():
            t1 = all_entities.get(u, {}).get('type', 'entity')
            t2 = all_entities.get(v, {}).get('type', 'entity')
            related.append({
                "source":            u,
                "target":            v,
                "weight":            weight,
                "relationship_type": get_relationship_type(t1, t2)
            })

        self._batch_insert_subclass(subclass)
        self._batch_insert_related(related)

        logger.info(f"EntityGraphBuilder pushed {len(all_entities)} entities to Neo4j.")
        logger.info(f"Edges: {len(subclass)} SUBCLASS_OF, {len(related)} RELATED_TO.")

    # ── Neo4j write helpers ───────────────────────────────────────────────────

    def _batch_insert_entities(self, entities: List[Dict[str, Any]], batch_size: int = 500):
        if not entities:
            return
        # Split into batches to avoid query-size limits
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            query = """
            UNWIND $entities AS entity
            MERGE (e:Entity {id: entity.id})
            SET e.type        = entity.type,
                e.name        = entity.name,
                e.description = entity.description,
                e.iabName     = entity.iabName,
                e.iptcName    = entity.iptcName,
                e.isTranscript= entity.isTranscript
            """
            self.conn.execute_write(query, {"entities": batch})
        logger.info(f"  Inserted/merged {len(entities)} entity nodes.")

    def _batch_insert_subclass(self, edges: List[Dict[str, Any]]):
        if not edges:
            return
        query = """
        UNWIND $edges AS edge
        MATCH (source:Entity {id: edge.source})
        MATCH (target:Entity {id: edge.target})
        MERGE (source)-[r:SUBCLASS_OF]->(target)
        """
        self.conn.execute_write(query, {"edges": edges})

    def _batch_insert_related(self, edges: List[Dict[str, Any]], batch_size: int = 1000):
        if not edges:
            return
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            query = """
            UNWIND $edges AS edge
            MATCH (source:Entity {id: edge.source})
            MATCH (target:Entity {id: edge.target})
            MERGE (source)-[r:RELATED_TO]->(target)
            SET r.weight            = edge.weight,
                r.relationship_type = edge.relationship_type
            """
            self.conn.execute_write(query, {"edges": batch})
        logger.info(f"  Inserted/merged {len(edges)} RELATED_TO edges.")
