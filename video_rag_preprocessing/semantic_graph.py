import networkx as nx
from typing import Dict, Any, List
import logging
import itertools
from temporal_clip_graph import parse_time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from entity_normalizer import EntityNormalizer

logger = logging.getLogger(__name__)

# Minimum seenDuration (seconds) for summarizedInsights entities.
# Filters out fleeting detections. Entities without seenDuration (topics, keywords) always pass.
MIN_SEEN_DURATION = 1.0

# Entity categories sourced from summarizedInsights
ENTITY_CATEGORIES = {
    'namedPeople':     'person',
    'brands':          'brand',
    'namedLocations':  'location',
    'topics':          'topic',
    'labels':          'label',
    'detectedObjects': 'detected_object',
    'keywords':        'keyword',
    'faces':           'face',
}


def normalize_name(name: str) -> str:
    if not name:
        return ""
    return str(name).strip().lower()


def passes_seen_duration(entity: Dict[str, Any]) -> bool:
    """True if entity has no seenDuration field (topics/keywords) or seenDuration >= threshold."""
    seen = entity.get('seenDuration')
    return seen is None or float(seen) >= MIN_SEEN_DURATION


def get_entity_id(entity: Dict[str, Any], entity_type: str) -> str:
    # Faces: distinguish Unknown faces by numeric id
    if entity_type == 'face':
        name = entity.get('name', '')
        face_id = entity.get('id')
        if not name or name.lower() == 'unknown':
            return f"face_{face_id}" if face_id is not None else ""
        return f"face_{normalize_name(name)}"

    # Use referenceId or wikiDataId if available (people, brands, locations)
    ref_id = entity.get('referenceId') or entity.get('wikiDataId')
    if ref_id:
        return str(ref_id)

    # Fallback to normalized name
    name = entity.get('name')
    if name:
        return f"{entity_type}_{normalize_name(name)}"

    # For sentiments/emotions (kept for safety, not used as entities)
    key = entity.get('sentimentKey') or entity.get('type')
    if key:
        return f"{entity_type}_{normalize_name(key)}"

    return ""


def _get_summarized_insights(payloads: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the summarizedInsights dict from a raw_insights payload."""
    raw = payloads.get('raw_insights')
    if not raw or not isinstance(raw, dict):
        return {}
    return raw.get('summarizedInsights', {})


def get_normalized_mapping(clip_data: Dict[str, Dict[str, Any]], merge_threshold: float = 0.85) -> Dict[str, str]:
    """
    First pass over clip_data to collect all entities, then runs EntityNormalizer
    to compute the canonical mapping.
    """
    raw_entities = []

    for folder_name, payloads in clip_data.items():
        si = _get_summarized_insights(payloads)
        if not si:
            continue

        for category, entity_type in ENTITY_CATEGORIES.items():
            for entity in si.get(category, []):
                if not isinstance(entity, dict):
                    continue
                if not passes_seen_duration(entity):
                    continue
                raw_id = get_entity_id(entity, entity_type)
                if raw_id:
                    raw_entities.append({
                        'id':          raw_id,
                        'type':        entity_type,
                        'name':        entity.get('name', ''),
                        'description': entity.get('description', '')
                    })

        # OCR text entities — always included, no seenDuration field
        ocr = payloads.get('ocr')
        if ocr and isinstance(ocr, list):
            for item in ocr:
                text = item.get('text') if isinstance(item, dict) else None
                if text:
                    raw_entities.append({
                        'id':          f"text_{normalize_name(text)}",
                        'type':        'text',
                        'name':        text,
                        'description': 'Visual text from OCR'
                    })

    normalizer = EntityNormalizer(merge_threshold=merge_threshold)
    return normalizer.normalize_entities(raw_entities)


def filter_instances(instances: List[Dict[str, Any]], default_conf: float = 1.0) -> List[Dict[str, Any]]:
    valid = []
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        try:
            conf = float(inst.get('confidence', default_conf))
        except (ValueError, TypeError):
            conf = default_conf
        if conf > 0.7:
            valid.append(inst)
    return valid


def check_overlap(inst1: Dict[str, Any], inst2: Dict[str, Any]) -> bool:
    """Returns True if two instance dicts overlap in time."""
    start1 = inst1.get('startSeconds')
    if start1 is None: start1 = parse_time(inst1.get('startTime', ''))
    end1 = inst1.get('endSeconds')
    if end1 is None: end1 = parse_time(inst1.get('endTime', ''))

    start2 = inst2.get('startSeconds')
    if start2 is None: start2 = parse_time(inst2.get('startTime', ''))
    end2 = inst2.get('endSeconds')
    if end2 is None: end2 = parse_time(inst2.get('endTime', ''))

    return max(float(start1), float(start2)) < min(float(end1), float(end2))


def build_semantic_graph(G: nx.DiGraph, clip_data: Dict[str, Dict[str, Any]], merge_threshold: float = 0.85) -> nx.DiGraph:
    canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=merge_threshold)

    def get_canonical(entity, e_type):
        raw_id = get_entity_id(entity, e_type)
        return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

    for folder_name, payloads in clip_data.items():
        si = _get_summarized_insights(payloads)

        # 1. Add entities from summarizedInsights
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

                    if node_id not in G:
                        attrs = {
                            'node_class':  'Entity',
                            'type':        entity_type,
                            'name':        entity.get('name', ''),
                            'description': entity.get('description', '')
                        }
                        if entity_type == 'topic':
                            attrs['iabName']  = entity.get('iabName', '')
                            attrs['iptcName'] = entity.get('iptcName', '')
                        if entity_type == 'keyword':
                            attrs['isTranscript'] = entity.get('isTranscript', False)
                        G.add_node(node_id, **attrs)
                    else:
                        if not G.nodes[node_id].get('description') and entity.get('description'):
                            G.nodes[node_id]['description'] = entity.get('description')

                    # Hierarchy edges for topics
                    if entity_type == 'topic':
                        name = entity.get('name', '')
                        if '/' in name:
                            parts = name.split('/')
                            for i in range(1, len(parts)):
                                parent_name = '/'.join(parts[:i])
                                child_name  = '/'.join(parts[:i + 1])
                                parent_id   = f"topic_{normalize_name(parent_name)}"
                                child_id    = f"topic_{normalize_name(child_name)}"

                                if parent_id not in G:
                                    G.add_node(parent_id, node_class='Entity', type='topic', name=parent_name, description='')
                                if child_id not in G:
                                    G.add_node(child_id, node_class='Entity', type='topic', name=child_name, description='')

                                G.add_edge(child_id, parent_id, type='SUBCLASS_OF')

        # 2. OCR text entities
        ocr = payloads.get('ocr')
        if ocr and isinstance(ocr, list):
            for item in ocr:
                if not isinstance(item, dict):
                    continue
                text = item.get('text')
                if not text:
                    continue
                node_id = f"text_{normalize_name(text)}"
                if node_id not in G:
                    G.add_node(node_id, node_class='Entity', type='text', name=text, description='Visual text from OCR')

    return G


def get_relationship_type(type1: str, type2: str) -> str:
    types = tuple(sorted([type1, type2]))
    if types == ('person', 'topic'):    return 'discussion'
    if types == ('location', 'person'): return 'located_in'
    if types == ('person', 'person'):   return 'co_occurrence'
    return 'co_occurrence'


def build_cooccurrence_edges(G: nx.DiGraph, clip_data: Dict[str, Dict[str, Any]], merge_threshold: float = 0.85) -> nx.DiGraph:
    """
    Adds RELATED_TO edges between Entity nodes based on co-occurrence
    within the same clip (both appear in the same summarizedInsights or OCR payload).
    """
    canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=merge_threshold)

    def get_canonical(entity, e_type):
        raw_id = get_entity_id(entity, e_type)
        return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

    for folder_name, payloads in clip_data.items():
        si  = _get_summarized_insights(payloads)
        ocr = payloads.get('ocr')

        if not si and not ocr:
            continue

        entities_in_clip = set()

        if si:
            for category, entity_type in ENTITY_CATEGORIES.items():
                for entity in si.get(category, []):
                    if not isinstance(entity, dict):
                        continue
                    if not passes_seen_duration(entity):
                        continue
                    node_id = get_canonical(entity, entity_type)
                    if node_id and node_id in G:
                        entities_in_clip.add(node_id)

        if ocr and isinstance(ocr, list):
            for item in ocr:
                text = item.get('text') if isinstance(item, dict) else None
                if text:
                    node_id = f"text_{normalize_name(text)}"
                    if node_id in G:
                        entities_in_clip.add(node_id)

        # Build co-occurrence pairs — typed entities only (not OCR text)
        typed_entities = [
            e for e in entities_in_clip
            if not e.startswith('text_') and not e.startswith('face_')
        ]
        for e1, e2 in itertools.combinations(typed_entities, 2):
            for u, v in [(e1, e2), (e2, e1)]:
                if G.has_edge(u, v) and G[u][v].get('type') == 'RELATED_TO':
                    G[u][v]['weight'] = G[u][v].get('weight', 1) + 1
                else:
                    t1 = G.nodes[u].get('type', 'entity')
                    t2 = G.nodes[v].get('type', 'entity')
                    rel_type = get_relationship_type(t1, t2)
                    G.add_edge(u, v, type='RELATED_TO', relationship_type=rel_type, weight=1)

    return G


def build_bipartite_mapping(G: nx.DiGraph, clip_data: Dict[str, Dict[str, Any]], merge_threshold: float = 0.85) -> Dict[str, List[Dict[str, Any]]]:
    """
    Builds the Entity→Clip bipartite mapping as a plain dict.
    Returns: { entity_id: [ {clip_id, confidence, source, timestamp}, ... ] }
    """
    mapping: Dict[str, List[Dict[str, Any]]] = {}

    clip_intervals = [
        {
            'node_id':  node_id,
            'start':    attr.get('start', 0.0),
            'end':      attr.get('end', 0.0),
            'video_id': attr.get('video_id')
        }
        for node_id, attr in G.nodes(data=True)
        if attr.get('node_class') == 'Clip' and 'start' in attr and 'end' in attr
    ]

    canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=merge_threshold)

    def get_canonical(entity, e_type):
        raw_id = get_entity_id(entity, e_type)
        return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

    for folder_name, payloads in clip_data.items():
        si  = _get_summarized_insights(payloads)
        ocr = payloads.get('ocr')

        video_id   = folder_name
        rag_chunks = payloads.get('rag_chunks')
        if rag_chunks and isinstance(rag_chunks, list) and rag_chunks:
            video_id = rag_chunks[0].get('video_id', folder_name)

        def map_instances_to_dict(entity_node_id, instances_array, source_name, default_conf=1.0):
            if entity_node_id not in G:
                return
            valid_instances = filter_instances(instances_array, default_conf)
            for inst in valid_instances:
                i_start = inst.get('startSeconds') or inst.get('start') or parse_time(inst.get('startTime', ''))
                i_end   = inst.get('endSeconds')   or inst.get('end')   or parse_time(inst.get('endTime', ''))
                try:
                    i_start = float(i_start)
                    i_end   = float(i_end)
                except (ValueError, TypeError):
                    continue

                confidence = float(inst.get('confidence', default_conf))

                for clip in clip_intervals:
                    if clip['video_id'] != video_id:
                        continue
                    if max(i_start, clip['start']) < min(i_end, clip['end']):
                        if entity_node_id not in mapping:
                            mapping[entity_node_id] = []
                        mapping[entity_node_id].append({
                            'clip_id':    clip['node_id'],
                            'confidence': confidence,
                            'source':     source_name,
                            'timestamp':  i_start
                        })

        if si:
            for category, entity_type in ENTITY_CATEGORIES.items():
                for entity in si.get(category, []):
                    if not isinstance(entity, dict):
                        continue
                    if not passes_seen_duration(entity):
                        continue
                    node_id    = get_canonical(entity, entity_type)
                    arrays     = entity.get('instances', []) + entity.get('appearances', [])
                    map_instances_to_dict(node_id, arrays, source_name=category)

        if ocr and isinstance(ocr, list):
            for item in ocr:
                if not isinstance(item, dict):
                    continue
                text = item.get('text')
                if not text:
                    continue
                node_id_str = f"text_{normalize_name(text)}"
                arrays = item.get('instances', []) + item.get('appearances', [])
                map_instances_to_dict(node_id_str, arrays, source_name='OCR')

    return mapping


def print_semantic_graph_samples(G: nx.DiGraph, num_samples: int = 5):
    print(f"\n--- Layer 2 (Semantic) & Bipartite Samples ---")
    semantic_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('node_class') == 'Entity']
    print(f"Sample Semantic Nodes ({min(num_samples, len(semantic_nodes))}):")
    for node_id, attrs in semantic_nodes[:num_samples]:
        print(f"  Node ID: {node_id}")
        for k, v in attrs.items():
            val = str(v)
            if len(val) > 80: val = val[:77] + "..."
            print(f"    {k}: {val}")

    bipartite_edges = [(u, v, d) for u, v, d in G.edges(data=True) if 'confidence' in d]
    print(f"\nSample Bipartite Edges ({min(num_samples, len(bipartite_edges))}):")
    for u, v, attrs in bipartite_edges[:num_samples]:
        print(f"  Edge: {u} <-> {v} (Confidence: {attrs.get('confidence')})")

    related_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('type') == 'RELATED_TO']
    print(f"\nSample RELATED_TO Edges ({min(num_samples, len(related_edges))}):")
    for u, v, attrs in related_edges[:num_samples]:
        print(f"  Edge: {u} -> {v} (Weight: {attrs.get('weight')}, Rel: {attrs.get('relationship_type')})")

    subclass_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('type') == 'SUBCLASS_OF']
    print(f"\nSample SUBCLASS_OF Edges ({min(num_samples, len(subclass_edges))}):")
    for u, v, attrs in subclass_edges[:num_samples]:
        print(f"  Edge: {u} -> {v}")


if __name__ == "__main__":
    from data_loader import VideoDataLoader
    from temporal_clip_graph import build_temporal_clip_graph
    import sys

    logging.basicConfig(level=logging.INFO)
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"

    loader = VideoDataLoader(target_dir)
    data   = loader.load_data()

    graph = build_temporal_clip_graph(data)
    nodes_layer1 = graph.number_of_nodes()
    logger.info(f"Layer 1 Graph: {nodes_layer1} clip nodes, {graph.number_of_edges()} NEXT edges.")

    graph = build_semantic_graph(graph, data)
    logger.info(f"Layer 2 added {graph.number_of_nodes() - nodes_layer1} entity nodes.")

    graph = build_cooccurrence_edges(graph, data)
    related_count = sum(1 for _, _, d in graph.edges(data=True) if d.get('type') == 'RELATED_TO')
    logger.info(f"Added {related_count} RELATED_TO co-occurrence edges.")

    bipartite_dict = build_bipartite_mapping(graph, data)
    logger.info(f"Bipartite mapping dict: {len(bipartite_dict)} entities mapped to clips.")

    print_semantic_graph_samples(graph)
