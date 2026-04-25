import networkx as nx
from typing import Dict, Any, List, Tuple
import logging
from temporal_clip_graph import parse_time
import itertools
import os
import sys

# Add the parent directory to path to import entity_normalizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from entity_normalizer import EntityNormalizer

logger = logging.getLogger(__name__)

def normalize_name(name: str) -> str:
    if not name:
        return ""
    return str(name).strip().lower()

def get_entity_id(entity: Dict[str, Any], entity_type: str) -> str:
    # Use referenceId or wikiDataId if available
    ref_id = entity.get('referenceId') or entity.get('wikiDataId')
    if ref_id:
        return str(ref_id)
    
    # Fallback to normalized name
    name = entity.get('name')
    if name:
        return f"{entity_type}_{normalize_name(name)}"
        
    # For sentiments/emotions
    key = entity.get('sentimentKey') or entity.get('type')
    if key:
        return f"{entity_type}_{normalize_name(key)}"
        
    return ""

def get_normalized_mapping(clip_data: Dict[str, Dict[str, Any]], merge_threshold: float = 0.85) -> Dict[str, str]:
    """
    Does a first pass over clip_data to collect all entities,
    then runs EntityNormalizer to compute the canonical mapping.
    """
    raw_entities = []
    entity_categories = {
        'namedPeople': 'person', 'brands': 'brand', 'namedLocations': 'location',
        'topics': 'topic', 'labels': 'label', 'detectedObjects': 'detected_object'
    }

    for folder_name, payloads in clip_data.items():
        raw_insights = payloads.get('raw_insights')
        if not raw_insights: continue

        for category, entity_type in entity_categories.items():
            for entity in raw_insights.get(category, []):
                raw_id = get_entity_id(entity, entity_type)
                if raw_id:
                    raw_entities.append({
                        'id': raw_id,
                        'type': entity_type,
                        'name': entity.get('name', ''),
                        'description': entity.get('description', '')
                    })

        summarized = raw_insights.get('summarizedInsights', {})
        for sent in summarized.get('sentiments', []):
            raw_id = get_entity_id(sent, 'sentiment')
            if raw_id:
                raw_entities.append({'id': raw_id, 'type': 'sentiment', 'name': sent.get('sentimentKey', ''), 'description': ''})
        
        for em in summarized.get('emotions', []):
            raw_id = get_entity_id(em, 'emotion')
            if raw_id:
                raw_entities.append({'id': raw_id, 'type': 'emotion', 'name': em.get('type', ''), 'description': ''})

    normalizer = EntityNormalizer(merge_threshold=merge_threshold)
    return normalizer.normalize_entities(raw_entities)

def filter_instances(instances: List[Dict[str, Any]], default_conf: float = 1.0) -> List[Dict[str, Any]]:
    valid = []
    for inst in instances:
        if not isinstance(inst, dict): continue
        try:
            conf = float(inst.get('confidence', default_conf))
        except (ValueError, TypeError):
            conf = default_conf
            
        if conf > 0.7:
            valid.append(inst)
    return valid

def check_overlap(inst1: Dict[str, Any], inst2: Dict[str, Any]) -> bool:
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
    entity_categories = {
        'namedPeople': 'person',
        'brands': 'brand',
        'namedLocations': 'location',
        'topics': 'topic',
        'labels': 'label',
        'detectedObjects': 'detected_object'
    }

    canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=merge_threshold)

    def get_canonical(entity, e_type):
        raw_id = get_entity_id(entity, e_type)
        return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

    for folder_name, payloads in clip_data.items():
        raw_insights = payloads.get('raw_insights')
        if not raw_insights:
            continue
            
        # 1. Add normal entities
        for category, entity_type in entity_categories.items():
            entities = raw_insights.get(category, [])
            if not isinstance(entities, list): continue
                
            for entity in entities:
                if not isinstance(entity, dict): continue
                
                node_id = get_canonical(entity, entity_type)
                if not node_id: continue
                
                if node_id not in G:
                    attrs = {
                        'node_class': 'Entity',
                        'type': entity_type,
                        'name': entity.get('name', ''),
                        'description': entity.get('description', '')
                    }
                    if entity_type == 'topic':
                        attrs['iabName'] = entity.get('iabName', '')
                        attrs['iptcName'] = entity.get('iptcName', '')
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
                            child_name = '/'.join(parts[:i+1])
                            parent_id = f"topic_{normalize_name(parent_name)}"
                            child_id = f"topic_{normalize_name(child_name)}"
                            
                            if parent_id not in G:
                                G.add_node(parent_id, node_class='Entity', type='topic', name=parent_name, description='')
                            if child_id not in G:
                                G.add_node(child_id, node_class='Entity', type='topic', name=child_name, description='')
                                
                            G.add_edge(child_id, parent_id, type='SUBCLASS_OF')
                            
        # 2. Add Sentiments and Emotions
        summarized = raw_insights.get('summarizedInsights', {})
        
        sentiments = summarized.get('sentiments', [])
        for sent in sentiments:
            node_id = get_canonical(sent, 'sentiment')
            if node_id and node_id not in G:
                G.add_node(node_id, node_class='Entity', type='sentiment', name=sent.get('sentimentKey', ''), description='')
                
        emotions = summarized.get('emotions', [])
        for em in emotions:
            node_id = get_canonical(em, 'emotion')
            if node_id and node_id not in G:
                G.add_node(node_id, node_class='Entity', type='emotion', name=em.get('type', ''), description='')
                
        # 3. Person -> Emotion/Sentiment edges
        people = raw_insights.get('namedPeople', [])
        for person in people:
            p_id = get_canonical(person, 'person')
            if not p_id or p_id not in G: continue
            
            p_instances = filter_instances(person.get('instances', []) + person.get('appearances', []))
            if not p_instances: continue
            
            # Check overlap with sentiments
            for sent in sentiments:
                s_id = get_canonical(sent, 'sentiment')
                if not s_id or s_id not in G: continue
                s_instances = filter_instances(sent.get('appearances', []), default_conf=1.0)
                
                for p_inst in p_instances:
                    for s_inst in s_instances:
                        if check_overlap(p_inst, s_inst):
                            G.add_edge(p_id, s_id, type='ASSOCIATED_WITH')
                            break
            
            # Check overlap with emotions
            for em in emotions:
                e_id = get_canonical(em, 'emotion')
                if not e_id or e_id not in G: continue
                e_instances = filter_instances(em.get('appearances', []), default_conf=1.0)
                
                for p_inst in p_instances:
                    for e_inst in e_instances:
                        if check_overlap(p_inst, e_inst):
                            G.add_edge(p_id, e_id, type='EXPRESSED')
                            break

    return G

def get_relationship_type(type1: str, type2: str) -> str:
    types = tuple(sorted([type1, type2]))
    if types == ('person', 'topic'): return 'discussion'
    if types == ('location', 'person'): return 'located_in'
    if types == ('person', 'person'): return 'co_occurrence'
    if 'sentiment' in types or 'emotion' in types: return 'emotional_context'
    return 'co_occurrence'

def build_bipartite_mapping(G: nx.DiGraph, clip_data: Dict[str, Dict[str, Any]], merge_threshold: float = 0.85) -> nx.DiGraph:
    clip_intervals = []
    for node_id, attr in G.nodes(data=True):
        if attr.get('node_class') == 'Clip' and 'start' in attr and 'end' in attr:
            clip_intervals.append({
                'node_id': node_id,
                'start': attr.get('start', 0.0),
                'end': attr.get('end', 0.0),
                'video_id': attr.get('video_id')
            })

    entity_categories = {
        'namedPeople': 'person',
        'brands': 'brand',
        'namedLocations': 'location',
        'topics': 'topic',
        'labels': 'label',
        'detectedObjects': 'detected_object'
    }

    canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=merge_threshold)

    def get_canonical(entity, e_type):
        raw_id = get_entity_id(entity, e_type)
        return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

    for folder_name, payloads in clip_data.items():
        raw_insights = payloads.get('raw_insights')
        ocr = payloads.get('ocr')
        
        video_id = folder_name
        rag_chunks = payloads.get('rag_chunks')
        if rag_chunks and isinstance(rag_chunks, list) and len(rag_chunks) > 0:
            video_id = rag_chunks[0].get('video_id', folder_name)

        def map_instances_to_clips(entity_node_id, instances_array, default_conf=1.0):
            if entity_node_id not in G:
                return
            
            valid_instances = filter_instances(instances_array, default_conf)
                
            for inst in valid_instances:
                i_start = inst.get('startSeconds')
                if i_start is None: i_start = inst.get('start')
                if i_start is None: i_start = inst.get('startTime', '')
                i_start = parse_time(str(i_start))
                
                i_end = inst.get('endSeconds')
                if i_end is None: i_end = inst.get('end')
                if i_end is None: i_end = inst.get('endTime', '')
                i_end = parse_time(str(i_end))
                
                confidence = float(inst.get('confidence', default_conf))
                
                for clip in clip_intervals:
                    if clip['video_id'] != video_id:
                        continue
                        
                    c_start = clip['start']
                    c_end = clip['end']
                    
                    if max(float(i_start), c_start) < min(float(i_end), c_end):
                        G.add_edge(entity_node_id, clip['node_id'], confidence=confidence)
                        G.add_edge(clip['node_id'], entity_node_id, confidence=confidence)

        if raw_insights:
            # Regular entities
            for category, entity_type in entity_categories.items():
                for entity in raw_insights.get(category, []):
                    node_id = get_canonical(entity, entity_type)
                    arrays_to_check = entity.get('instances', []) + entity.get('appearances', [])
                    map_instances_to_clips(node_id, arrays_to_check)
            
            # Sentiments and Emotions
            summarized = raw_insights.get('summarizedInsights', {})
            for sent in summarized.get('sentiments', []):
                node_id = get_canonical(sent, 'sentiment')
                map_instances_to_clips(node_id, sent.get('appearances', []), default_conf=1.0)
            for em in summarized.get('emotions', []):
                node_id = get_canonical(em, 'emotion')
                map_instances_to_clips(node_id, em.get('appearances', []), default_conf=1.0)

        # 3. Process ocr.json
        if ocr and isinstance(ocr, list):
            for item in ocr:
                if not isinstance(item, dict): continue
                text = item.get('text')
                if not text: continue
                
                node_id_str = f"text_{normalize_name(text)}"
                if node_id_str not in G:
                    G.add_node(node_id_str, node_class='Entity', type='text', name=text, description='Visual text from OCR')
                    
                arrays_to_check = item.get('instances', []) + item.get('appearances', [])
                map_instances_to_clips(node_id_str, arrays_to_check)
                
    # Create Clip-Based Co-occurrence Edges (RELATED_TO)
    # We find all Entity nodes connected to a Clip node.
    clip_nodes = [n for n, d in G.nodes(data=True) if d.get('node_class') == 'Clip']
    
    for clip_node in clip_nodes:
        # Entities are neighbors of the Clip node
        entities_in_clip = [n for n in G.neighbors(clip_node) if G.nodes[n].get('node_class') == 'Entity']
        
        # Iterate over all pairs of entities in this clip
        for e1, e2 in itertools.combinations(entities_in_clip, 2):
            if G.has_edge(e1, e2) and G[e1][e2].get('type') == 'RELATED_TO':
                G[e1][e2]['weight'] += 1
            else:
                t1 = G.nodes[e1].get('type', 'entity')
                t2 = G.nodes[e2].get('type', 'entity')
                rel_type = get_relationship_type(t1, t2)
                G.add_edge(e1, e2, type='RELATED_TO', relationship_type=rel_type, weight=1)
                
            # DiGraph: add both directions
            if G.has_edge(e2, e1) and G[e2][e1].get('type') == 'RELATED_TO':
                G[e2][e1]['weight'] += 1
            else:
                t1 = G.nodes[e1].get('type', 'entity')
                t2 = G.nodes[e2].get('type', 'entity')
                rel_type = get_relationship_type(t1, t2)
                G.add_edge(e2, e1, type='RELATED_TO', relationship_type=rel_type, weight=1)

    return G

def build_bipartite_mapping_dict(G: nx.DiGraph, clip_data: Dict[str, Dict[str, Any]], merge_threshold: float = 0.85) -> Dict[str, List[Dict[str, Any]]]:
    """
    Creates a Bipartite Mapping dictionary linking Semantic Entity IDs to Clip IDs.
    This allows for O(1) lookup: Given an Entity ID, returns all overlapping Clip IDs
    with their associated metadata.
    """
    mapping = {}
    
    clip_intervals = []
    for node_id, attr in G.nodes(data=True):
        if attr.get('node_class') == 'Clip' and 'start' in attr and 'end' in attr:
            clip_intervals.append({
                'node_id': node_id,
                'start': attr.get('start', 0.0),
                'end': attr.get('end', 0.0),
                'video_id': attr.get('video_id')
            })

    entity_categories = {
        'namedPeople': 'person',
        'brands': 'brand',
        'namedLocations': 'location',
        'topics': 'topic',
        'labels': 'label',
        'detectedObjects': 'detected_object'
    }

    canonical_mapping = get_normalized_mapping(clip_data, merge_threshold=merge_threshold)

    def get_canonical(entity, e_type):
        raw_id = get_entity_id(entity, e_type)
        return canonical_mapping.get(raw_id, raw_id) if raw_id else ""

    for folder_name, payloads in clip_data.items():
        raw_insights = payloads.get('raw_insights')
        ocr = payloads.get('ocr')
        
        video_id = folder_name
        rag_chunks = payloads.get('rag_chunks')
        if rag_chunks and isinstance(rag_chunks, list) and len(rag_chunks) > 0:
            video_id = rag_chunks[0].get('video_id', folder_name)

        def map_instances_to_dict(entity_node_id, instances_array, source_name, default_conf=1.0):
            if entity_node_id not in G:
                return
            
            valid_instances = filter_instances(instances_array, default_conf)
                
            for inst in valid_instances:
                i_start = inst.get('startSeconds')
                if i_start is None: i_start = inst.get('start')
                if i_start is None: i_start = inst.get('startTime', '')
                i_start = parse_time(str(i_start))
                
                i_end = inst.get('endSeconds')
                if i_end is None: i_end = inst.get('end')
                if i_end is None: i_end = inst.get('endTime', '')
                i_end = parse_time(str(i_end))
                
                confidence = float(inst.get('confidence', default_conf))
                
                for clip in clip_intervals:
                    if clip['video_id'] != video_id:
                        continue
                        
                    c_start = clip['start']
                    c_end = clip['end']
                    
                    if max(float(i_start), c_start) < min(float(i_end), c_end):
                        if entity_node_id not in mapping:
                            mapping[entity_node_id] = []
                        mapping[entity_node_id].append({
                            'clip_id': clip['node_id'],
                            'confidence': confidence,
                            'source': source_name,
                            'timestamp': float(i_start)
                        })

        if raw_insights:
            # Regular entities
            for category, entity_type in entity_categories.items():
                for entity in raw_insights.get(category, []):
                    node_id = get_canonical(entity, entity_type)
                    arrays_to_check = entity.get('instances', []) + entity.get('appearances', [])
                    map_instances_to_dict(node_id, arrays_to_check, source_name=category)
            
            # Sentiments and Emotions
            summarized = raw_insights.get('summarizedInsights', {})
            for sent in summarized.get('sentiments', []):
                node_id = get_canonical(sent, 'sentiment')
                map_instances_to_dict(node_id, sent.get('appearances', []), source_name='sentiments', default_conf=1.0)
            for em in summarized.get('emotions', []):
                node_id = get_canonical(em, 'emotion')
                map_instances_to_dict(node_id, em.get('appearances', []), source_name='emotions', default_conf=1.0)

        # Process ocr.json
        if ocr and isinstance(ocr, list):
            for item in ocr:
                if not isinstance(item, dict): continue
                text = item.get('text')
                if not text: continue
                
                node_id_str = f"text_{normalize_name(text)}"
                arrays_to_check = item.get('instances', []) + item.get('appearances', [])
                map_instances_to_dict(node_id_str, arrays_to_check, source_name='OCR')

    return mapping

import math

def build_clip_to_clip_edges(G: nx.DiGraph, significance_threshold: float = 0.5) -> nx.DiGraph:
    """
    Builds SHARES_ENTITY edges between Clip nodes based on shared Entity nodes.
    Uses inverse frequency (IDF) weighting to penalize ubiquitous entities.
    """
    clip_nodes = [n for n, d in G.nodes(data=True) if d.get('node_class') == 'Clip']
    N = len(clip_nodes)
    if N == 0:
        return G

    # Calculate Document Frequency (df) for each entity
    entity_df = {}
    clip_entities = {} # clip -> dict of {entity: confidence}

    for clip in clip_nodes:
        clip_entities[clip] = {}
        for neighbor in G.neighbors(clip):
            if G.nodes[neighbor].get('node_class') == 'Entity':
                conf = G[clip][neighbor].get('confidence', 1.0)
                clip_entities[clip][neighbor] = conf
                entity_df[neighbor] = entity_df.get(neighbor, 0) + 1

    # Calculate IDF
    entity_idf = {}
    for e, df in entity_df.items():
        # log(N / df)
        entity_idf[e] = math.log(N / max(1, df))

    # Build edges
    edges_added = 0
    for c1, c2 in itertools.combinations(clip_nodes, 2):
        shared = set(clip_entities[c1].keys()).intersection(set(clip_entities[c2].keys()))
        if not shared:
            continue

        score = 0.0
        for e in shared:
            conf1 = clip_entities[c1][e]
            conf2 = clip_entities[c2][e]
            avg_conf = (conf1 + conf2) / 2.0
            # TF-IDF style weighting
            score += avg_conf * entity_idf[e]

        if score >= significance_threshold:
            # Undirected semantic edge, add both directions for DiGraph traversal
            G.add_edge(c1, c2, type='SHARES_ENTITY', weight=score)
            G.add_edge(c2, c1, type='SHARES_ENTITY', weight=score)
            edges_added += 1

    logger.info(f"Clip-to-Clip Connectivity: Added {edges_added} SHARES_ENTITY edges (threshold={significance_threshold}).")
    return G

def print_semantic_graph_samples(G: nx.DiGraph, num_samples: int = 5):
    print(f"\n--- Layer 2 (Semantic) & Bipartite Samples ---")
    
    semantic_nodes = [ (n, d) for n, d in G.nodes(data=True) if d.get('node_class') == 'Entity' ]
    sample_nodes = semantic_nodes[:num_samples]
    print(f"Sample Semantic Nodes ({len(sample_nodes)}):")
    for node_id, attrs in sample_nodes:
        print(f"  Node ID: {node_id}")
        for k, v in attrs.items():
            val = str(v)
            if len(val) > 80: val = val[:77] + "..."
            print(f"    {k}: {val}")
            
    # Sample Bipartite Edges
    bipartite_edges = [ (u, v, d) for u, v, d in G.edges(data=True) if 'confidence' in d ]
    print(f"\nSample Bipartite Edges ({min(num_samples, len(bipartite_edges))}):")
    for u, v, attrs in bipartite_edges[:num_samples]:
        print(f"  Edge: {u} <-> {v} (Confidence: {attrs.get('confidence')})")

    # Sample Co-occurrence Edges
    related_edges = [ (u, v, d) for u, v, d in G.edges(data=True) if d.get('type') == 'RELATED_TO' ]
    print(f"\nSample RELATED_TO Edges ({min(num_samples, len(related_edges))}):")
    for u, v, attrs in related_edges[:num_samples]:
        print(f"  Edge: {u} -> {v} (Weight: {attrs.get('weight')}, Rel: {attrs.get('relationship_type')})")
        
    # Sample Hierarchy Edges
    subclass_edges = [ (u, v, d) for u, v, d in G.edges(data=True) if d.get('type') == 'SUBCLASS_OF' ]
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
    data = loader.load_data()
    
    graph = build_temporal_clip_graph(data)
    nodes_layer1 = graph.number_of_nodes()
    logger.info(f"Layer 1 Graph generated with {nodes_layer1} nodes and {graph.number_of_edges()} edges.")
    
    graph = build_semantic_graph(graph, data)
    nodes_layer2 = graph.number_of_nodes()
    logger.info(f"Layer 2 Semantic Graph added. Total nodes: {nodes_layer2} (+{nodes_layer2 - nodes_layer1} entities).")
    
    edges_before = graph.number_of_edges()
    graph = build_bipartite_mapping(graph, data)
    edges_after = graph.number_of_edges()
    logger.info(f"Bipartite & Co-occurrence added. New edges: {edges_after - edges_before}.")
    
    graph = build_clip_to_clip_edges(graph, significance_threshold=0.5)

    print_semantic_graph_samples(graph)
    
    mapping_dict = build_bipartite_mapping_dict(graph, data)
    logger.info(f"Bipartite mapping dict generated with {len(mapping_dict)} entities mapped to clips.")
    
    # Print a small sample from the dictionary
    print("\n--- Bipartite Mapping Dictionary Sample ---")
    sample_items = list(mapping_dict.items())[:3]
    for entity_id, clips in sample_items:
        print(f"Entity: {entity_id}")
        for c in clips[:2]:  # Show max 2 clips per entity in sample
            print(f"  - Clip: {c['clip_id']}, Conf: {c['confidence']}, Source: {c['source']}, Timestamp: {c['timestamp']}")
        if len(clips) > 2:
            print(f"  - ... ({len(clips) - 2} more clips)")

