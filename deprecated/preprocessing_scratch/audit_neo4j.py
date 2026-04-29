"""
audit_neo4j.py
~~~~~~~~~~~~~~
Verifies that the two Neo4j graph instances (Clip Graph + Entity Graph)
match the schema/structure defined in the code:

Clip Graph (Instance 1)
  - Node label  : Clip
  - Properties  : id, video_id, start, end, transcript, ocr, keywords,
                  summary, average_sentiment, emotion, speaker_ids
  - Constraint  : Clip.id IS UNIQUE
  - Index       : Clip.video_id
  - Edges       : NEXT  (Clip)-[:NEXT]->(Clip)
  - NO other node labels or edge types expected

Entity Graph (Instance 2)
  - Node label  : Entity
  - Properties  : id, type, name, description, (iabName, iptcName for topics)
  - Constraint  : Entity.id IS UNIQUE
  - Index       : Entity.type
  - Edges       : SUBCLASS_OF  (Entity)-[:SUBCLASS_OF]->(Entity)  [topic hierarchy]
                  RELATED_TO   (Entity)-[:RELATED_TO]->(Entity)    [co-occurrence]
  - RELATED_TO properties: weight, relationship_type
  - NO APPEARS_IN / SHARES_ENTITY / mapping edges expected
  - Entity types: person, brand, location, topic, label, detected_object, text

Run from: /mnt/MIG_store/Datasets/blending/madhav/VRAG/video_rag_preprocessing/
  python scratch/audit_neo4j.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_store.connection import MultiGraphManager

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Expected schema ──────────────────────────────────────────────────────────

CLIP_EXPECTED_PROPS = {
    "id", "video_id", "start", "end", "transcript",
    "ocr", "keywords", "summary", "average_sentiment",
    "emotion", "speaker_ids"
}
CLIP_EXPECTED_REL_TYPES = {"NEXT"}
CLIP_EXPECTED_NODE_LABELS = {"Clip"}

ENTITY_EXPECTED_PROPS = {
    "id", "type", "name", "description", "iabName", "iptcName"
}
ENTITY_EXPECTED_REL_TYPES = {"SUBCLASS_OF", "RELATED_TO"}
ENTITY_EXPECTED_NODE_LABELS = {"Entity"}
ENTITY_VALID_TYPES = {"person", "brand", "location", "topic", "label", "detected_object", "text"}
RELATED_TO_EXPECTED_PROPS = {"weight", "relationship_type"}

FORBIDDEN_REL_TYPES = {"APPEARS_IN", "SHARES_ENTITY", "ASSOCIATED_WITH", "EXPRESSED"}

issues = []

def log_issue(graph: str, msg: str):
    full = f"[{graph}] ⚠ {msg}"
    issues.append(full)
    logger.warning(full)

def log_ok(msg: str):
    logger.info(f"  ✓ {msg}")


# ── Clip Graph audit ─────────────────────────────────────────────────────────

def audit_clip_graph(conn):
    logger.info("\n══════════════════════════════════════════")
    logger.info("  CLIP GRAPH AUDIT")
    logger.info("══════════════════════════════════════════")

    # 1. Node label inventory
    result = conn.execute_query("MATCH (n) RETURN labels(n) AS labels, count(*) AS cnt")
    logger.info("\n[1] Node label counts:")
    total_clips = 0
    for row in result:
        labs = set(row["labels"])
        cnt  = row["cnt"]
        logger.info(f"    {labs} → {cnt}")
        if labs != CLIP_EXPECTED_NODE_LABELS:
            log_issue("ClipGraph", f"Unexpected node label(s) found: {labs} (count={cnt})")
        else:
            total_clips += cnt
    log_ok(f"Total Clip nodes: {total_clips}")

    # 2. Relationship type inventory
    result = conn.execute_query("MATCH ()-[r]->() RETURN type(r) AS rtype, count(*) AS cnt")
    logger.info("\n[2] Relationship type counts:")
    if not result:
        log_issue("ClipGraph", "No relationships found at all!")
    for row in result:
        rtype = row["rtype"]
        cnt   = row["cnt"]
        logger.info(f"    {rtype} → {cnt}")
        if rtype not in CLIP_EXPECTED_REL_TYPES:
            log_issue("ClipGraph", f"Unexpected relationship type: {rtype} (count={cnt})")
        if rtype in FORBIDDEN_REL_TYPES:
            log_issue("ClipGraph", f"FORBIDDEN relationship type present: {rtype}")

    # 3. Property completeness on a sample of Clip nodes
    result = conn.execute_query(
        "MATCH (c:Clip) RETURN properties(c) AS props LIMIT 50"
    )
    logger.info(f"\n[3] Clip property audit (sample of {len(result)} nodes):")
    missing_props_counts = {p: 0 for p in CLIP_EXPECTED_PROPS}
    extra_props_found = set()
    null_id_count = 0
    for row in result:
        props = set(row["props"].keys())
        missing = CLIP_EXPECTED_PROPS - props
        extra   = props - CLIP_EXPECTED_PROPS
        for m in missing:
            missing_props_counts[m] += 1
        extra_props_found.update(extra)
        if row["props"].get("id") is None:
            null_id_count += 1

    for prop, cnt in missing_props_counts.items():
        if cnt > 0:
            log_issue("ClipGraph", f"Property '{prop}' missing in {cnt}/50 sampled Clip nodes")
        else:
            log_ok(f"Property '{prop}' present in all sampled nodes")
    if extra_props_found:
        log_issue("ClipGraph", f"Extra unexpected properties on Clip nodes: {extra_props_found}")
    if null_id_count:
        log_issue("ClipGraph", f"{null_id_count} Clip nodes with null/missing 'id'")

    # 4. NEXT edge integrity — are both endpoints always Clip nodes?
    result = conn.execute_query(
        """MATCH (a)-[r:NEXT]->(b)
           RETURN
             labels(a) AS src_labels,
             labels(b) AS tgt_labels,
             count(*) AS cnt
           LIMIT 20"""
    )
    logger.info("\n[4] NEXT edge endpoint labels:")
    for row in result:
        src = set(row["src_labels"])
        tgt = set(row["tgt_labels"])
        cnt = row["cnt"]
        logger.info(f"    ({src})-[:NEXT]->({tgt})  ×{cnt}")
        if src != {"Clip"} or tgt != {"Clip"}:
            log_issue("ClipGraph", f"NEXT edge between unexpected labels: ({src})->({tgt})")

    # 5. Uniqueness constraint check (indirectly: any duplicate ids?)
    result = conn.execute_query(
        "MATCH (c:Clip) WITH c.id AS id, count(*) AS cnt WHERE cnt > 1 RETURN id, cnt LIMIT 10"
    )
    logger.info(f"\n[5] Duplicate Clip id check:")
    if result:
        for row in result:
            log_issue("ClipGraph", f"Duplicate Clip id='{row['id']}' appears {row['cnt']} times")
    else:
        log_ok("No duplicate Clip.id found")

    # 6. Nodes missing 'id' entirely
    result = conn.execute_query("MATCH (c:Clip) WHERE c.id IS NULL RETURN count(*) AS cnt")
    cnt = result[0]["cnt"] if result else 0
    if cnt > 0:
        log_issue("ClipGraph", f"{cnt} Clip nodes have no 'id' property")
    else:
        log_ok("All Clip nodes have an 'id'")

    # 7. Connectivity: any Clip nodes with zero NEXT edges (isolated)?
    result = conn.execute_query(
        """MATCH (c:Clip)
           WHERE NOT (c)-[:NEXT]-() 
           RETURN count(*) AS cnt"""
    )
    isolated = result[0]["cnt"] if result else 0
    logger.info(f"\n[7] Isolated Clip nodes (no NEXT edge in or out): {isolated}")
    if isolated > 0:
        # Could be legitimate (single-clip video), flag as info only
        logger.info(f"    → {isolated} clip(s) have no NEXT edge (may be single-segment videos)")

    # 8. id format check — expected: {video_id}_{start:.2f}_{end:.2f}
    result = conn.execute_query("MATCH (c:Clip) RETURN c.id AS id LIMIT 100")
    bad_format = []
    for row in result:
        cid = row["id"] or ""
        parts = cid.rsplit("_", 2)
        if len(parts) < 3:
            bad_format.append(cid)
        else:
            try:
                float(parts[-1])
                float(parts[-2])
            except ValueError:
                bad_format.append(cid)
    logger.info(f"\n[8] Clip id format check (sample 100):")
    if bad_format:
        log_issue("ClipGraph", f"{len(bad_format)} Clip ids don't match {{video_id}}_{{start}}_{{end}} pattern: {bad_format[:5]}")
    else:
        log_ok("All sampled Clip ids match expected {video_id}_{start:.2f}_{end:.2f} format")


# ── Entity Graph audit ───────────────────────────────────────────────────────

def audit_entity_graph(conn):
    logger.info("\n══════════════════════════════════════════")
    logger.info("  ENTITY GRAPH AUDIT")
    logger.info("══════════════════════════════════════════")

    # 1. Node label inventory
    result = conn.execute_query("MATCH (n) RETURN labels(n) AS labels, count(*) AS cnt")
    logger.info("\n[1] Node label counts:")
    total_entities = 0
    for row in result:
        labs = set(row["labels"])
        cnt  = row["cnt"]
        logger.info(f"    {labs} → {cnt}")
        if labs != ENTITY_EXPECTED_NODE_LABELS:
            log_issue("EntityGraph", f"Unexpected node label(s): {labs} (count={cnt})")
        else:
            total_entities += cnt
    log_ok(f"Total Entity nodes: {total_entities}")

    # 2. Relationship type inventory
    result = conn.execute_query("MATCH ()-[r]->() RETURN type(r) AS rtype, count(*) AS cnt")
    logger.info("\n[2] Relationship type counts:")
    if not result:
        log_issue("EntityGraph", "No relationships found at all!")
    for row in result:
        rtype = row["rtype"]
        cnt   = row["cnt"]
        logger.info(f"    {rtype} → {cnt}")
        if rtype not in ENTITY_EXPECTED_REL_TYPES:
            log_issue("EntityGraph", f"Unexpected relationship type: {rtype} (count={cnt})")
        if rtype in FORBIDDEN_REL_TYPES:
            log_issue("EntityGraph", f"FORBIDDEN relationship type present: {rtype}")

    # 3. Entity type distribution
    result = conn.execute_query(
        "MATCH (e:Entity) RETURN e.type AS etype, count(*) AS cnt ORDER BY cnt DESC"
    )
    logger.info("\n[3] Entity type distribution:")
    for row in result:
        etype = row["etype"]
        cnt   = row["cnt"]
        logger.info(f"    {etype} → {cnt}")
        if etype not in ENTITY_VALID_TYPES:
            log_issue("EntityGraph", f"Unknown entity type '{etype}' found ({cnt} nodes)")

    # 4. Property completeness on Entity nodes
    result = conn.execute_query(
        "MATCH (e:Entity) RETURN properties(e) AS props LIMIT 50"
    )
    logger.info(f"\n[4] Entity property audit (sample of {len(result)} nodes):")
    required = {"id", "type", "name", "description"}
    missing_counts = {p: 0 for p in required}
    extra_props = set()
    for row in result:
        props = set(row["props"].keys())
        for p in required:
            if p not in props:
                missing_counts[p] += 1
        extra = props - ENTITY_EXPECTED_PROPS
        extra_props.update(extra)

    for prop, cnt in missing_counts.items():
        if cnt > 0:
            log_issue("EntityGraph", f"Required property '{prop}' missing in {cnt}/50 sampled Entity nodes")
        else:
            log_ok(f"Property '{prop}' present in all sampled nodes")
    if extra_props:
        log_issue("EntityGraph", f"Unexpected extra properties on Entity nodes: {extra_props}")

    # 5. RELATED_TO edge property check
    result = conn.execute_query(
        "MATCH ()-[r:RELATED_TO]->() RETURN properties(r) AS props LIMIT 30"
    )
    logger.info(f"\n[5] RELATED_TO edge properties (sample {len(result)}):")
    missing_rel_props = {p: 0 for p in RELATED_TO_EXPECTED_PROPS}
    for row in result:
        props = set(row["props"].keys())
        for p in RELATED_TO_EXPECTED_PROPS:
            if p not in props:
                missing_rel_props[p] += 1
    for prop, cnt in missing_rel_props.items():
        if cnt > 0:
            log_issue("EntityGraph", f"RELATED_TO edge missing '{prop}' in {cnt}/30 sampled edges")
        else:
            log_ok(f"RELATED_TO edges all have '{prop}'")

    # 6. SUBCLASS_OF endpoint check — must be Entity→Entity (topics only)
    result = conn.execute_query(
        """MATCH (a)-[:SUBCLASS_OF]->(b)
           RETURN labels(a) AS src, labels(b) AS tgt,
                  a.type AS src_type, b.type AS tgt_type,
                  count(*) AS cnt
           LIMIT 20"""
    )
    logger.info(f"\n[6] SUBCLASS_OF edge endpoint audit:")
    if not result:
        logger.info("    No SUBCLASS_OF edges found")
    for row in result:
        src = set(row["src"])
        tgt = set(row["tgt"])
        cnt = row["cnt"]
        logger.info(f"    ({src}[type={row['src_type']}])-[:SUBCLASS_OF]->({tgt}[type={row['tgt_type']}])  ×{cnt}")
        if src != {"Entity"} or tgt != {"Entity"}:
            log_issue("EntityGraph", f"SUBCLASS_OF between non-Entity labels: ({src})->({tgt})")
        if row["src_type"] != "topic" or row["tgt_type"] != "topic":
            log_issue("EntityGraph", f"SUBCLASS_OF between non-topic entities: {row['src_type']} → {row['tgt_type']}")

    # 7. Duplicate Entity id check
    result = conn.execute_query(
        "MATCH (e:Entity) WITH e.id AS id, count(*) AS cnt WHERE cnt > 1 RETURN id, cnt LIMIT 10"
    )
    logger.info(f"\n[7] Duplicate Entity id check:")
    if result:
        for row in result:
            log_issue("EntityGraph", f"Duplicate Entity id='{row['id']}' appears {row['cnt']} times")
    else:
        log_ok("No duplicate Entity.id found")

    # 8. Entities missing 'id'
    result = conn.execute_query("MATCH (e:Entity) WHERE e.id IS NULL RETURN count(*) AS cnt")
    cnt = result[0]["cnt"] if result else 0
    if cnt > 0:
        log_issue("EntityGraph", f"{cnt} Entity nodes have no 'id' property")
    else:
        log_ok("All Entity nodes have an 'id'")

    # 9. RELATED_TO edge endpoints — should only be Entity→Entity
    result = conn.execute_query(
        """MATCH (a)-[:RELATED_TO]->(b)
           RETURN labels(a) AS src, labels(b) AS tgt, count(*) AS cnt
           LIMIT 10"""
    )
    logger.info(f"\n[9] RELATED_TO endpoint labels:")
    for row in result:
        src = set(row["src"])
        tgt = set(row["tgt"])
        if src != {"Entity"} or tgt != {"Entity"}:
            log_issue("EntityGraph", f"RELATED_TO between non-Entity labels: ({src})->({tgt})")
        else:
            log_ok(f"({src})-[:RELATED_TO]->({tgt})  ×{row['cnt']}")

    # 10. Cross-graph contamination: any Clip nodes in Entity graph?
    result = conn.execute_query("MATCH (c:Clip) RETURN count(*) AS cnt")
    cnt = result[0]["cnt"] if result else 0
    if cnt > 0:
        log_issue("EntityGraph", f"Entity graph contains {cnt} Clip nodes! (cross-contamination)")
    else:
        log_ok("No Clip nodes in Entity graph")

    # 11. Summary stats
    result = conn.execute_query("MATCH (e:Entity) RETURN count(*) AS cnt")
    e_cnt = result[0]["cnt"] if result else 0
    result = conn.execute_query("MATCH ()-[r:RELATED_TO]->() RETURN count(*) AS cnt")
    rt_cnt = result[0]["cnt"] if result else 0
    result = conn.execute_query("MATCH ()-[r:SUBCLASS_OF]->() RETURN count(*) AS cnt")
    so_cnt = result[0]["cnt"] if result else 0
    logger.info(f"\n[11] Summary:")
    logger.info(f"    Entity nodes      : {e_cnt}")
    logger.info(f"    RELATED_TO edges  : {rt_cnt}")
    logger.info(f"    SUBCLASS_OF edges : {so_cnt}")


# ── Cross-graph checks ───────────────────────────────────────────────────────

def audit_cross_graph(clip_conn, entity_conn):
    logger.info("\n══════════════════════════════════════════")
    logger.info("  CROSS-GRAPH CHECKS")
    logger.info("══════════════════════════════════════════")

    # 1. Entity graph shouldn't have Clip nodes and vice versa
    result = entity_conn.execute_query("MATCH (c:Clip) RETURN count(*) AS cnt")
    cnt = result[0]["cnt"] if result else 0
    if cnt > 0:
        log_issue("Cross", f"Entity graph has {cnt} Clip nodes (should be zero)")
    else:
        log_ok("Entity graph has zero Clip nodes")

    result = clip_conn.execute_query("MATCH (e:Entity) RETURN count(*) AS cnt")
    cnt = result[0]["cnt"] if result else 0
    if cnt > 0:
        log_issue("Cross", f"Clip graph has {cnt} Entity nodes (should be zero)")
    else:
        log_ok("Clip graph has zero Entity nodes")

    # 2. Count unique video_ids in clip graph
    result = clip_conn.execute_query(
        "MATCH (c:Clip) RETURN c.video_id AS vid, count(*) AS clips ORDER BY clips DESC LIMIT 15"
    )
    logger.info(f"\n[2] Video ID → Clip count (top 15):")
    for row in result:
        logger.info(f"    {row['vid']}  →  {row['clips']} clips")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("Starting Neo4j structure audit...")

    try:
        with MultiGraphManager() as manager:
            audit_clip_graph(manager.clip_graph)
            audit_entity_graph(manager.entity_graph)
            audit_cross_graph(manager.clip_graph, manager.entity_graph)
    except Exception as e:
        logger.error(f"Fatal error during audit: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n══════════════════════════════════════════")
    logger.info("  AUDIT COMPLETE")
    logger.info("══════════════════════════════════════════")
    if issues:
        logger.warning(f"\n⚠  {len(issues)} issue(s) found:\n")
        for i, iss in enumerate(issues, 1):
            logger.warning(f"  {i:02d}. {iss}")
    else:
        logger.info("\n✅  No structural deviations found. Both graphs match expected schema.")


if __name__ == "__main__":
    main()
