# E2E Test Results

### Query: Who is Barack Obama?
```json
{
  "query_type": "informational",
  "entities": [
    {
      "name": "Barack Obama",
      "type": "person",
      "resolved_entity_id": "person_barack_obama"
    }
  ],
  "actions": [
    "lookup"
  ],
  "temporal_constraints": {
    "relation": "none",
    "anchor_event": "null",
    "direction": "none"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "lookup",
      "goal": "entity_info",
      "required_graph_components": [
        "APPEARS_IN"
      ]
    }
  ],
  "execution_plan": [
    "lookup_entity('Barack Obama')"
  ],
  "confidence": 0.0,
  "ambiguity_flags": []
}
```

### Query: Was Obama at the White House before he discussed Climate Change?
```json
{
  "query_type": "causal",
  "entities": [
    {
      "name": "Obama",
      "type": "person",
      "resolved_entity_id": "person_barack_obama"
    },
    {
      "name": "White House",
      "type": "location",
      "resolved_entity_id": "person_barack_obama"
    },
    {
      "name": "Climate Change",
      "type": "event",
      "resolved_entity_id": "person_barack_obama"
    }
  ],
  "actions": [
    "discussed"
  ],
  "temporal_constraints": {
    "relation": "before",
    "anchor_event": "discussed Climate Change",
    "direction": "forward"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "entity_location",
      "goal": "Obama's location before discussing Climate Change",
      "required_graph_components": [
        "APPEARS_IN"
      ]
    },
    {
      "id": "Q2",
      "type": "event_action",
      "goal": "Obama discussed Climate Change",
      "required_graph_components": [
        "RELATED_TO"
      ]
    },
    {
      "id": "Q3",
      "type": "location_event",
      "goal": "Climate Change at the White House",
      "required_graph_components": [
        "SHARES_ENTITY"
      ]
    }
  ],
  "execution_plan": [
    "Find Obama's location before discussing Climate Change",
    "Find Obama discussed Climate Change",
    "Find Climate Change at the White House"
  ],
  "confidence": 0.0,
  "ambiguity_flags": []
}
```

### Query: Find the person who discussed Climate Change, and then show me where they went next.
```json
{
  "query_type": "find_and_follow",
  "entities": [
    {
      "name": "person",
      "type": "entity",
      "resolved_entity_id": "person_barack_obama"
    }
  ],
  "actions": [
    "discussed",
    "went"
  ],
  "temporal_constraints": {
    "relation": "after",
    "anchor_event": "discussed Climate Change",
    "direction": "forward"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "find",
      "goal": "person who discussed Climate Change",
      "required_graph_components": [
        "APPEARS_IN",
        "NEXT"
      ]
    },
    {
      "id": "Q2",
      "type": "follow",
      "goal": "where they went next",
      "required_graph_components": [
        "NEXT"
      ]
    }
  ],
  "execution_plan": [
    "Find person who discussed Climate Change",
    "Get next event of person"
  ],
  "confidence": 0.0,
  "ambiguity_flags": []
}
```

### Query: When he was at the White House, did he smile?
```json
{
  "query_type": "factoid_question",
  "entities": [
    {
      "name": "he",
      "type": "person",
      "resolved_entity_id": "person_barack_obama"
    },
    {
      "name": "White House",
      "type": "location",
      "resolved_entity_id": "person_barack_obama"
    }
  ],
  "actions": [
    "visit"
  ],
  "temporal_constraints": {
    "relation": "during",
    "anchor_event": "visit",
    "direction": "none"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "entity_resolution",
      "goal": "identify person",
      "required_graph_components": [
        "APPEARS_IN"
      ]
    },
    {
      "id": "Q2",
      "type": "event_extraction",
      "goal": "extract visit event",
      "required_graph_components": [
        "NEXT",
        "SHARES_ENTITY"
      ]
    },
    {
      "id": "Q3",
      "type": "event_attribute_extraction",
      "goal": "extract smile attribute",
      "required_graph_components": [
        "RELATED_TO"
      ]
    }
  ],
  "execution_plan": [
    "Q1",
    "Q2",
    "Q3"
  ],
  "confidence": 0.0,
  "ambiguity_flags": []
}
```

### Query: Simulate failure fallback to Groq
```json
{
  "status": "failure",
  "reason": "llm_unavailable_or_invalid_output",
  "fallback": null
}
```

### Query: Simulate total failure
```json
{
  "status": "failure",
  "reason": "llm_unavailable_or_invalid_output",
  "fallback": null
}
```

