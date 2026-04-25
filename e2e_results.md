# E2E Test Results

### Query: Who is Barack Obama?
```json
{
  "query_type": "ENTITY_INFO",
  "entities": [
    {
      "name": "Barack Obama",
      "type": "Person",
      "resolved_entity_id": "person_barack_obama"
    }
  ],
  "actions": [
    "GET_INFO"
  ],
  "temporal_constraints": {
    "relation": "none",
    "anchor_event": null,
    "direction": "none"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "ENTITY_INFO",
      "goal": "Retrieve basic information about Barack Obama",
      "required_graph_components": []
    }
  ],
  "execution_plan": [
    "Filter entity nodes where name='Barack Obama' and type='Person'",
    "Extract properties from matched entity node"
  ],
  "confidence": 0.9,
  "ambiguity_flags": []
}
```

### Query: Was Obama at the White House before he discussed Climate Change?
```json
{
  "query_type": "temporal_comparison",
  "entities": [
    {
      "name": "Obama",
      "type": "Person",
      "resolved_entity_id": "person_barack_obama"
    },
    {
      "name": "White House",
      "type": "Location",
      "resolved_entity_id": null
    },
    {
      "name": "Climate Change",
      "type": "Topic",
      "resolved_entity_id": null
    }
  ],
  "actions": [
    "was at",
    "discussed"
  ],
  "temporal_constraints": {
    "relation": "before",
    "anchor_event": "discussed Climate Change",
    "direction": "backward"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "event_presence",
      "goal": "Identify Obama's presence at White House",
      "required_graph_components": [
        "APPEARS_IN"
      ]
    },
    {
      "id": "Q2",
      "type": "event_relation",
      "goal": "Identify Obama's discussion about Climate Change",
      "required_graph_components": [
        "RELATED_TO"
      ]
    }
  ],
  "execution_plan": [
    "Filter EntityRef[Obama] \u2192 ClipRef[White House] via APPEARS_IN",
    "Filter EntityRef[Obama] \u2192 ClipRef[Climate Change discussion] via RELATED_TO",
    "Compare temporal order of ClipRef[White House] and ClipRef[Climate Change discussion]"
  ],
  "confidence": 0.3,
  "ambiguity_flags": [
    "uncertain about specific discussion event"
  ]
}
```

### Query: Find the person who discussed Climate Change, and then show me where they went next.
```json
{
  "query_type": "event_sequence",
  "entities": [
    {
      "name": "Climate Change",
      "type": "Topic",
      "resolved_entity_id": null
    },
    {
      "name": "Person",
      "type": "Person",
      "resolved_entity_id": "person_barack_obama"
    }
  ],
  "actions": [
    "discussed",
    "went next"
  ],
  "temporal_constraints": {
    "relation": "after",
    "anchor_event": "discussed Climate Change",
    "direction": "forward"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "entity_identification",
      "goal": "Identify person node connected to Climate Change discussion",
      "required_graph_components": [
        "APPEARS_IN",
        "RELATED_TO"
      ]
    },
    {
      "id": "Q2",
      "type": "temporal_sequence",
      "goal": "Find subsequent locations/events after Climate Change discussion",
      "required_graph_components": [
        "NEXT",
        "RELATED_TO"
      ]
    }
  ],
  "execution_plan": [
    "Filter Person nodes [P] WHERE P APPEARS_IN Clip C AND C RELATED_TO Topic 'Climate Change'",
    "Traverse NEXT edges from identified Person node P to find subsequent locations/events",
    "Extract properties of target nodes from traversal results"
  ],
  "confidence": 0.45,
  "ambiguity_flags": [
    "Multiple persons may discuss Climate Change",
    "Ambiguous 'next' destination definition"
  ]
}
```

### Query: When he was at the White House, did he smile?
```json
{
  "query_type": "event_verification",
  "entities": [
    {
      "name": "he",
      "type": "Person",
      "resolved_entity_id": "person_barack_obama"
    },
    {
      "name": "White House",
      "type": "Location",
      "resolved_entity_id": null
    }
  ],
  "actions": [
    "smile"
  ],
  "temporal_constraints": {
    "relation": "during",
    "anchor_event": "he was at the White House",
    "direction": "neutral"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "location_presence",
      "goal": "Identify clips where 'he' appears at the White House",
      "required_graph_components": [
        "APPEARS_IN"
      ]
    },
    {
      "id": "Q2",
      "type": "action_verification",
      "goal": "Identify clips where 'he' smiles",
      "required_graph_components": [
        "APPEARS_IN"
      ]
    },
    {
      "id": "Q3",
      "type": "temporal_overlap",
      "goal": "Check if smiling clips overlap temporally with White House presence",
      "required_graph_components": [
        "RELATED_TO"
      ]
    }
  ],
  "execution_plan": [
    "Filter clips where EntityRef(he) APPEARS_IN ClipRef(White House)",
    "Traverse EntityRef(he) -> APPEARS_IN to find ClipRef(smiling) where action='smile'",
    "Check temporal overlap between ClipRef(White House) and ClipRef(smiling) via RELATED_TO"
  ],
  "confidence": 0.45,
  "ambiguity_flags": [
    "ambiguous_person_reference"
  ]
}
```

### Query: Simulate failure fallback to Groq
```json
{
  "query_type": "process_simulation",
  "entities": [
    {
      "name": "Groq",
      "type": "organization",
      "resolved_entity_id": null
    }
  ],
  "actions": [
    "simulate failure",
    "fallback"
  ],
  "temporal_constraints": {
    "relation": "none",
    "anchor_event": null,
    "direction": "neutral"
  },
  "sub_queries": [
    {
      "id": "Q1",
      "type": "failure_scenario",
      "goal": "Identify failure scenarios related to Groq",
      "required_graph_components": [
        "RELATED_TO"
      ]
    },
    {
      "id": "Q2",
      "type": "fallback_mechanism",
      "goal": "Locate fallback procedures involving Groq",
      "required_graph_components": [
        "RELATED_TO"
      ]
    }
  ],
  "execution_plan": [
    "Filter nodes [EntityRef:Groq] connected via [RELATED_TO] edge",
    "Traverse [RELATED_TO] edges from failure nodes to [EntityRef:Groq]",
    "Extract fallback pathways from [ClipRef:failure_scenario] to [EntityRef:Groq]"
  ],
  "confidence": 0.0,
  "ambiguity_flags": []
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

