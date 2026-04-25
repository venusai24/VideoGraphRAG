from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union


class Entity(BaseModel):
    name: str = Field(..., description="The extracted name of the entity.")
    type: str = Field(..., description="The type/class of the entity (person, location, topic, object, event).")
    resolved_entity_id: Optional[str] = Field(None, description="Canonical EntityRef ID, populated post-extraction.")


class TemporalConstraints(BaseModel):
    relation: Literal["before", "after", "during", "none"] = Field("none")
    anchor_event: Optional[str] = Field(None)
    direction: Literal["forward", "backward", "neutral", "none"] = Field("none")


class SubQuery(BaseModel):
    id: str = Field(..., description="Unique identifier, e.g. 'Q1'.")
    type: str = Field(..., description="Sub-query type, e.g. 'entity_lookup', 'temporal_traversal'.")
    goal: str = Field(..., description="What this sub-query resolves.")
    required_graph_components: List[Literal["APPEARS_IN", "NEXT", "SHARES_ENTITY", "RELATED_TO"]] = Field(...)


# ─── Structured Execution Step types ───────────────────────────────────────────

class StepResolveEntity(BaseModel):
    step: int
    operation: Literal["resolve_entity"]
    input: str = Field(..., description="Entity name string to resolve.")
    output: str = Field(..., description="Resulting EntityRef node ID variable name.")


class StepTraverse(BaseModel):
    step: int
    operation: Literal["traverse"]
    from_node: str = Field(..., alias="from", description="Source node type or variable (EntityRef / ClipRef).")
    edge: Literal["APPEARS_IN", "NEXT", "SHARES_ENTITY", "RELATED_TO"]
    to_node: str = Field(..., alias="to", description="Target node type or variable (EntityRef / ClipRef).")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter conditions on traversal.")

    model_config = {"populate_by_name": True}


class StepFilter(BaseModel):
    step: int
    operation: Literal["filter"]
    condition: Dict[str, Any] = Field(..., description="Filter conditions (field, operator, value).")


class StepTemporalTraverse(BaseModel):
    step: int
    operation: Literal["temporal_traverse"]
    edge: Literal["NEXT"]
    direction: Literal["forward", "backward", "neutral"]
    limit: Optional[int] = Field(None, description="Max hops to traverse.")


class StepExtract(BaseModel):
    step: int
    operation: Literal["extract"]
    target: Literal["EntityRef", "ClipRef"]
    fields: List[str] = Field(..., description="Fields to extract from the target node.")


ExecutionStep = Union[StepResolveEntity, StepTraverse, StepFilter, StepTemporalTraverse, StepExtract]


# ─── Top-level models ──────────────────────────────────────────────────────────

class QueryDecomposition(BaseModel):
    query_type: str
    entities: List[Entity]
    actions: List[str]
    temporal_constraints: TemporalConstraints
    sub_queries: List[SubQuery]
    execution_plan: List[Dict[str, Any]] = Field(
        ..., description="Ordered list of structured execution steps."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    ambiguity_flags: List[str] = Field(default_factory=list)

    def get_typed_execution_plan(self) -> List[ExecutionStep]:
        """Parse execution_plan dicts into typed step objects."""
        typed = []
        op_map = {
            "resolve_entity": StepResolveEntity,
            "traverse": StepTraverse,
            "filter": StepFilter,
            "temporal_traverse": StepTemporalTraverse,
            "extract": StepExtract,
        }
        for raw in self.execution_plan:
            op = raw.get("operation")
            cls = op_map.get(op)
            if cls:
                typed.append(cls.model_validate(raw))
        return typed


class FailureResponse(BaseModel):
    status: str = Field("failure")
    reason: str
    fallback: Optional[Any] = None
