from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum
from typing import Annotated, Literal, Union


# === Entity Types (Definition 6) ===

class EntityType(str, Enum):
    SIGNAL = "SIGNAL"
    STORAGE = "STORAGE"
    EVENT = "EVENT"
    CHANNEL = "CHANNEL"
    STATE = "STATE"
    VALUE = "VALUE"
    ABSTRACT = "ABSTRACT"


class OperationType(str, Enum):
    CALCULATE = "calculate"
    READ = "read"
    STORE = "store"
    FIRE = "fire"
    TRANSMIT = "transmit"
    RECEIVE = "receive"
    SET_STATE = "set_state"
    COMPARE = "compare"


VALID_OPERATIONS: dict[EntityType, set[OperationType]] = {
    EntityType.SIGNAL: {OperationType.CALCULATE},
    EntityType.STORAGE: {OperationType.READ, OperationType.STORE},
    EntityType.EVENT: {OperationType.FIRE},
    EntityType.CHANNEL: {OperationType.TRANSMIT, OperationType.RECEIVE},
    EntityType.STATE: {OperationType.SET_STATE},
    EntityType.VALUE: set(),
    EntityType.ABSTRACT: {OperationType.COMPARE},
}


# === Entities (Definition 10) ===

class Entity(BaseModel):
    name: str = Field(description="Snake_case identifier, e.g. 'power_up', 'tasks_running'")
    type: EntityType
    modifiers: list[str] = Field(default_factory=list, description="Optional modifier properties for this entity, usually empty")
    description: str = Field(description="One-sentence description of what this entity represents in the requirement")


class EntityExtractionResult(BaseModel):
    entities: list[Entity] = Field(min_length=1)


# === Ambiguity Review ===

class AmbiguityNote(BaseModel):
    category: str = Field(description="Category from the checklist, e.g. 'Temporal ambiguity', 'Scope of negation', 'Missing causes'")
    text: str = Field(description="Brief description of the ambiguity found")
    severity: Literal["low", "medium", "high"] = Field(description="How problematic this ambiguity is")


class AmbiguityReviewResult(BaseModel):
    notes: list[AmbiguityNote] = Field(default_factory=list)


# === Expressions (§4.1) ===

class Constant(BaseModel):
    expr_type: Literal["Constant"] = "Constant"
    value: float
    unit: str | None = Field(None, description="Time or measurement unit, e.g. 's', 'ms', 'ns'")


class EntityVal(BaseModel):
    """Val_ρ(e, t) — value of entity e at time t (Definition 23)"""
    expr_type: Literal["EntityVal"] = "EntityVal"
    entity: str = Field(description="Entity name from the extraction step")
    time_var: str = Field(description="Time variable, e.g. 't', 't_prime'")


class Diff(BaseModel):
    """Diff(t1, t2) = |t1 - t2| — absolute duration between two time points"""
    expr_type: Literal["Diff"] = "Diff"
    t1: str = Field(description="First time variable, e.g. 't'")
    t2: str = Field(description="Second time variable, e.g. 't_prime'")


Expression = Annotated[
    Union[Constant, EntityVal, Diff],
    Field(discriminator="expr_type"),
]


# === Predicates (§4.2) ===

class Cmp(BaseModel):
    """Cmp(a, op, b) — compare two expressions"""
    pred_type: Literal["Cmp"] = "Cmp"
    left: Expression
    operator: str = Field(description="One of: =, !=, <, <=, >, >=")
    right: Expression


class Not(BaseModel):
    """Not(P) — logical negation"""
    pred_type: Literal["Not"] = "Not"
    predicate: Predicate


class AllOf(BaseModel):
    """AllOf — all predicates must be true (logical AND)"""
    pred_type: Literal["AllOf"] = "AllOf"
    predicates: list[Predicate] = Field(min_length=1)


class AnyOf(BaseModel):
    """AnyOf — at least one predicate must be true (logical OR)"""
    pred_type: Literal["AnyOf"] = "AnyOf"
    predicates: list[Predicate] = Field(min_length=1)


class Implies(BaseModel):
    """Implies(A, B) — if A then B"""
    pred_type: Literal["Implies"] = "Implies"
    antecedent: Predicate
    consequent: Predicate


class ForAll(BaseModel):
    """ForAll — the predicate must hold for all values of the variable"""
    pred_type: Literal["ForAll"] = "ForAll"
    variable: str
    predicate: Predicate


class Exists(BaseModel):
    """Exists — there must be some value of the variable for which the predicate holds"""
    pred_type: Literal["Exists"] = "Exists"
    variable: str
    predicate: Predicate


# === Operations (§5) — each produces a predicate ===

class Operation(BaseModel):
    """An operation on an entity at a time point (e.g. fire, set_state, calculate)"""
    pred_type: Literal["Operation"] = "Operation"
    operation: OperationType
    entity: str = Field(description="Entity name from the extraction step")
    time_var: str = Field(description="Time variable, e.g. 't', 't_prime'")
    args: list[str] = Field(default_factory=list, description="Extra arguments if needed, e.g. address for read/store, value for transmit/receive")


Predicate = Annotated[
    Union[Cmp, Not, AllOf, AnyOf, Implies, ForAll, Exists, Operation],
    Field(discriminator="pred_type"),
]

# Rebuild models to resolve forward references
for cls in [Not, AllOf, AnyOf, Implies, ForAll, Exists, Cmp]:
    cls.model_rebuild()


# === Step 2 output ===

class RequirementFormula(BaseModel):
    """The formalized requirement as a predicate tree."""
    time_type: Literal["discrete", "continuous"] = Field(description="'discrete' for step-based timing, 'continuous' for real-valued time")
    formula: Predicate


# === Step 3 output ===

class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
