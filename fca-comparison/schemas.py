from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum
from typing import Annotated, Literal, Union


# === Entity Types (Definition 8) ===

class EntityType(str, Enum):
    SIGNAL = "Signal"
    STORAGE = "Storage"
    EVENT = "Event"
    EVENT_TRIGGER = "EventTrigger"
    CHANNEL = "Channel"
    STATE = "State"
    VALUE = "Value"
    ABSTRACT = "Abstract"
    SET = "Set"


class EventTypeLabel(str, Enum):
    GENERIC = "generic"
    CALCULATED = "calculated"
    WRITTEN = "written"
    READ = "read"
    ENTERED = "entered"
    TRANSMITTED = "transmitted"
    RECEIVED = "received"
    INSERTED = "inserted"
    REMOVED = "removed"
    CLEARED = "cleared"


class OperationType(str, Enum):
    CALCULATE = "calculate"
    WRITE = "write"
    READ = "read"
    FIRE = "fire"
    SET_STATE = "set_state"
    TRANSMIT = "transmit"
    RECEIVE = "receive"
    INSERT = "insert"
    REMOVE = "remove"
    CLEAR = "clear"


VALID_OPERATIONS: dict[EntityType, set[OperationType]] = {
    EntityType.SIGNAL: {OperationType.CALCULATE},
    EntityType.STORAGE: {OperationType.WRITE, OperationType.READ},
    EntityType.EVENT: set(),
    EntityType.EVENT_TRIGGER: {OperationType.FIRE},
    EntityType.CHANNEL: {OperationType.TRANSMIT, OperationType.RECEIVE},
    EntityType.STATE: {OperationType.SET_STATE},
    EntityType.VALUE: set(),
    EntityType.ABSTRACT: set(),
    EntityType.SET: {OperationType.INSERT, OperationType.REMOVE, OperationType.CLEAR},
}


# === Modifier (Definition 10) ===

class Modifier(BaseModel):
    key: str = Field(description="Modifier key, e.g. 'address', 'target', 'type', 'register', 'non_volatile'")
    value: str | float | bool | None = Field(None, description="Modifier value. None for flag modifiers (presence alone carries meaning)")


# === Entity (Definition 11) ===

class Entity(BaseModel):
    name: str = Field(description="Unique identifier, e.g. 'power_up', 'ev_power_up', 'DTSCON'")
    type: EntityType
    modifiers: list[Modifier] = Field(default_factory=list)
    description: str = Field(description="What this entity represents in the requirement")


class EntityExtractionResult(BaseModel):
    entities: list[Entity] = Field(min_length=1)


# === Ambiguity Review ===

class AmbiguityNote(BaseModel):
    category: str = Field(description="E.g. 'Temporal ambiguity', 'Scope of negation', 'Missing causes'")
    text: str = Field(description="Brief description of the ambiguity found")
    severity: Literal["low", "medium", "high"]


class AmbiguityReviewResult(BaseModel):
    notes: list[AmbiguityNote] = Field(default_factory=list)


# === Flavour (time model) ===

class FlavourResult(BaseModel):
    flavour: Literal["Discrete", "Continuous"]


# === Expressions (§5.1) ===

class Constant(BaseModel):
    expr_type: Literal["Constant"] = "Constant"
    value: float | bool | str
    unit: str | None = Field(None, description="E.g. 's', 'ms', 'ns'")


class ArithOp(BaseModel):
    expr_type: Literal["ArithOp"] = "ArithOp"
    operator: Literal["+", "-", "*", "/"]
    left: Expression
    right: Expression


class Negate(BaseModel):
    expr_type: Literal["Negate"] = "Negate"
    operand: Expression


class Now(BaseModel):
    expr_type: Literal["Now"] = "Now"


class Prev(BaseModel):
    expr_type: Literal["Prev"] = "Prev"
    time: Expression


class Next(BaseModel):
    expr_type: Literal["Next"] = "Next"
    time: Expression


class RelTime(BaseModel):
    expr_type: Literal["RelTime"] = "RelTime"
    time: Expression
    offset: int


class Diff(BaseModel):
    expr_type: Literal["Diff"] = "Diff"
    t1: Expression
    t2: Expression


class MkInterval(BaseModel):
    expr_type: Literal["MkInterval"] = "MkInterval"
    t1: Expression
    t2: Expression
    left_open: bool = False
    right_open: bool = False


class SinceZero(BaseModel):
    expr_type: Literal["SinceZero"] = "SinceZero"
    time: Expression


class Start(BaseModel):
    expr_type: Literal["Start"] = "Start"
    interval: Expression


class End(BaseModel):
    expr_type: Literal["End"] = "End"
    interval: Expression


class Duration(BaseModel):
    expr_type: Literal["Duration"] = "Duration"
    interval: Expression


class Addr(BaseModel):
    expr_type: Literal["Addr"] = "Addr"
    entity: str


class Val(BaseModel):
    expr_type: Literal["Val"] = "Val"
    entity: str
    time: Expression


class ValBefore(BaseModel):
    expr_type: Literal["ValBefore"] = "ValBefore"
    entity: str
    time: Expression


class EvtOccCount(BaseModel):
    expr_type: Literal["EvtOccCount"] = "EvtOccCount"
    entity: str
    interval: Expression


class LastOcc(BaseModel):
    expr_type: Literal["LastOcc"] = "LastOcc"
    entity: str
    time: Expression
    n: int = 1


class FirstOcc(BaseModel):
    expr_type: Literal["FirstOcc"] = "FirstOcc"
    entity: str
    time: Expression
    n: int = 1


class MaxVal(BaseModel):
    expr_type: Literal["MaxVal"] = "MaxVal"
    entity: str
    interval: Expression


class MinVal(BaseModel):
    expr_type: Literal["MinVal"] = "MinVal"
    entity: str
    interval: Expression


class Size(BaseModel):
    expr_type: Literal["Size"] = "Size"
    set_expr: Expression


class Filter(BaseModel):
    expr_type: Literal["Filter"] = "Filter"
    set_expr: Expression
    variable: str
    predicate: Predicate


Expression = Annotated[
    Union[
        Constant, ArithOp, Negate, Now, Prev, Next, RelTime, Diff,
        MkInterval, SinceZero, Start, End, Duration,
        Addr, Val, ValBefore, EvtOccCount, LastOcc, FirstOcc, MaxVal, MinVal,
        Size, Filter,
    ],
    Field(discriminator="expr_type"),
]


# === Predicates (§5.2) ===

class Cmp(BaseModel):
    pred_type: Literal["Cmp"] = "Cmp"
    left: Expression
    operator: Literal["=", "!=", "<", "<=", ">", ">="]
    right: Expression


class Not(BaseModel):
    pred_type: Literal["Not"] = "Not"
    predicate: Predicate


class AllOf(BaseModel):
    pred_type: Literal["AllOf"] = "AllOf"
    predicates: list[Predicate] = Field(min_length=1)


class AnyOf(BaseModel):
    pred_type: Literal["AnyOf"] = "AnyOf"
    predicates: list[Predicate] = Field(min_length=1)


class Implies(BaseModel):
    pred_type: Literal["Implies"] = "Implies"
    antecedent: Predicate
    consequent: Predicate


class ForAll(BaseModel):
    pred_type: Literal["ForAll"] = "ForAll"
    set_expr: Expression
    variable: str
    predicate: Predicate


class Exists(BaseModel):
    pred_type: Literal["Exists"] = "Exists"
    set_expr: Expression
    variable: str
    predicate: Predicate


class Happening(BaseModel):
    pred_type: Literal["Happening"] = "Happening"
    entity: str
    time: Expression


class HasHappened(BaseModel):
    pred_type: Literal["HasHappened"] = "HasHappened"
    entity: str
    time: Expression


class IntervalIncludes(BaseModel):
    pred_type: Literal["IntervalIncludes"] = "IntervalIncludes"
    outer: Expression
    inner: Expression


class IntervalExcludes(BaseModel):
    pred_type: Literal["IntervalExcludes"] = "IntervalExcludes"
    i1: Expression
    i2: Expression


# === Operations (§6) ===

class Operation(BaseModel):
    pred_type: Literal["Operation"] = "Operation"
    operation: OperationType
    entity: str
    args: list[Expression] = Field(default_factory=list, description="Additional arguments: expr for calculate/write/set_state/transmit, destination entity for read/receive, value for insert/remove")


Predicate = Annotated[
    Union[
        Cmp, Not, AllOf, AnyOf, Implies, ForAll, Exists,
        Happening, HasHappened, IntervalIncludes, IntervalExcludes,
        Operation,
    ],
    Field(discriminator="pred_type"),
]

# Rebuild models to resolve forward references
for cls in [Not, AllOf, AnyOf, Implies, ForAll, Exists, Cmp,
            ArithOp, Negate, Filter, MkInterval, SinceZero,
            Start, End, Duration, Prev, Next, Diff, RelTime,
            Val, ValBefore, EvtOccCount, LastOcc, FirstOcc, MaxVal, MinVal,
            Size, IntervalIncludes, IntervalExcludes]:
    cls.model_rebuild()


# === Temporal Constructs (Definition 56) ===

class Always(BaseModel):
    construct_type: Literal["Always"] = "Always"
    predicate: Predicate


class Eventually(BaseModel):
    construct_type: Literal["Eventually"] = "Eventually"
    predicate: Predicate


class Initial(BaseModel):
    construct_type: Literal["Initial"] = "Initial"
    predicate: Predicate


class Causes(BaseModel):
    construct_type: Literal["Causes"] = "Causes"
    condition: Predicate
    effect: Predicate


class CausesWithin(BaseModel):
    construct_type: Literal["CausesWithin"] = "CausesWithin"
    condition: Predicate
    effect: Predicate
    duration: float
    duration_unit: str = Field(default="s", description="Time unit, e.g. 's', 'ms', 'ns'")


class Sequence(BaseModel):
    construct_type: Literal["Sequence"] = "Sequence"
    steps: list[Predicate] = Field(min_length=2)


class Precedes(BaseModel):
    construct_type: Literal["Precedes"] = "Precedes"
    before: Predicate
    after: Predicate


class Excludes(BaseModel):
    construct_type: Literal["Excludes"] = "Excludes"
    p1: Predicate
    p2: Predicate


class Immediately(BaseModel):
    construct_type: Literal["Immediately"] = "Immediately"
    trigger: Predicate
    effect: Predicate


TemporalConstruct = Annotated[
    Union[Always, Eventually, Initial, Causes, CausesWithin, Sequence, Precedes, Excludes, Immediately],
    Field(discriminator="construct_type"),
]


# === Composed trace predicates ===

class TraceAllOf(BaseModel):
    construct_type: Literal["TraceAllOf"] = "TraceAllOf"
    constructs: list[TemporalConstruct] = Field(min_length=1)


class TraceAnyOf(BaseModel):
    construct_type: Literal["TraceAnyOf"] = "TraceAnyOf"
    constructs: list[TemporalConstruct] = Field(min_length=1)


Constraint = Annotated[
    Union[Always, Eventually, Initial, Causes, CausesWithin, Sequence, Precedes, Excludes, Immediately, TraceAllOf, TraceAnyOf],
    Field(discriminator="construct_type"),
]


# === Requirement (Definition 57) ===

class Requirement(BaseModel):
    flavour: Literal["Discrete", "Continuous"]
    entities: list[Entity] = Field(min_length=1)
    constraint: Constraint


# === Validation ===

class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
