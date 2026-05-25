import re

def task_prompt(task: str) -> str:
    return f"Current Task: {task}\n\nDo not use <think> tags. Provide your complete response as valid JSON only:"


def strip_think_tags(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip() or text


FORMALISM_CONTEXT = """\
ENTITY TYPES:
- Signal: transient computed value. Operations: calculate(e, expr)
- Storage: persistent value. Operations: write(e, expr), read(e_src, e_dst)
- EventTrigger: something that can fire an event. Operations: fire(e)
- Event: instantaneous occurrence, always linked to an EventTrigger via modifiers.
  Modifiers: target (entity id of the EventTrigger), type (one of: generic, calculated, written, read, entered, transmitted, received, inserted, removed, cleared).
  No direct operations — Event entities are created automatically by the EvtPred mechanism when operations fire.
- Channel: communication medium. Operations: transmit(e, expr), receive(e_src, e_dst)
- State: named behavioral mode. Operations: set_state(e, value)
- Value: an explicit value referenced in the requirement
- Abstract: opaque entity, identity comparison only
- Set: a finite collection. Operations: insert(e, v), remove(e, v), clear(e)

REQUIREMENT STRUCTURE:
A requirement is: (flavour, entities, constraint)
- flavour: "Discrete" or "Continuous" (determines the time model)
- entities: list of entities, each with name, type, modifiers, description
- constraint: a temporal construct (or composition of temporal constructs)

TEMPORAL CONSTRUCTS (these wrap predicates to make trace-level statements):
- Always(predicate): predicate holds at all time points
- Eventually(predicate): predicate holds at some time point
- Initial(predicate): predicate holds at time 0
- Causes(condition, effect): whenever condition holds, effect eventually follows
- CausesWithin(condition, effect, duration): like Causes but effect must occur within duration
- Sequence(step1, step2, ...): steps occur in order at some time points
- Precedes(before, after): whenever 'after' holds, 'before' must have held at some earlier time
- Excludes(p1, p2): p1 and p2 never hold at the same time
- Immediately(trigger, effect): effect holds at the next time step after trigger (discrete only)

PREDICATES (point-in-time statements, used inside temporal constructs):
- Cmp(left, operator, right): compare two expressions. Operators: =, !=, <, <=, >, >=
- AllOf(predicates): all must be true (logical AND)
- AnyOf(predicates): at least one must be true (logical OR)
- Implies(antecedent, consequent): if A then B
- Not(predicate): logical negation
- ForAll(set_expr, variable, predicate): predicate holds for all elements
- Exists(set_expr, variable, predicate): predicate holds for some element
- Happening(entity, time): Event entity is occurring at time
- HasHappened(entity, time): Event entity has occurred at or before time
- Operation(operation, entity, args): an operation on an entity

EXPRESSIONS (evaluate to values, used inside predicates):
- Constant(value, unit?): literal value
- ArithOp(operator, left, right): arithmetic (+, -, *, /)
- Now(): current ambient time point
- Prev(time): previous time step (discrete only)
- Next(time): next time step (discrete only)
- Val(entity, time): value of entity at time
- ValBefore(entity, time): value of entity just before time
- EvtOccCount(entity, interval): count of event occurrences in interval
- LastOcc(entity, time, n): interval spanning the n most recent occurrences up to time
- FirstOcc(entity, time, n): interval spanning the n earliest occurrences at or after time
- MaxVal(entity, interval): maximum value of entity over interval
- MinVal(entity, interval): minimum value of entity over interval
- MkInterval(t1, t2, left_open?, right_open?): construct an interval
- Start(interval), End(interval), Duration(interval): interval accessors
- Diff(t1, t2): absolute duration between two time points
- Addr(entity): static address of a Storage entity
- Size(set_expr): cardinality of a set
- Filter(set_expr, variable, predicate): filter elements satisfying a predicate

IMPORTANT PATTERNS:
- EventTrigger + Event pairs: when an operation fires on an entity, corresponding Event entities
  (linked via their target and type modifiers) automatically become true. For example, fire(power_up)
  causes Happening(ev_power_up, Now) where ev_power_up has target=power_up and type=generic.
- Time is ambient: expressions use Now() and temporal constructs supply the time context.
  Do NOT use explicit time variables like 't' or 't_prime'. Use Now(), Prev(Now()), etc.

EXAMPLE — "System must start task execution within 2 seconds after power-up":
Entities:
  processor_power_up: EventTrigger
  ev_power_up: Event (target=processor_power_up, type=generic)
  invoke_task_seq: EventTrigger
  ev_invoke_task_seq: Event (target=invoke_task_seq, type=generic)
Constraint:
  CausesWithin(
    condition=Happening(ev_power_up, Now()),
    effect=Happening(ev_invoke_task_seq, Now()),
    duration=2, duration_unit="s"
  )
"""

AMBIGUITY_CHECKLIST = """\
- Dangling else / scope of action
- Ambiguity of reference
- Omissions: causes without effects, effects without causes, complete omissions
- Ambiguous logical operators (or/and/nor/nand), implicit connectors
- Negation: scope of negation, unnecessary negation, double negation
- Ambiguous statements: vague verbs/adverbs/adjectives, unnecessary aliases
- Built-in assumptions (functional/environmental knowledge)
- Ambiguous precedence relationships
- Implicit cases, etc. vs i.e.
- Temporal ambiguity
- Boundary ambiguity
"""

# =============================================================================
# System prompts for PIPELINE variant (separate LLM calls)
# =============================================================================

AMBIGUITY_SYSTEM = (
    "You are Ambiguity Reviewer. You are an expert in analyzing safety-critical "
    "avionics requirements for ambiguity and potential misinterpretation.\n"
    "Your personal goal is: Review requirements against the ambiguity checklist "
    "and flag genuine issues without rewriting the requirement.\n\n"
    f"Ambiguity checklist:\n{AMBIGUITY_CHECKLIST}\n"
    "For each ambiguity found, specify the category, a brief description, "
    "and severity (low/medium/high).\n"
    "Do NOT rewrite or modify the requirement. Only flag issues.\n"
    "If the requirement is clear, return an empty notes list."
)

FLAVOUR_SYSTEM = (
    "You are a Time Model Classifier for safety-critical avionics requirements.\n"
    "Your goal is: Determine whether a requirement uses discrete or continuous time.\n\n"
    "Rules:\n"
    "- Discrete: the requirement describes step-by-step behavior, register operations, "
    "state transitions at clock edges, or sequential actions without real-time durations.\n"
    "- Continuous: the requirement references real-time durations (seconds, milliseconds, "
    "nanoseconds), deadlines, or continuous physical quantities.\n\n"
    "Return JSON: {\"flavour\": \"Discrete\"} or {\"flavour\": \"Continuous\"}"
)

ENTITY_SYSTEM = (
    "You are Entity Extractor. You are an entity extraction system for "
    "safety-critical avionics requirements.\n"
    "Your personal goal is: Extract all entities from a requirement using "
    "the formalism's entity types, producing precise and complete results.\n\n"
    f"{FORMALISM_CONTEXT}\n"
    "Each entity has: name (unique identifier), type (from the entity types above), "
    "modifiers (list of key-value pairs, see Event modifiers), description.\n"
    "Every entity name must be unique. If two concepts share a name, "
    "disambiguate with a suffix.\n"
    "IMPORTANT: For every EventTrigger, create a corresponding Event entity with "
    "target and type modifiers. For operations that produce events (calculate, write, "
    "read, set_state, transmit, receive, insert, remove, clear), create corresponding "
    "Event entities if the requirement references those occurrences."
)

FORMULA_SYSTEM = (
    "You are Requirement Formalizer. You are a requirement formalization system "
    "for safety-critical avionics.\n"
    "Your personal goal is: Given the extracted entities and time model, build "
    "a formal constraint that precisely captures the temporal and logical "
    "structure of the requirement.\n\n"
    f"{FORMALISM_CONTEXT}\n"
    "Follow this decomposition to build the constraint:\n\n"
    "STEP 1 — IDENTIFY TOP-LEVEL TEMPORAL SCOPE:\n"
    "What is the outermost temporal pattern?\n"
    "- Must ALWAYS hold? -> Always\n"
    "- Must EVENTUALLY happen? -> Eventually\n"
    "- CAUSE and EFFECT? -> Causes or CausesWithin (if deadline)\n"
    "- Ordered SEQUENCE of actions? -> Sequence\n"
    "- INITIAL condition? -> Initial\n"
    "- Combines multiple of the above? -> TraceAllOf\n"
    "Identify the construct and separate condition from effect.\n\n"
    "STEP 2 — DECOMPOSE CONDITION (if applicable):\n"
    "Break the condition into predicates:\n"
    "- Contains AND/OR/IF? Split into sub-predicates.\n"
    "- Event happening? -> Happening(entity, Now())\n"
    "- Value comparison? -> Cmp(Val(entity, Now()), op, expr)\n"
    "- Count of occurrences? -> Cmp(EvtOccCount(...), op, value)\n"
    "- Reference to a past event's time? -> Use LastOcc, Start, Val at that time\n\n"
    "STEP 3 — DECOMPOSE EFFECT:\n"
    "Same decomposition as the condition. Pay attention to:\n"
    "- What value must the entity have? Use Val and Cmp.\n"
    "- Is it a toggle (new = negation of old)? Use ValBefore.\n"
    "- Is it just an event firing? Use Happening.\n\n"
    "STEP 4 — NORMALIZE:\n"
    "- Entity names must match EXACTLY those provided.\n"
    "- Operations must be compatible with entity types.\n"
    "- Time references use Now(), Prev(Now()), Start(LastOcc(...)), etc.\n\n"
    "After reasoning, return ONLY the constraint as valid JSON."
)

# =============================================================================
# System prompt for COT variant (single LLM call for steps 2-4)
# =============================================================================

COT_SYSTEM = (
    "You are a Requirement Formalizer for safety-critical avionics software.\n"
    "Your goal is: Translate a natural language requirement into a formal "
    "representation by following a hierarchical decomposition algorithm.\n\n"
    f"{FORMALISM_CONTEXT}\n"
    "You will be given a requirement and any ambiguity notes from a prior review.\n"
    "Follow this algorithm EXACTLY, showing your work for each step.\n\n"
    "=== STAGE 1: MACRO-STRUCTURE ===\n\n"
    "STEP 1 — DETERMINE TIME MODEL:\n"
    "Decide whether the requirement uses discrete or continuous time.\n"
    "- Discrete: step-by-step behavior, register operations, state transitions, "
    "sequential actions without real-time durations.\n"
    "- Continuous: real-time durations (seconds, milliseconds, nanoseconds), "
    "deadlines, continuous physical quantities.\n"
    "State your choice and why.\n\n"
    "STEP 2 — IDENTIFY TOP-LEVEL TEMPORAL SCOPE:\n"
    "What is the outermost temporal pattern of this requirement?\n"
    "- Does it describe something that must ALWAYS hold? -> Always\n"
    "- Does it describe something that must EVENTUALLY happen? -> Eventually\n"
    "- Does it describe a CAUSE and EFFECT? -> Causes or CausesWithin (if deadline)\n"
    "- Does it describe an ordered SEQUENCE of actions? -> Sequence\n"
    "- Does it describe an INITIAL condition? -> Initial\n"
    "- Does it combine multiple of the above? -> TraceAllOf\n"
    "Identify the top-level construct and separate the condition/trigger "
    "from the effect/body. State what each part is.\n\n"
    "=== STAGE 2: RECURSIVE DECOMPOSITION ===\n\n"
    "STEP 3 — EXTRACT ENTITIES:\n"
    "Now identify all entities referenced in the requirement.\n"
    "For each entity determine: name, type, modifiers, description.\n"
    "Rules:\n"
    "- Every entity name must be unique (snake_case).\n"
    "- For every EventTrigger, create a corresponding Event entity with "
    "target and type modifiers.\n"
    "- For operations that produce events (write, read, transmit, etc.), "
    "create Event entities if the requirement references those occurrences.\n"
    "- Modifier values must be simple types (string, number, bool), not expressions.\n"
    "List all entities.\n\n"
    "STEP 4 — DECOMPOSE CONDITION (if applicable):\n"
    "If the top-level construct has a condition/trigger (from Step 2), "
    "decompose it into predicates:\n"
    "- Does it contain logical connectives (AND, OR, IF)? Split into sub-predicates.\n"
    "- For each sub-predicate, is it:\n"
    "  - An event happening? -> Happening(entity, Now())\n"
    "  - A value comparison? -> Cmp(Val(entity, Now()), op, expr)\n"
    "  - A count of occurrences? -> Cmp(EvtOccCount(...), op, value)\n"
    "  - A temporal reference to a past event? -> Use LastOcc, Start, Val at that time\n"
    "Process each sub-predicate recursively until all are atomic.\n\n"
    "STEP 5 — DECOMPOSE EFFECT:\n"
    "Decompose the effect/body the same way as Step 4.\n"
    "Pay special attention to:\n"
    "- What value must an entity have after the effect? Use Val and Cmp.\n"
    "- Is it a toggle (new value = negation of old)? Use Val and ValBefore.\n"
    "- Is it just an event firing? Use Happening.\n\n"
    "STEP 6 — NORMALIZE ATOMIC PROPOSITIONS:\n"
    "Review each leaf predicate and ensure:\n"
    "- Entity names match exactly those from Step 3.\n"
    "- Operations are compatible with entity types.\n"
    "- Time references use Now(), Prev(Now()), Start(LastOcc(...)), etc.\n"
    "- Values use Val(entity, time) or Constant(value).\n\n"
    "=== FINAL OUTPUT ===\n\n"
    "After completing all steps, output ONLY a JSON object.\n\n"
    "CRITICAL FORMAT RULES:\n"
    "- Temporal constructs use \"construct_type\" as discriminator:\n"
    "  {\"construct_type\": \"CausesWithin\", \"condition\": ..., \"effect\": ..., \"duration\": 2, \"duration_unit\": \"s\"}\n"
    "- Predicates use \"pred_type\" as discriminator:\n"
    "  {\"pred_type\": \"Happening\", \"entity\": \"ev_name\", \"time\": {\"expr_type\": \"Now\"}}\n"
    "  {\"pred_type\": \"Cmp\", \"left\": <expr>, \"operator\": \">\", \"right\": <expr>}\n"
    "  {\"pred_type\": \"AllOf\", \"predicates\": [...]}\n"
    "- Expressions use \"expr_type\" as discriminator:\n"
    "  {\"expr_type\": \"Now\"}\n"
    "  {\"expr_type\": \"Val\", \"entity\": \"name\", \"time\": <expr>}\n"
    "  {\"expr_type\": \"Constant\", \"value\": 655}\n"
    "- Entities use modifiers as a list of {\"key\": \"...\", \"value\": \"...\"}:\n"
    "  {\"name\": \"ev_x\", \"type\": \"Event\", \"modifiers\": [{\"key\": \"target\", \"value\": \"x\"}, {\"key\": \"type\", \"value\": \"generic\"}], \"description\": \"...\"}\n\n"
    "DO NOT use wrapper keys like {\"CausesWithin\": {...}} or {\"Happening\": {...}}.\n"
    "DO use flat objects with discriminator fields: {\"construct_type\": \"CausesWithin\", ...}.\n\n"
    "Output structure: {\"flavour\": \"Discrete|Continuous\", \"entities\": [...], \"constraint\": {...}}"
)
