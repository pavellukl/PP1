import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Test: Python code output vs JSON output for requirement formalization.
Uses phi4-reasoning:plus via Ollama native API.
"""
import json
import time
import requests
from pydantic import ValidationError
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from schemas import (
    Requirement as RequirementModel,
    EntityExtractionResult,
    AmbiguityReviewResult,
    FlavourResult,
)
from prompts import AMBIGUITY_SYSTEM, FLAVOUR_SYSTEM, ENTITY_SYSTEM, FORMALISM_CONTEXT, strip_think_tags

OLLAMA = "http://172.31.128.1:11434/api/chat"
MODEL = "phi4:latest"

# Read the Python API source to include in the prompt
with open("formalism_api.py") as f:
    PYTHON_API_SOURCE = f.read()

PYTHON_FORMULA_SYSTEM = f"""\
You are a Requirement Formalizer for safety-critical avionics software.
Your goal is: Given a requirement, its extracted entities, and the time model,
write Python code that builds the formal constraint using the provided API.

{FORMALISM_CONTEXT}

Here is the Python API you must use (all functions return dictionaries):

```python
{PYTHON_API_SOURCE}
```

IMPORTANT RULES:
- Import nothing. All functions are already available.
- Your code must assign the result to a variable called `result`.
- `result` must be a call to `Requirement(flavour, entities, constraint)`.
- Use ONLY the functions defined in the API above.
- Use entity names EXACTLY as provided from the extraction step.
- Use `Happening(entity, Now())` to check if an Event is occurring.
- Use `Val(entity, Now())` to read an entity's current value.
- Prefer the simplest formula that captures the requirement.

EXAMPLE 1 — "Start task execution within 2 seconds after power-up":
```python
result = Requirement(
    flavour="Continuous",
    entities=[
        Entity("processor_power_up", "EventTrigger", description="Processor power-up trigger"),
        Entity("ev_power_up", "Event", [Modifier("target", "processor_power_up"), Modifier("type", "generic")], "Power-up event"),
        Entity("invoke_task_seq", "EventTrigger", description="Task sequence start trigger"),
        Entity("ev_invoke_task_seq", "Event", [Modifier("target", "invoke_task_seq"), Modifier("type", "generic")], "Task sequence started event"),
    ],
    constraint=CausesWithin(
        condition=Happening("ev_power_up", Now()),
        effect=Happening("ev_invoke_task_seq", Now()),
        duration=2,
        duration_unit="s",
    ),
)
```

EXAMPLE 2 — "Store data in memory":
```python
result = Requirement(
    flavour="Discrete",
    entities=[
        Entity("data_source", "Abstract", description="The data to store"),
        Entity("memory_block", "Storage", [Modifier("non_volatile", True)], "Target memory location"),
    ],
    constraint=Eventually(
        Cmp(Val("memory_block", Now()), "=", Val("data_source", Now()))
    ),
)
```

EXAMPLE 3 — "Read X then store X in register Y":
```python
result = Requirement(
    flavour="Discrete",
    entities=[
        Entity("X", "Storage", [Modifier("address", "0xAA000018")], "Source storage"),
        Entity("Y", "Storage", [Modifier("register", True)], "Destination register"),
        Entity("ev_read_x", "Event", [Modifier("target", "X"), Modifier("type", "read")], "X read event"),
        Entity("ev_written_y", "Event", [Modifier("target", "Y"), Modifier("type", "written")], "Y written event"),
    ],
    constraint=Sequence(
        Happening("ev_read_x", Now()),
        AllOf(
            Happening("ev_written_y", Now()),
            Cmp(Val("Y", Now()), "=", Val("X", Now()))
        ),
    ),
)
```

Return ONLY Python code. No explanations, no markdown fences, no imports.
"""

JSON_FORMULA_SYSTEM = (
    "You are a Requirement Formalizer for safety-critical avionics software.\n"
    "Your goal is: Given a requirement, its extracted entities, and the time model,\n"
    "produce a JSON object representing the formal constraint.\n\n"
    + FORMALISM_CONTEXT + "\n\n"
    "The JSON output must use the following structure.\n\n"
    "TEMPORAL CONSTRUCTS (top-level, each has \"construct_type\"):\n"
    '- Always:       {construct_type: "Always", predicate: <predicate>}\n'
    '- Eventually:   {construct_type: "Eventually", predicate: <predicate>}\n'
    '- Initial:      {construct_type: "Initial", predicate: <predicate>}\n'
    '- Causes:       {construct_type: "Causes", condition: <predicate>, effect: <predicate>}\n'
    '- CausesWithin: {construct_type: "CausesWithin", condition: <predicate>, effect: <predicate>, duration: <number>, duration_unit: "<unit>"}\n'
    '- Sequence:     {construct_type: "Sequence", steps: [<predicate>, ...]}\n'
    '- Precedes:     {construct_type: "Precedes", before: <predicate>, after: <predicate>}\n'
    '- Excludes:     {construct_type: "Excludes", p1: <predicate>, p2: <predicate>}\n'
    '- Immediately:  {construct_type: "Immediately", trigger: <predicate>, effect: <predicate>}\n'
    '- TraceAllOf:   {construct_type: "TraceAllOf", constructs: [<construct>, ...]}\n\n'
    "PREDICATES (each has \"pred_type\"):\n"
    '- Happening:    {pred_type: "Happening", entity: "<name>", time: <expr>}\n'
    '- HasHappened:  {pred_type: "HasHappened", entity: "<name>", time: <expr>}\n'
    '- Cmp:          {pred_type: "Cmp", left: <expr>, operator: "=|!=|<|<=|>|>=", right: <expr>}\n'
    '- AllOf:        {pred_type: "AllOf", predicates: [<predicate>, ...]}\n'
    '- AnyOf:        {pred_type: "AnyOf", predicates: [<predicate>, ...]}\n'
    '- Implies:      {pred_type: "Implies", antecedent: <predicate>, consequent: <predicate>}\n'
    '- Not:          {pred_type: "Not", predicate: <predicate>}\n'
    '- Operation:    {pred_type: "Operation", operation: "<op>", entity: "<name>", args: [<expr>, ...]}\n'
    '  Operations: calculate, write, read, fire, set_state, transmit, receive, insert, remove, clear\n\n'
    "EXPRESSIONS (each has \"expr_type\"):\n"
    '- Now:          {expr_type: "Now"}\n'
    '- Constant:     {expr_type: "Constant", value: <number|bool|string>}\n'
    '- Val:          {expr_type: "Val", entity: "<name>", time: <expr>}\n'
    '- ValBefore:    {expr_type: "ValBefore", entity: "<name>", time: <expr>}\n'
    '- Prev:         {expr_type: "Prev", time: <expr>}\n'
    '- Next:         {expr_type: "Next", time: <expr>}\n'
    '- ArithOp:      {expr_type: "ArithOp", operator: "+|-|*|/", left: <expr>, right: <expr>}\n'
    '- Diff:         {expr_type: "Diff", t1: <expr>, t2: <expr>}\n'
    '- Addr:         {expr_type: "Addr", entity: "<name>"}\n'
    '- EvtOccCount:  {expr_type: "EvtOccCount", entity: "<name>", interval: <expr>}\n'
    '- MkInterval:   {expr_type: "MkInterval", t1: <expr>, t2: <expr>, left_open: false, right_open: false}\n\n'
    'EXAMPLE 1 -- "Start task execution within 2 seconds after power-up":\n'
    '{"construct_type": "CausesWithin", "condition": {"pred_type": "Happening", "entity": "ev_power_up", "time": {"expr_type": "Now"}}, "effect": {"pred_type": "Happening", "entity": "ev_task_start", "time": {"expr_type": "Now"}}, "duration": 2, "duration_unit": "s"}\n\n'
    'EXAMPLE 2 -- "Store data in memory":\n'
    '{"construct_type": "Eventually", "predicate": {"pred_type": "Cmp", "left": {"expr_type": "Val", "entity": "memory_block", "time": {"expr_type": "Now"}}, "operator": "=", "right": {"expr_type": "Val", "entity": "data_source", "time": {"expr_type": "Now"}}}}\n\n'
    'EXAMPLE 3 -- "Read X then store X in register Y":\n'
    '{"construct_type": "Sequence", "steps": [{"pred_type": "Happening", "entity": "ev_read_x", "time": {"expr_type": "Now"}}, {"pred_type": "AllOf", "predicates": [{"pred_type": "Happening", "entity": "ev_written_y", "time": {"expr_type": "Now"}}, {"pred_type": "Cmp", "left": {"expr_type": "Val", "entity": "Y", "time": {"expr_type": "Now"}}, "operator": "=", "right": {"expr_type": "Val", "entity": "X", "time": {"expr_type": "Now"}}}]}]}\n\n'
    "Return ONLY valid JSON. No explanations, no markdown."
)


REQUIREMENTS = [
    (
        "The software must start to execute the sequence of tasks allocated "
        "to the execution schedule in less than 2 seconds after processor power-up."
    ),
    (
        "The software shall store the maximum execution time measurement data "
        "in non-volatile memory at address MEASUREMT_BLOCK."
    ),
    (
        "The software must perform the following actions in the specified order: "
        "1) Read the calibration constant value at the address 0xAA000018; "
        "2) Store the calibration constant value in the DTSCON register."
    ),
]


def call_ollama(system, user, max_retries=2):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(max_retries + 1):
        r = requests.post(OLLAMA, json={
            "model": MODEL, "stream": False,
            "messages": messages,
        }, timeout=300)
        content = r.json()["message"]["content"]
        content = strip_think_tags(content)
        return content
    return None


def parse_json_response(text):
    text = strip_think_tags(text)
    # Try markdown code fences first
    if "```" in text:
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        if json_lines:
            return json.loads("\n".join(json_lines))
    # Try to find JSON object in the text
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            candidate = text[idx:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    # Last resort: try the whole text
    return json.loads(text.strip())


def normalize_modifiers(data):
    if isinstance(data, dict):
        if "entities" in data:
            for entity in data["entities"]:
                if "modifiers" in entity:
                    normalized = []
                    for mod in entity["modifiers"]:
                        if isinstance(mod, dict) and "key" in mod:
                            normalized.append(mod)
                        elif isinstance(mod, dict):
                            for k, v in mod.items():
                                normalized.append({"key": k, "value": v})
                        elif isinstance(mod, str):
                            normalized.append({"key": mod, "value": None})
                        elif isinstance(mod, list) and len(mod) == 2:
                            normalized.append({"key": mod[0], "value": mod[1]})
                    entity["modifiers"] = normalized
    return data


def extract_entities(requirement, retries=2):
    for attempt in range(retries + 1):
        prompt = f"Extract entities from:\n\n{requirement}\n\nAfter your reasoning, return ONLY a valid JSON object with an \"entities\" array."
        content = call_ollama(ENTITY_SYSTEM, prompt)
        try:
            data = parse_json_response(content)
            data = normalize_modifiers(data)
            return EntityExtractionResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < retries:
                print(f"    Retry entity extraction ({e.__class__.__name__})...")
            else:
                raise


def extract_flavour(requirement, retries=2):
    for attempt in range(retries + 1):
        prompt = f"Determine the time model:\n\n{requirement}\n\nAfter your reasoning, return ONLY a valid JSON object."
        content = call_ollama(FLAVOUR_SYSTEM, prompt)
        try:
            return FlavourResult.model_validate(parse_json_response(content))
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < retries:
                print(f"    Retry flavour extraction ({e.__class__.__name__})...")
            else:
                raise


def test_json_output(requirement, entities, flavour, max_retries=2):
    feedback = ""
    for attempt in range(max_retries + 1):
        prompt = (
            f"Requirement: {requirement}\n\n"
            f"Time model: {flavour}\n\n"
            f"Extracted entities:\n{entities.model_dump_json(indent=2)}\n\n"
            f"Return ONLY the constraint as valid JSON after your reasoning."
        )
        if feedback:
            prompt += f"\n\nPrevious attempt failed validation:\n{feedback}\nPlease fix these issues."

        content = call_ollama(JSON_FORMULA_SYSTEM, prompt)
        try:
            data = parse_json_response(content)
            req_data = {"flavour": flavour, "entities": entities.model_dump()["entities"], "constraint": data}
            result = RequirementModel.model_validate(req_data)
            return True, result, None, attempt + 1
        except (json.JSONDecodeError, ValidationError) as e:
            feedback = f"{type(e).__name__}: {str(e)[:300]}"
            if attempt < max_retries:
                print(f"    Retry JSON (attempt {attempt + 2})...")

    return False, None, feedback, max_retries + 1


def test_python_output(requirement, entities, flavour, max_retries=2):
    import formalism_api
    namespace_base = {name: getattr(formalism_api, name) for name in dir(formalism_api) if not name.startswith("_")}

    feedback = ""
    for attempt in range(max_retries + 1):
        prompt = (
            f"Requirement: {requirement}\n\n"
            f"Time model: {flavour}\n\n"
            f"Extracted entities:\n{entities.model_dump_json(indent=2)}\n\n"
            f"Write Python code using the API to build the Requirement. "
            f"Assign the result to `result`. Return ONLY Python code after your reasoning."
        )
        if feedback:
            prompt += f"\n\nPrevious attempt failed:\n{feedback}\nPlease fix the code."

        content = call_ollama(PYTHON_FORMULA_SYSTEM, prompt)
        code = strip_think_tags(content).strip()
        if "```" in code:
            lines = code.split("\n")
            code_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    code_lines.append(line)
            if code_lines:
                code = "\n".join(code_lines)

        namespace = dict(namespace_base)
        try:
            exec(code, namespace)
            result_data = namespace.get("result")
            if result_data is None:
                feedback = "Code did not assign to `result`"
                if attempt < max_retries:
                    print(f"    Retry Python (attempt {attempt + 2})...")
                continue
            result = RequirementModel.model_validate(result_data)
            return True, result, None, attempt + 1
        except Exception as e:
            feedback = f"{type(e).__name__}: {str(e)[:300]}"
            if attempt < max_retries:
                print(f"    Retry Python (attempt {attempt + 2})...")

    return False, None, feedback, max_retries + 1


# === Main ===

if __name__ == "__main__":
    print("=" * 70)
    print("PYTHON vs JSON OUTPUT COMPARISON")
    print(f"Model: {MODEL}")
    print("=" * 70)

    for i, req in enumerate(REQUIREMENTS):
        print(f"\n{'='*70}")
        print(f"REQUIREMENT {i+1}: {req[:80]}...")
        print(f"{'='*70}")

        # Shared steps
        try:
            print("\n  Extracting flavour...")
            t = time.time()
            flavour = extract_flavour(req)
            print(f"  Flavour: {flavour.flavour} ({time.time()-t:.1f}s)")

            print("  Extracting entities...")
            t = time.time()
            entities = extract_entities(req)
            print(f"  Entities: {len(entities.entities)} ({time.time()-t:.1f}s)")
            for e in entities.entities:
                print(f"    - {e.name}: {e.type.value}")
        except Exception as e:
            print(f"  SETUP FAILED: {type(e).__name__}: {str(e)[:200]}")
            continue

        # JSON test
        print("\n  --- JSON OUTPUT ---")
        t = time.time()
        json_ok, json_result, json_err, json_attempts = test_json_output(req, entities, flavour.flavour)
        json_time = time.time() - t
        if json_ok:
            print(f"  SUCCESS ({json_time:.1f}s, {json_attempts} attempt(s))")
            print(f"  Constraint type: {json_result.constraint.construct_type}")
        else:
            print(f"  FAILED ({json_time:.1f}s, {json_attempts} attempt(s))")
            print(f"  Error: {json_err}")

        # Python test
        print("\n  --- PYTHON OUTPUT ---")
        t = time.time()
        py_ok, py_result, py_err, py_attempts = test_python_output(req, entities, flavour.flavour)
        py_time = time.time() - t
        if py_ok:
            print(f"  SUCCESS ({py_time:.1f}s, {py_attempts} attempt(s))")
            print(f"  Constraint type: {py_result.constraint.construct_type}")
        else:
            print(f"  FAILED ({py_time:.1f}s, {py_attempts} attempt(s))")
            print(f"  Error: {py_err}")

    print("\n" + "=" * 70)
    print("DONE")
