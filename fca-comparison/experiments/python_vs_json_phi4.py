import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Comprehensive Python vs JSON comparison.
Runs all requirements, multiple times, with retry.
Outputs structured results for analysis.
"""
import json
import time
import os
import sys
import requests
from pathlib import Path
from pydantic import ValidationError
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from schemas import (
    Requirement as RequirementModel,
    EntityExtractionResult,
    FlavourResult,
)
from prompts import ENTITY_SYSTEM, FLAVOUR_SYSTEM, FORMALISM_CONTEXT, strip_think_tags

# === Configuration ===

OLLAMA = "http://172.31.128.1:11434/api/chat"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"

MODEL = os.environ.get("MODEL_NAME", "phi4:latest")
RUNS_PER_COMBO = 3
MAX_RETRIES = 2
REQ_DIR = Path(__file__).resolve().parent.parent.parent / "req"

# Load Python API source
with open(Path(__file__).parent.parent / "formalism_api.py") as f:
    PYTHON_API_SOURCE = f.read()


# === Prompts (same formalism context, different output format) ===

PYTHON_FORMULA_SYSTEM = (
    "You are a Requirement Formalizer for safety-critical avionics software.\n"
    "Your goal is: Given a requirement, its extracted entities, and the time model,\n"
    "write Python code that builds the formal constraint using the provided API.\n\n"
    + FORMALISM_CONTEXT + "\n\n"
    "Here is the Python API you must use (all functions return dictionaries):\n\n"
    "```python\n" + PYTHON_API_SOURCE + "\n```\n\n"
    "IMPORTANT RULES:\n"
    "- Import nothing. All functions are already available.\n"
    "- Your code must assign the result to a variable called `result`.\n"
    "- `result` must be a call to `Requirement(flavour, entities, constraint)`.\n"
    "- Use ONLY the functions defined in the API above.\n"
    "- Use entity names EXACTLY as provided from the extraction step.\n"
    "- Modifier values must be simple types (string, number, bool, None), not expressions.\n"
    "- Use `Happening(entity, Now())` to check if an Event is occurring.\n"
    "- Use `Val(entity, Now())` to read an entity's current value.\n"
    "- Prefer the simplest formula that captures the requirement.\n\n"
    "EXAMPLE 1 -- \"Start task execution within 2 seconds after power-up\":\n"
    "```python\n"
    "result = Requirement(\n"
    "    flavour=\"Continuous\",\n"
    "    entities=[\n"
    "        Entity(\"power_up\", \"EventTrigger\", description=\"Power-up trigger\"),\n"
    "        Entity(\"ev_power_up\", \"Event\", [Modifier(\"target\", \"power_up\"), Modifier(\"type\", \"generic\")], \"Power-up event\"),\n"
    "        Entity(\"task_start\", \"EventTrigger\", description=\"Task start trigger\"),\n"
    "        Entity(\"ev_task_start\", \"Event\", [Modifier(\"target\", \"task_start\"), Modifier(\"type\", \"generic\")], \"Task started event\"),\n"
    "    ],\n"
    "    constraint=CausesWithin(\n"
    "        condition=Happening(\"ev_power_up\", Now()),\n"
    "        effect=Happening(\"ev_task_start\", Now()),\n"
    "        duration=2, duration_unit=\"s\",\n"
    "    ),\n"
    ")\n"
    "```\n\n"
    "EXAMPLE 2 -- \"Store data in memory\":\n"
    "```python\n"
    "result = Requirement(\n"
    "    flavour=\"Discrete\",\n"
    "    entities=[\n"
    "        Entity(\"data\", \"Abstract\", description=\"Data to store\"),\n"
    "        Entity(\"mem\", \"Storage\", [Modifier(\"non_volatile\", True)], \"Target memory\"),\n"
    "    ],\n"
    "    constraint=Eventually(Cmp(Val(\"mem\", Now()), \"=\", Val(\"data\", Now()))),\n"
    ")\n"
    "```\n\n"
    "EXAMPLE 3 -- \"Read X then store in Y\":\n"
    "```python\n"
    "result = Requirement(\n"
    "    flavour=\"Discrete\",\n"
    "    entities=[\n"
    "        Entity(\"X\", \"Storage\", [Modifier(\"address\", \"0xAA000018\")], \"Source\"),\n"
    "        Entity(\"Y\", \"Storage\", [Modifier(\"register\", True)], \"Destination\"),\n"
    "        Entity(\"ev_read_x\", \"Event\", [Modifier(\"target\", \"X\"), Modifier(\"type\", \"read\")], \"Read event\"),\n"
    "        Entity(\"ev_written_y\", \"Event\", [Modifier(\"target\", \"Y\"), Modifier(\"type\", \"written\")], \"Write event\"),\n"
    "    ],\n"
    "    constraint=Sequence(\n"
    "        Happening(\"ev_read_x\", Now()),\n"
    "        AllOf(Happening(\"ev_written_y\", Now()), Cmp(Val(\"Y\", Now()), \"=\", Val(\"X\", Now()))),\n"
    "    ),\n"
    ")\n"
    "```\n\n"
    "Return ONLY Python code. No explanations, no markdown fences, no imports."
)

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
    '{"construct_type": "Eventually", "predicate": {"pred_type": "Cmp", "left": {"expr_type": "Val", "entity": "mem", "time": {"expr_type": "Now"}}, "operator": "=", "right": {"expr_type": "Val", "entity": "data", "time": {"expr_type": "Now"}}}}\n\n'
    'EXAMPLE 3 -- "Read X then store in Y":\n'
    '{"construct_type": "Sequence", "steps": [{"pred_type": "Happening", "entity": "ev_read_x", "time": {"expr_type": "Now"}}, {"pred_type": "AllOf", "predicates": [{"pred_type": "Happening", "entity": "ev_written_y", "time": {"expr_type": "Now"}}, {"pred_type": "Cmp", "left": {"expr_type": "Val", "entity": "Y", "time": {"expr_type": "Now"}}, "operator": "=", "right": {"expr_type": "Val", "entity": "X", "time": {"expr_type": "Now"}}}]}]}\n\n'
    "Return ONLY valid JSON. No explanations, no markdown."
)


# === LLM call ===

def call_ollama(system, user):
    r = requests.post(OLLAMA, json={
        "model": MODEL, "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }, timeout=300)
    content = r.json()["message"]["content"]
    return strip_think_tags(content)


# === Parsing helpers ===

def parse_json_response(text):
    text = strip_think_tags(text)
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
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            candidate = text[idx:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    return json.loads(text.strip())


def normalize_modifiers(data):
    if isinstance(data, dict) and "entities" in data:
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


def extract_code(text):
    text = strip_think_tags(text).strip()
    if "```" in text:
        lines = text.split("\n")
        code_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines)
    return text


# === Shared pipeline steps ===

def extract_flavour(requirement, retries=2):
    for attempt in range(retries + 1):
        content = call_ollama(FLAVOUR_SYSTEM,
            f"Determine the time model:\n\n{requirement}\n\nAfter your reasoning, return ONLY a valid JSON object.")
        try:
            return FlavourResult.model_validate(parse_json_response(content))
        except (json.JSONDecodeError, ValidationError):
            if attempt == retries:
                raise


def extract_entities(requirement, retries=2):
    for attempt in range(retries + 1):
        content = call_ollama(ENTITY_SYSTEM,
            f"Extract entities from:\n\n{requirement}\n\nAfter your reasoning, return ONLY a valid JSON object with an \"entities\" array.")
        try:
            data = normalize_modifiers(parse_json_response(content))
            return EntityExtractionResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            if attempt == retries:
                raise


# === Format-specific constraint generation with retry ===

def generate_json_constraint(requirement, entities, flavour, max_retries=MAX_RETRIES):
    feedback = ""
    for attempt in range(max_retries + 1):
        prompt = (
            f"Requirement: {requirement}\n\n"
            f"Time model: {flavour}\n\n"
            f"Extracted entities:\n{entities.model_dump_json(indent=2)}\n\n"
            f"Return ONLY the constraint as valid JSON after your reasoning."
        )
        if feedback:
            prompt += f"\n\nPrevious attempt failed:\n{feedback}\nPlease fix."

        content = call_ollama(JSON_FORMULA_SYSTEM, prompt)
        try:
            data = parse_json_response(content)
            req_data = {"flavour": flavour, "entities": entities.model_dump()["entities"], "constraint": data}
            result = RequirementModel.model_validate(req_data)
            return result, attempt + 1, None
        except (json.JSONDecodeError, ValidationError) as e:
            feedback = f"{type(e).__name__}: {str(e)[:300]}"

    return None, max_retries + 1, feedback


def generate_python_constraint(requirement, entities, flavour, max_retries=MAX_RETRIES):
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
        code = extract_code(content)

        namespace = dict(namespace_base)
        try:
            exec(code, namespace)
            result_data = namespace.get("result")
            if result_data is None:
                feedback = "Code did not assign to `result`"
                continue
            result = RequirementModel.model_validate(result_data)
            return result, attempt + 1, None
        except Exception as e:
            feedback = f"{type(e).__name__}: {str(e)[:300]}"

    return None, max_retries + 1, feedback


# === Main ===

if __name__ == "__main__":
    # Load requirements
    req_files = sorted(REQ_DIR.glob("*.md"))
    requirements = [(f.stem, f.read_text().strip()) for f in req_files]

    print("=" * 70)
    print(f"COMPREHENSIVE PYTHON vs JSON COMPARISON")
    print(f"Model: {MODEL}")
    print(f"Requirements: {len(requirements)}")
    print(f"Runs per combo: {RUNS_PER_COMBO}")
    print(f"Max retries: {MAX_RETRIES}")
    print("=" * 70)

    all_results = []

    for req_id, req_text in requirements:
        print(f"\n{'='*70}")
        print(f"REQ {req_id}: {req_text[:80]}...")
        print(f"{'='*70}")

        for run in range(1, RUNS_PER_COMBO + 1):
            print(f"\n  --- Run {run}/{RUNS_PER_COMBO} ---")

            # Shared setup
            try:
                t0 = time.time()
                flavour = extract_flavour(req_text)
                entities = extract_entities(req_text)
                setup_time = time.time() - t0
                print(f"  Setup: {flavour.flavour}, {len(entities.entities)} entities ({setup_time:.1f}s)")
            except Exception as e:
                print(f"  SETUP FAILED: {e}")
                all_results.append({
                    "req": req_id, "run": run,
                    "json_success": False, "python_success": False,
                    "setup_failed": True, "error": str(e)[:200],
                })
                continue

            # JSON
            t = time.time()
            json_result, json_attempts, json_err = generate_json_constraint(
                req_text, entities, flavour.flavour)
            json_time = time.time() - t
            json_ok = json_result is not None
            json_constraint_type = json_result.constraint.construct_type if json_ok else None
            print(f"  JSON:   {'OK' if json_ok else 'FAIL'} ({json_time:.1f}s, {json_attempts} attempt(s))"
                  + (f" [{json_constraint_type}]" if json_ok else f" [{json_err[:80]}]"))

            # Python
            t = time.time()
            py_result, py_attempts, py_err = generate_python_constraint(
                req_text, entities, flavour.flavour)
            py_time = time.time() - t
            py_ok = py_result is not None
            py_constraint_type = py_result.constraint.construct_type if py_ok else None
            print(f"  Python: {'OK' if py_ok else 'FAIL'} ({py_time:.1f}s, {py_attempts} attempt(s))"
                  + (f" [{py_constraint_type}]" if py_ok else f" [{py_err[:80]}]"))

            all_results.append({
                "req": req_id,
                "run": run,
                "model": MODEL,
                "setup_failed": False,
                "flavour": flavour.flavour,
                "num_entities": len(entities.entities),
                "json_success": json_ok,
                "json_time": round(json_time, 1),
                "json_attempts": json_attempts,
                "json_constraint_type": json_constraint_type,
                "json_error": json_err,
                "python_success": py_ok,
                "python_time": round(py_time, 1),
                "python_attempts": py_attempts,
                "python_constraint_type": py_constraint_type,
                "python_error": py_err,
            })

    # === Summary ===
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = [r for r in all_results if not r.get("setup_failed")]
    json_successes = sum(1 for r in valid if r["json_success"])
    py_successes = sum(1 for r in valid if r["python_success"])
    setup_failures = sum(1 for r in all_results if r.get("setup_failed"))

    print(f"Total runs: {len(all_results)}")
    print(f"Setup failures: {setup_failures}")
    print(f"JSON success:   {json_successes}/{len(valid)} ({100*json_successes/max(len(valid),1):.0f}%)")
    print(f"Python success: {py_successes}/{len(valid)} ({100*py_successes/max(len(valid),1):.0f}%)")

    json_times = [r["json_time"] for r in valid if r["json_success"]]
    py_times = [r["python_time"] for r in valid if r["python_success"]]
    if json_times:
        print(f"JSON avg time:   {sum(json_times)/len(json_times):.1f}s")
    if py_times:
        print(f"Python avg time: {sum(py_times)/len(py_times):.1f}s")

    json_retries = [r["json_attempts"] for r in valid if r["json_success"]]
    py_retries = [r["python_attempts"] for r in valid if r["python_success"]]
    if json_retries:
        print(f"JSON avg attempts:   {sum(json_retries)/len(json_retries):.1f}")
    if py_retries:
        print(f"Python avg attempts: {sum(py_retries)/len(py_retries):.1f}")

    # Per-requirement breakdown
    print("\nPer-requirement:")
    for req_id, _ in requirements:
        req_results = [r for r in valid if r["req"] == req_id]
        if not req_results:
            print(f"  {req_id}: all setup failures")
            continue
        j = sum(1 for r in req_results if r["json_success"])
        p = sum(1 for r in req_results if r["python_success"])
        n = len(req_results)
        print(f"  {req_id}: JSON {j}/{n}, Python {p}/{n}")

    # Save results
    output_path = Path(__file__).parent.parent / "results" / "comprehensive_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
