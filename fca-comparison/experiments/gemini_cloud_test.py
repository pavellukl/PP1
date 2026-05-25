import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
Test Gemini 2.5 Flash on the hardest requirement (Req 09).
Runs both JSON and Python output formats.
Logs everything for the knowledge base.
"""
import json
import time
import os
from pathlib import Path
from pydantic import ValidationError
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from google import genai

from schemas import (
    Requirement as RequirementModel,
    EntityExtractionResult,
    FlavourResult,
)
from prompts import ENTITY_SYSTEM, FLAVOUR_SYSTEM, FORMALISM_CONTEXT, strip_think_tags

GEMINI_MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Load Python API source and prompts from comprehensive test
with open(Path(__file__).parent.parent / "formalism_api.py") as f:
    PYTHON_API_SOURCE = f.read()

# Import the same prompts as comprehensive test
from python_vs_json_phi4 import (
    JSON_FORMULA_SYSTEM, PYTHON_FORMULA_SYSTEM,
    parse_json_response, normalize_modifiers, extract_code,
)

REQ_DIR = Path(__file__).resolve().parent.parent.parent / "req"


def call_gemini(system, user):
    r = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{system}\n\n{user}",
    )
    return r.text


def gemini_extract_flavour(requirement):
    content = call_gemini(FLAVOUR_SYSTEM,
        f"Determine the time model:\n\n{requirement}\n\nReturn ONLY a valid JSON object.")
    return FlavourResult.model_validate(parse_json_response(content))


def gemini_extract_entities(requirement):
    content = call_gemini(ENTITY_SYSTEM,
        f"Extract entities from:\n\n{requirement}\n\nReturn ONLY a valid JSON object with an \"entities\" array.")
    data = normalize_modifiers(parse_json_response(content))
    return EntityExtractionResult.model_validate(data)


def gemini_json_constraint(requirement, entities, flavour, max_retries=2):
    feedback = ""
    for attempt in range(max_retries + 1):
        prompt = (
            f"Requirement: {requirement}\n\n"
            f"Time model: {flavour}\n\n"
            f"Extracted entities:\n{entities.model_dump_json(indent=2)}\n\n"
            f"Return ONLY the constraint as valid JSON."
        )
        if feedback:
            prompt += f"\n\nPrevious attempt failed:\n{feedback}\nPlease fix."

        content = call_gemini(JSON_FORMULA_SYSTEM, prompt)
        try:
            data = parse_json_response(content)
            req_data = {"flavour": flavour, "entities": entities.model_dump()["entities"], "constraint": data}
            result = RequirementModel.model_validate(req_data)
            return result, attempt + 1, None
        except (json.JSONDecodeError, ValidationError) as e:
            feedback = f"{type(e).__name__}: {str(e)[:300]}"

    return None, max_retries + 1, feedback


def gemini_python_constraint(requirement, entities, flavour, max_retries=2):
    import formalism_api
    namespace_base = {name: getattr(formalism_api, name) for name in dir(formalism_api) if not name.startswith("_")}

    feedback = ""
    for attempt in range(max_retries + 1):
        prompt = (
            f"Requirement: {requirement}\n\n"
            f"Time model: {flavour}\n\n"
            f"Extracted entities:\n{entities.model_dump_json(indent=2)}\n\n"
            f"Write Python code using the API to build the Requirement. "
            f"Assign the result to `result`. Return ONLY Python code."
        )
        if feedback:
            prompt += f"\n\nPrevious attempt failed:\n{feedback}\nPlease fix the code."

        content = call_gemini(PYTHON_FORMULA_SYSTEM, prompt)
        code = extract_code(content)

        namespace = dict(namespace_base)
        try:
            exec(code, namespace)
            result_data = namespace.get("result")
            if result_data is None:
                feedback = "Code did not assign to `result`"
                continue
            result = RequirementModel.model_validate(result_data)
            return result, attempt + 1, None, code
        except Exception as e:
            feedback = f"{type(e).__name__}: {str(e)[:300]}"

    return None, max_retries + 1, feedback, None


if __name__ == "__main__":
    # Test on Req 09 (hardest) and Req 06 (simplest, as sanity check)
    test_reqs = ["09", "06"]

    results = []

    for req_id in test_reqs:
        req_text = (REQ_DIR / f"{req_id}.md").read_text().strip()
        print(f"\n{'='*70}")
        print(f"REQ {req_id}: {req_text[:80]}...")
        print(f"{'='*70}")

        # Setup
        print("\n  Extracting flavour...")
        t = time.time()
        try:
            flavour = gemini_extract_flavour(req_text)
            print(f"  Flavour: {flavour.flavour} ({time.time()-t:.1f}s)")
        except Exception as e:
            print(f"  FLAVOUR FAILED: {e}")
            continue

        print("  Extracting entities...")
        t = time.time()
        try:
            entities = gemini_extract_entities(req_text)
            print(f"  Entities: {len(entities.entities)} ({time.time()-t:.1f}s)")
            for e in entities.entities:
                mods = ", ".join(f"{m.key}={m.value}" for m in e.modifiers) if e.modifiers else ""
                print(f"    - {e.name}: {e.type.value}" + (f" ({mods})" if mods else ""))
        except Exception as e:
            print(f"  ENTITY FAILED: {e}")
            continue

        # JSON
        print("\n  --- JSON OUTPUT ---")
        t = time.time()
        json_result, json_attempts, json_err = gemini_json_constraint(
            req_text, entities, flavour.flavour)
        json_time = time.time() - t
        if json_result:
            print(f"  SUCCESS ({json_time:.1f}s, {json_attempts} attempt(s))")
            print(f"  Constraint type: {json_result.constraint.construct_type}")
            print(f"  Full JSON:\n{json.dumps(json_result.constraint.model_dump(), indent=2)[:500]}")
        else:
            print(f"  FAILED ({json_time:.1f}s, {json_attempts} attempt(s))")
            print(f"  Error: {json_err}")

        # Python
        print("\n  --- PYTHON OUTPUT ---")
        t = time.time()
        py_result, py_attempts, py_err, py_code = gemini_python_constraint(
            req_text, entities, flavour.flavour)
        py_time = time.time() - t
        if py_result:
            print(f"  SUCCESS ({py_time:.1f}s, {py_attempts} attempt(s))")
            print(f"  Constraint type: {py_result.constraint.construct_type}")
            if py_code:
                print(f"  Generated code:\n{py_code[:500]}")
        else:
            print(f"  FAILED ({py_time:.1f}s, {py_attempts} attempt(s))")
            print(f"  Error: {py_err}")

        results.append({
            "model": GEMINI_MODEL,
            "req": req_id,
            "flavour": flavour.flavour,
            "num_entities": len(entities.entities),
            "json_success": json_result is not None,
            "json_time": round(json_time, 1),
            "json_attempts": json_attempts,
            "json_constraint_type": json_result.constraint.construct_type if json_result else None,
            "python_success": py_result is not None,
            "python_time": round(py_time, 1),
            "python_attempts": py_attempts,
            "python_constraint_type": py_result.constraint.construct_type if py_result else None,
        })

    # Save
    output_path = Path(__file__).parent.parent / "results" / "gemini_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
