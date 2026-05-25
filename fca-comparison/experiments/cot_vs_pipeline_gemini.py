import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""
CoT vs Pipeline comparison using Gemini 3.5 Flash.
Budget: 20 RPD, 5 RPM. Each requirement costs ~4 calls (3 pipeline + 1 CoT).
Test on 5 requirements of varying difficulty.
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
from prompts import (
    ENTITY_SYSTEM, FLAVOUR_SYSTEM, FORMALISM_CONTEXT,
    FORMULA_SYSTEM, COT_SYSTEM, strip_think_tags,
)
from python_vs_json_phi4 import (
    JSON_FORMULA_SYSTEM, parse_json_response, normalize_modifiers,
)

GEMINI_MODEL = "gemini-3.5-flash"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
REQ_DIR = Path(__file__).resolve().parent.parent.parent / "req"

# Pick 5 requirements spanning difficulty:
# 06 (simple CausesWithin), 02 (Eventually), 03 (Sequence),
# 08 (Causes with value check), 09 (complex multi-condition)
TEST_REQS = ["06", "02", "03", "09"]

request_count = 0
last_call_time = 0


def call_gemini(system, user):
    global request_count, last_call_time
    request_count += 1
    # 5 RPM = 12s spacing minimum, add margin
    elapsed = time.time() - last_call_time
    if elapsed < 13 and last_call_time > 0:
        wait = 13 - elapsed
        print(f"    (rate limit pause {wait:.0f}s)")
        time.sleep(wait)

    for attempt in range(3):
        try:
            last_call_time = time.time()
            r = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{system}\n\n{user}",
            )
            return r.text
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print(f"    (rate limited, waiting 15s...)")
                time.sleep(15)
            else:
                raise


# === Pipeline approach (3 calls) ===

def run_pipeline(requirement):
    errors = []

    # Call 1: Flavour
    content = call_gemini(FLAVOUR_SYSTEM,
        f"Determine the time model:\n\n{requirement}\n\nReturn ONLY a valid JSON object.")
    try:
        flavour = FlavourResult.model_validate(parse_json_response(content)).flavour
    except Exception as e:
        return None, f"Flavour failed: {e}"

    # Call 2: Entities
    content = call_gemini(ENTITY_SYSTEM,
        f"Extract entities from:\n\n{requirement}\n\nReturn ONLY a valid JSON object with an \"entities\" array.")
    try:
        data = normalize_modifiers(parse_json_response(content))
        entities = EntityExtractionResult.model_validate(data)
    except Exception as e:
        return None, f"Entity extraction failed: {e}"

    # Call 3: Constraint
    prompt = (
        f"Requirement: {requirement}\n\n"
        f"Time model: {flavour}\n\n"
        f"Extracted entities:\n{entities.model_dump_json(indent=2)}\n\n"
        f"Return ONLY the constraint as valid JSON."
    )
    content = call_gemini(JSON_FORMULA_SYSTEM, prompt)
    try:
        constraint_data = parse_json_response(content)
        req_data = {
            "flavour": flavour,
            "entities": entities.model_dump()["entities"],
            "constraint": constraint_data,
        }
        result = RequirementModel.model_validate(req_data)
        return result, None
    except Exception as e:
        return None, f"Constraint failed: {type(e).__name__}: {str(e)[:200]}"


# === CoT approach (1 call) ===

def run_cot(requirement):
    prompt = f"Requirement: {requirement}\n\nNo ambiguity notes."
    content = call_gemini(COT_SYSTEM, prompt)

    # Extract the last JSON block
    last_brace = content.rfind("}")
    if last_brace < 0:
        return None, "No JSON found in response"

    depth = 0
    start = last_brace
    for i in range(last_brace, -1, -1):
        if content[i] == "}": depth += 1
        elif content[i] == "{": depth -= 1
        if depth == 0:
            start = i
            break

    json_str = content[start:last_brace + 1]

    try:
        data = json.loads(json_str)
        data = normalize_modifiers(data)
        result = RequirementModel.model_validate(data)
        return result, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"


# === Main ===

if __name__ == "__main__":
    print("=" * 70)
    print(f"CoT vs PIPELINE COMPARISON")
    print(f"Model: {GEMINI_MODEL}")
    print(f"Requirements: {TEST_REQS}")
    print("=" * 70)

    results = []

    for req_id in TEST_REQS:
        req_text = (REQ_DIR / f"{req_id}.md").read_text().strip()
        print(f"\n{'='*70}")
        print(f"REQ {req_id}: {req_text[:80]}...")
        print(f"{'='*70}")

        # Pipeline
        print("\n  --- PIPELINE (3 calls) ---")
        t = time.time()
        try:
            pipe_result, pipe_err = run_pipeline(req_text)
        except Exception as e:
            pipe_result, pipe_err = None, f"Exception: {type(e).__name__}: {str(e)[:200]}"
        pipe_time = time.time() - t
        if pipe_result:
            print(f"  SUCCESS ({pipe_time:.1f}s)")
            print(f"  Constraint: {pipe_result.constraint.construct_type}")
            print(f"  Entities: {len(pipe_result.entities)}")
            constraint_json = json.dumps(pipe_result.constraint.model_dump(), indent=2)
        else:
            print(f"  FAILED ({pipe_time:.1f}s)")
            print(f"  Error: {pipe_err}")
            constraint_json = None

        # CoT
        print("\n  --- CoT (1 call) ---")
        t = time.time()
        try:
            cot_result, cot_err = run_cot(req_text)
        except Exception as e:
            cot_result, cot_err = None, f"Exception: {type(e).__name__}: {str(e)[:200]}"
        cot_time = time.time() - t
        if cot_result:
            print(f"  SUCCESS ({cot_time:.1f}s)")
            print(f"  Constraint: {cot_result.constraint.construct_type}")
            print(f"  Entities: {len(cot_result.entities)}")
            cot_constraint_json = json.dumps(cot_result.constraint.model_dump(), indent=2)
        else:
            print(f"  FAILED ({cot_time:.1f}s)")
            print(f"  Error: {cot_err}")
            cot_constraint_json = None

        # Compare
        if pipe_result and cot_result:
            same_type = pipe_result.constraint.construct_type == cot_result.constraint.construct_type
            print(f"\n  Same constraint type: {same_type}")
            print(f"  Pipeline entities: {len(pipe_result.entities)}, CoT entities: {len(cot_result.entities)}")

        results.append({
            "model": GEMINI_MODEL,
            "req": req_id,
            "pipeline_success": pipe_result is not None,
            "pipeline_time": round(pipe_time, 1),
            "pipeline_calls": 3,
            "pipeline_constraint_type": pipe_result.constraint.construct_type if pipe_result else None,
            "pipeline_num_entities": len(pipe_result.entities) if pipe_result else None,
            "pipeline_error": pipe_err,
            "pipeline_constraint": json.loads(constraint_json) if constraint_json else None,
            "cot_success": cot_result is not None,
            "cot_time": round(cot_time, 1),
            "cot_calls": 1,
            "cot_constraint_type": cot_result.constraint.construct_type if cot_result else None,
            "cot_num_entities": len(cot_result.entities) if cot_result else None,
            "cot_error": cot_err,
            "cot_constraint": json.loads(cot_constraint_json) if cot_constraint_json else None,
        })

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    pipe_ok = sum(1 for r in results if r["pipeline_success"])
    cot_ok = sum(1 for r in results if r["cot_success"])
    print(f"Pipeline: {pipe_ok}/{len(results)} success")
    print(f"CoT:      {cot_ok}/{len(results)} success")
    print(f"Total API requests used: {request_count}")

    pipe_times = [r["pipeline_time"] for r in results if r["pipeline_success"]]
    cot_times = [r["cot_time"] for r in results if r["cot_success"]]
    if pipe_times:
        print(f"Pipeline avg time: {sum(pipe_times)/len(pipe_times):.1f}s")
    if cot_times:
        print(f"CoT avg time:      {sum(cot_times)/len(cot_times):.1f}s")

    # Per-requirement
    print("\nPer requirement:")
    for r in results:
        p = "OK" if r["pipeline_success"] else "FAIL"
        c = "OK" if r["cot_success"] else "FAIL"
        pt = r["pipeline_constraint_type"] or "-"
        ct = r["cot_constraint_type"] or "-"
        print(f"  Req {r['req']}: Pipeline={p} [{pt}], CoT={c} [{ct}]")

    # Save
    output_path = Path(__file__).parent.parent / "results" / "cot_vs_pipeline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
