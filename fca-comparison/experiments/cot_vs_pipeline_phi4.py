import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import requests
from pathlib import Path
from pydantic import ValidationError
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from schemas import Requirement as RequirementModel, EntityExtractionResult, FlavourResult
from prompts import ENTITY_SYSTEM, FLAVOUR_SYSTEM, COT_SYSTEM, strip_think_tags
from python_vs_json_phi4 import (
    JSON_FORMULA_SYSTEM, parse_json_response, normalize_modifiers,
    call_ollama, extract_flavour, extract_entities, generate_json_constraint,
)

OLLAMA = "http://172.31.128.1:11434/api/chat"
MODEL = os.environ.get("MODEL_NAME", "phi4:latest")
REQ_DIR = Path(__file__).resolve().parent.parent.parent / "req"
RUNS_PER_REQ = 1
MAX_RETRIES = 1

TEST_REQS = ["06", "02", "03", "09"]


def run_pipeline(requirement):
    try:
        flavour = extract_flavour(requirement)
    except Exception as e:
        return None, f"Flavour failed: {e}"
    try:
        entities = extract_entities(requirement)
    except Exception as e:
        return None, f"Entity extraction failed: {e}"

    result, attempts, err = generate_json_constraint(
        requirement, entities, flavour.flavour, max_retries=MAX_RETRIES)
    if result:
        return result, None
    return None, f"Constraint failed ({attempts} attempts): {err}"


def run_cot(requirement):
    prompt = f"Requirement: {requirement}\n\nNo ambiguity notes."
    content = call_ollama(COT_SYSTEM, prompt)

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
        return None, f"{type(e).__name__}: {str(e)[:300]}"


if __name__ == "__main__":
    req_files = sorted(Path(REQ_DIR).glob("*.md"))
    requirements = [(f.stem, f.read_text().strip()) for f in req_files if f.stem in TEST_REQS]

    print("=" * 70)
    print(f"CoT vs PIPELINE — phi4:latest")
    print(f"Requirements: {[r[0] for r in requirements]}")
    print(f"Runs per requirement: {RUNS_PER_REQ}")
    print("=" * 70)

    all_results = []

    for req_id, req_text in requirements:
        print(f"\n{'='*70}")
        print(f"REQ {req_id}: {req_text[:80]}...")
        print(f"{'='*70}")

        for run in range(1, RUNS_PER_REQ + 1):
            print(f"\n  --- Run {run}/{RUNS_PER_REQ} ---")

            # Pipeline
            t = time.time()
            pipe_result, pipe_err = run_pipeline(req_text)
            pipe_time = time.time() - t
            pipe_ok = pipe_result is not None
            pipe_type = pipe_result.constraint.construct_type if pipe_ok else None
            print(f"  Pipeline: {'OK' if pipe_ok else 'FAIL'} ({pipe_time:.1f}s)"
                  + (f" [{pipe_type}]" if pipe_ok else f" [{pipe_err[:80]}]"))

            # CoT
            t = time.time()
            cot_result, cot_err = run_cot(req_text)
            cot_time = time.time() - t
            cot_ok = cot_result is not None
            cot_type = cot_result.constraint.construct_type if cot_ok else None
            print(f"  CoT:      {'OK' if cot_ok else 'FAIL'} ({cot_time:.1f}s)"
                  + (f" [{cot_type}]" if cot_ok else f" [{cot_err[:80]}]"))

            all_results.append({
                "model": MODEL,
                "req": req_id,
                "run": run,
                "pipeline_success": pipe_ok,
                "pipeline_time": round(pipe_time, 1),
                "pipeline_constraint_type": pipe_type,
                "pipeline_error": pipe_err,
                "cot_success": cot_ok,
                "cot_time": round(cot_time, 1),
                "cot_constraint_type": cot_type,
                "cot_error": cot_err,
            })

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    pipe_ok = sum(1 for r in all_results if r["pipeline_success"])
    cot_ok = sum(1 for r in all_results if r["cot_success"])
    total = len(all_results)
    print(f"Pipeline: {pipe_ok}/{total} ({100*pipe_ok/total:.0f}%)")
    print(f"CoT:      {cot_ok}/{total} ({100*cot_ok/total:.0f}%)")

    pipe_times = [r["pipeline_time"] for r in all_results if r["pipeline_success"]]
    cot_times = [r["cot_time"] for r in all_results if r["cot_success"]]
    if pipe_times:
        print(f"Pipeline avg time: {sum(pipe_times)/len(pipe_times):.1f}s")
    if cot_times:
        print(f"CoT avg time:      {sum(cot_times)/len(cot_times):.1f}s")

    print("\nPer requirement:")
    for req_id, _ in requirements:
        req_r = [r for r in all_results if r["req"] == req_id]
        p = sum(1 for r in req_r if r["pipeline_success"])
        c = sum(1 for r in req_r if r["cot_success"])
        n = len(req_r)
        print(f"  {req_id}: Pipeline {p}/{n}, CoT {c}/{n}")

    output_path = Path(__file__).parent.parent / "results" / "cot_vs_pipeline_phi4_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
