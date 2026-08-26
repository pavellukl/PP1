import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import subprocess
import sys
import time
import json
import re
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

FRAMEWORKS = ["pydantic-ai", "haystack", "langgraph", "crewai"]
REQUIREMENTS = [2, 3, 6, 8]
RUNS_PER_COMBO = 3

VENV_PYTHON = Path(__file__).parent.parent / ".venv" / "bin" / "python"

results = []

total = len(FRAMEWORKS) * len(REQUIREMENTS) * RUNS_PER_COMBO
current = 0

for req_num in REQUIREMENTS:
    for framework in FRAMEWORKS:
        for run in range(1, RUNS_PER_COMBO + 1):
            current += 1
            label = f"[{current}/{total}] {framework} req={req_num} run={run}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            main_py = Path(__file__).parent.parent / "frameworks" / framework / "main.py"
            start = time.time()

            try:
                proc = subprocess.run(
                    [str(VENV_PYTHON), "-W", "ignore::UserWarning", str(main_py), str(req_num)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(main_py.parent),
                )
                elapsed = time.time() - start
                output = proc.stdout + proc.stderr

                success = "Pipeline completed successfully" in output
                failed = "Pipeline failed" in output

                formula_match = re.search(r"Formula:\n(.+?)(?=\n---|\nPipeline)", output, re.DOTALL)
                formula_text = formula_match.group(1).strip() if formula_match else ""

                has_implies = "Implies(" in formula_text
                has_allof = "AllOf(" in formula_text
                has_forall = "ForAll(" in formula_text
                has_exists = "Exists(" in formula_text
                has_cmp = "Cmp(" in formula_text
                has_time_bound = bool(re.search(r"Diff\(", formula_text))

                depth = max(formula_text.count("(") - formula_text.count(")"), 0) if formula_text else 0
                nesting = formula_text.count("(") if formula_text else 0

                validation_match = re.search(r'"valid": (true|false)', output)
                valid = validation_match.group(1) == "true" if validation_match else False

                attempts_matches = re.findall(r"attempt (\d+)", output)
                attempts = int(attempts_matches[-1]) if attempts_matches else 0

                structural_failures = output.count("LLM failed to produce valid structure")

                result = {
                    "framework": framework,
                    "requirement": req_num,
                    "run": run,
                    "success": success,
                    "elapsed_s": round(elapsed, 1),
                    "attempts": attempts,
                    "structural_failures": structural_failures,
                    "valid": valid,
                    "formula": formula_text,
                    "has_implies": has_implies,
                    "has_allof": has_allof,
                    "has_forall": has_forall,
                    "has_exists": has_exists,
                    "has_cmp": has_cmp,
                    "has_time_bound": has_time_bound,
                    "nesting_depth": nesting,
                }
                results.append(result)

                status = "OK" if success else ("FAIL" if failed else "ERROR")
                print(f"  -> {status} in {elapsed:.1f}s, attempts={attempts}, structural_fails={structural_failures}")
                if formula_text:
                    print(f"  -> Formula:\n{formula_text}")

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                results.append({
                    "framework": framework,
                    "requirement": req_num,
                    "run": run,
                    "success": False,
                    "elapsed_s": round(elapsed, 1),
                    "attempts": 0,
                    "structural_failures": 0,
                    "valid": False,
                    "formula": "",
                    "has_implies": False,
                    "has_allof": False,
                    "has_forall": False,
                    "has_exists": False,
                    "has_cmp": False,
                    "has_time_bound": False,
                    "nesting_depth": 0,
                })
                print(f"  -> TIMEOUT after {elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "framework": framework,
                    "requirement": req_num,
                    "run": run,
                    "success": False,
                    "elapsed_s": round(elapsed, 1),
                    "attempts": 0,
                    "structural_failures": 0,
                    "valid": False,
                    "formula": "",
                    "has_implies": False,
                    "has_allof": False,
                    "has_forall": False,
                    "has_exists": False,
                    "has_cmp": False,
                    "has_time_bound": False,
                    "nesting_depth": 0,
                })
                print(f"  -> ERROR: {e}")

# --- Summary ---
print("\n\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

# Per framework
for fw in FRAMEWORKS:
    fw_results = [r for r in results if r["framework"] == fw]
    total_runs = len(fw_results)
    successes = sum(1 for r in fw_results if r["success"])
    avg_time = sum(r["elapsed_s"] for r in fw_results) / total_runs if total_runs else 0
    avg_attempts = sum(r["attempts"] for r in fw_results) / total_runs if total_runs else 0
    struct_fails = sum(r["structural_failures"] for r in fw_results)
    implies_count = sum(1 for r in fw_results if r["has_implies"])
    forall_count = sum(1 for r in fw_results if r["has_forall"])
    exists_count = sum(1 for r in fw_results if r["has_exists"])
    avg_nesting = sum(r["nesting_depth"] for r in fw_results if r["success"]) / max(successes, 1)

    print(f"\n--- {fw} ---")
    print(f"  Success rate: {successes}/{total_runs} ({100*successes/total_runs:.0f}%)")
    print(f"  Avg time: {avg_time:.1f}s")
    print(f"  Avg attempts: {avg_attempts:.1f}")
    print(f"  Structural failures: {struct_fails}")
    print(f"  Uses Implies: {implies_count}/{total_runs}")
    print(f"  Uses ForAll: {forall_count}/{total_runs}")
    print(f"  Uses Exists: {exists_count}/{total_runs}")
    print(f"  Avg nesting (successful): {avg_nesting:.0f}")

# Per requirement
print("\n\n--- By Requirement ---")
for req in REQUIREMENTS:
    print(f"\nReq {req}:")
    for fw in FRAMEWORKS:
        fw_req = [r for r in results if r["framework"] == fw and r["requirement"] == req]
        successes = sum(1 for r in fw_req if r["success"])
        avg_time = sum(r["elapsed_s"] for r in fw_req) / len(fw_req) if fw_req else 0
        print(f"  {fw:15s}: {successes}/{len(fw_req)} success, avg {avg_time:.1f}s")

# Save raw results
output_path = Path(__file__).parent.parent / "results" / "comparison_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nRaw results saved to {output_path}")
