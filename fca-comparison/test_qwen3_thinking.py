"""Run the pipeline with qwen3 via Ollama native API to see thinking output."""
import json
import time
import requests
from pydantic import ValidationError
from schemas import (
    AmbiguityReviewResult, EntityExtractionResult, RequirementFormula,
    VALID_OPERATIONS, Operation, Cmp, Not, AllOf, AnyOf, Implies,
    ForAll, Exists, EntityVal, Constant, Diff,
)
from prompts import AMBIGUITY_SYSTEM, ENTITY_SYSTEM, FORMULA_SYSTEM, task_prompt

OLLAMA = "http://172.31.128.1:11434/api/chat"
MODEL = "qwen3:14b"
REQ = (
    "The software must toggle valid_range, within 500 ns of a write to d, "
    "if ALL the following conditions are satisfied:\n"
    "- The difference between last Maximum Peak and Minimum Peak of d is higher than 655;\n"
    "- It is the second write to d after the Minimum Peak detection;\n"
    "- Signed value of d is higher than the signed value of last d."
)


def call_ollama(system, user, model_class, max_retries=3):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for attempt in range(max_retries):
        r = requests.post(OLLAMA, json={
            "model": MODEL, "stream": False,
            "messages": messages,
        })
        msg = r.json()["message"]
        content = msg["content"]
        thinking = msg.get("thinking", "")

        if thinking:
            print(f"\n  --- THINKING ({len(thinking)} chars) ---")
            print(thinking)
            print(f"  --- END THINKING ---\n")

        text = content

        try:
            # Strip markdown code fences if present
            if "```" in text:
                lines = text.split("\n")
                in_block = False
                json_lines = []
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                if json_lines:
                    text = "\n".join(json_lines)
            data = json.loads(text)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  [RETRY {attempt+1}] {str(e)[:300]}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Invalid: {e}\nReturn ONLY valid JSON."})

    raise RuntimeError(f"Failed after {max_retries} attempts")


def fmt_expr(expr):
    if isinstance(expr, Constant):
        return f"{expr.value}{expr.unit}" if expr.unit else str(expr.value)
    elif isinstance(expr, EntityVal):
        return f"Val({expr.entity}, {expr.time_var})"
    elif isinstance(expr, Diff):
        return f"Diff({expr.t1}, {expr.t2})"
    return str(expr)


def fmt_pred(pred, indent=0):
    pad = "  " * indent
    if isinstance(pred, Operation):
        args = f", {', '.join(pred.args)}" if pred.args else ""
        return f"{pad}{pred.operation.value}({pred.time_var}, {pred.entity}{args})"
    elif isinstance(pred, Cmp):
        return f"{pad}Cmp({fmt_expr(pred.left)}, {pred.operator}, {fmt_expr(pred.right)})"
    elif isinstance(pred, Not):
        return f"{pad}Not(\n{fmt_pred(pred.predicate, indent+1)}\n{pad})"
    elif isinstance(pred, (AllOf, AnyOf)):
        items = ",\n".join(fmt_pred(p, indent+1) for p in pred.predicates)
        return f"{pad}{pred.pred_type}(\n{items}\n{pad})"
    elif isinstance(pred, Implies):
        return f"{pad}Implies(\n{fmt_pred(pred.antecedent, indent+1)},\n{fmt_pred(pred.consequent, indent+1)}\n{pad})"
    elif isinstance(pred, (ForAll, Exists)):
        return f"{pad}{pred.pred_type}({pred.variable},\n{fmt_pred(pred.predicate, indent+1)}\n{pad})"
    return f"{pad}{pred}"


print("=" * 60)
print("REQUIREMENT:", REQ)
print("=" * 60)

# Step 1
print("\n--- Step 1: Ambiguity Review ---")
t = time.time()
ambiguity = call_ollama(AMBIGUITY_SYSTEM, task_prompt(f"Review this requirement for ambiguity:\n\n{REQ}"), AmbiguityReviewResult)
print(f"({time.time()-t:.1f}s)")
print(ambiguity.model_dump_json(indent=2))

# Step 2
print("\n--- Step 2: Entity Extraction ---")
t = time.time()
entities = call_ollama(ENTITY_SYSTEM, task_prompt(f"Extract entities from this requirement. IMPORTANT: every entity name must be unique — if two concepts share a word, add a suffix like _mode, _value, _event.\n\n{REQ}"), EntityExtractionResult)
print(f"({time.time()-t:.1f}s)")
print(entities.model_dump_json(indent=2))

# Step 3
print("\n--- Step 3: Requirement Formula ---")
t = time.time()
prompt = f"Requirement: {REQ}\n\nExtracted entities:\n{entities.model_dump_json(indent=2)}"
try:
    formula = call_ollama(FORMULA_SYSTEM, task_prompt(prompt), RequirementFormula)
    print(f"({time.time()-t:.1f}s)")
    print(f"\nFormula:\n[{formula.time_type} time]")
    print(fmt_pred(formula.formula))

    # Step 4
    print("\n--- Step 4: Validation ---")
    errors = []
    entity_map = {e.name: e for e in entities.entities}
    def check_pred(pred, path="formula"):
        if isinstance(pred, Operation):
            if pred.entity not in entity_map:
                errors.append(f"{path}: entity '{pred.entity}' not declared")
            elif pred.operation not in VALID_OPERATIONS.get(entity_map[pred.entity].type, set()):
                errors.append(f"{path}: op '{pred.operation.value}' invalid for {entity_map[pred.entity].type.value}")
        elif isinstance(pred, (AllOf, AnyOf)):
            for i, p in enumerate(pred.predicates):
                check_pred(p, f"{path}[{i}]")
        elif isinstance(pred, Implies):
            check_pred(pred.antecedent, f"{path}.ant")
            check_pred(pred.consequent, f"{path}.con")
        elif isinstance(pred, (ForAll, Exists)):
            check_pred(pred.predicate, f"{path}.body")
    check_pred(formula.formula)
    print(f"Valid: {len(errors) == 0}")
    for e in errors:
        print(f"  {e}")
except RuntimeError as e:
    print(f"({time.time()-t:.1f}s)")
    print(f"FAILED: {e}")
