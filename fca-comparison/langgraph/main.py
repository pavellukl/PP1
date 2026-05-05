import sys
sys.path.insert(0, "..")

import json
import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import ValidationError

from schemas import (
    AmbiguityReviewResult,
    EntityExtractionResult,
    RequirementFormula,
    ValidationResult,
    VALID_OPERATIONS,
    EntityType,
    OperationType,
    Operation,
    Cmp,
    Not,
    AllOf,
    AnyOf,
    Implies,
    ForAll,
    Exists,
    EntityVal,
    Constant,
    Diff,
)

TEST_REQUIREMENT = (
    "The software must start to execute the sequence of tasks allocated "
    "to the execution schedule in less than 2 seconds after processor power-up."
)

from prompts import AMBIGUITY_SYSTEM, ENTITY_SYSTEM, FORMULA_SYSTEM, task_prompt

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://172.31.128.1:11434/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:14b")

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=OLLAMA_URL,
    api_key="ollama",
)


# --- State ---

class PipelineState(TypedDict):
    requirement: str
    ambiguity: AmbiguityReviewResult | None
    entities: EntityExtractionResult | None
    formula: RequirementFormula | None
    validation: ValidationResult | None
    formula_attempts: int
    formula_feedback: str


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def llm_call_with_retry(system_prompt: str, user_prompt: str, model_class, max_retries: int = 3):
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries):
        response = llm.invoke(messages, response_format={"type": "json_object"})
        text = response.content

        try:
            data = parse_json_response(text)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            messages.append(AIMessage(content=text))
            messages.append(HumanMessage(
                content=f"Your response was invalid: {e}\n"
                "Please fix and return ONLY valid JSON matching the schema."
            ))

    raise RuntimeError(f"Failed to get valid output after {max_retries} attempts")


# --- Node functions ---

def ambiguity_review(state: PipelineState) -> dict:
    print("\n--- Step 1: Ambiguity Review ---")
    result = llm_call_with_retry(
        system_prompt=AMBIGUITY_SYSTEM,
        user_prompt=task_prompt(f"Review this requirement for ambiguity:\n\n{state['requirement']}"),
        model_class=AmbiguityReviewResult,
    )
    print(result.model_dump_json(indent=2))
    return {"ambiguity": result}


def entity_extraction(state: PipelineState) -> dict:
    print("\n--- Step 2: Entity Extraction ---")
    result = llm_call_with_retry(
        system_prompt=ENTITY_SYSTEM,
        user_prompt=task_prompt(f"Extract entities from this requirement:\n\n{state['requirement']}"),
        model_class=EntityExtractionResult,
    )
    print(result.model_dump_json(indent=2))
    return {"entities": result}


def formula_generation(state: PipelineState) -> dict:
    attempt = state.get("formula_attempts", 0) + 1
    print(f"\n--- Step 3: Requirement Formula (attempt {attempt}) ---")

    prompt = (
        f"Requirement: {state['requirement']}\n\n"
        f"Extracted entities:\n{state['entities'].model_dump_json(indent=2)}"
    )
    feedback = state.get("formula_feedback", "")
    if feedback:
        prompt += f"\n\nPrevious attempt failed validation:\n{feedback}\nPlease fix these issues."

    try:
        result = llm_call_with_retry(
            system_prompt=FORMULA_SYSTEM,
            user_prompt=task_prompt(prompt),
            model_class=RequirementFormula,
        )
        print(result.model_dump_json(indent=2))
        print("\nFormula:")
        print(fmt_formula(result))
        return {"formula": result, "formula_attempts": attempt}
    except RuntimeError as e:
        print(f"LLM failed to produce valid structure: {e}")
        return {"formula": None, "formula_attempts": attempt}


def validation(state: PipelineState) -> dict:
    attempt = state.get("formula_attempts", 1)
    print(f"\n--- Step 4: Validation (attempt {attempt}) ---")

    if state.get("formula") is None:
        result = ValidationResult(valid=False, errors=["Formula generation failed"], confidence=0.0)
        print(result.model_dump_json(indent=2))
        return {"validation": result, "formula_feedback": "Formula generation failed completely. Use ONLY constructs from the schema."}

    errors: list[str] = []
    entity_map = {e.name: e for e in state["entities"].entities}

    def check_predicate(pred, path: str = "formula"):
        if isinstance(pred, Operation):
            if pred.entity not in entity_map:
                errors.append(f"{path}: entity '{pred.entity}' not declared in step 1")
            else:
                entity = entity_map[pred.entity]
                valid_ops = VALID_OPERATIONS.get(entity.type, set())
                if pred.operation not in valid_ops:
                    errors.append(
                        f"{path}: operation '{pred.operation.value}' not valid for "
                        f"entity type {entity.type.value} "
                        f"(valid: {[o.value for o in valid_ops]})"
                    )
        elif isinstance(pred, Cmp):
            check_expression(pred.left, f"{path}.left")
            check_expression(pred.right, f"{path}.right")
        elif isinstance(pred, Not):
            check_predicate(pred.predicate, f"{path}.Not")
        elif isinstance(pred, (AllOf, AnyOf)):
            for i, p in enumerate(pred.predicates):
                check_predicate(p, f"{path}.{pred.pred_type}[{i}]")
        elif isinstance(pred, Implies):
            check_predicate(pred.antecedent, f"{path}.antecedent")
            check_predicate(pred.consequent, f"{path}.consequent")
        elif isinstance(pred, (ForAll, Exists)):
            check_predicate(pred.predicate, f"{path}.{pred.pred_type}({pred.variable})")

    def check_expression(expr, path: str):
        if isinstance(expr, EntityVal):
            if expr.entity not in entity_map:
                errors.append(f"{path}: entity '{expr.entity}' not declared in step 1")

    check_predicate(state["formula"].formula)

    confidence = 0.95 if not errors else 0.3
    result = ValidationResult(valid=len(errors) == 0, errors=errors, confidence=confidence)
    print(result.model_dump_json(indent=2))

    feedback = "\n".join(errors) if errors else ""
    return {"validation": result, "formula_feedback": feedback}


# --- Conditional edge ---

def should_retry_formula(state: PipelineState) -> str:
    if state["validation"].valid:
        print("\nPipeline completed successfully.")
        return "done"
    elif state.get("formula_attempts", 0) >= 3:
        print(f"\nPipeline failed after {state['formula_attempts']} attempts.")
        return "done"
    else:
        print(f"\nValidation failed, retrying step 3...")
        return "retry"


# --- Pretty Printer ---

def fmt_expr(expr) -> str:
    if isinstance(expr, Constant):
        return f"{expr.value}{expr.unit}" if expr.unit else str(expr.value)
    elif isinstance(expr, EntityVal):
        return f"Val({expr.entity}, {expr.time_var})"
    elif isinstance(expr, Diff):
        return f"Diff({expr.t1}, {expr.t2})"
    return str(expr)


def fmt_pred(pred, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(pred, Operation):
        args = f", {', '.join(pred.args)}" if pred.args else ""
        return f"{pad}{pred.operation.value}({pred.time_var}, {pred.entity}{args})"
    elif isinstance(pred, Cmp):
        return f"{pad}Cmp({fmt_expr(pred.left)}, {pred.operator}, {fmt_expr(pred.right)})"
    elif isinstance(pred, Not):
        return f"{pad}Not(\n{fmt_pred(pred.predicate, indent + 1)}\n{pad})"
    elif isinstance(pred, (AllOf, AnyOf)):
        name = pred.pred_type
        items = ",\n".join(fmt_pred(p, indent + 1) for p in pred.predicates)
        return f"{pad}{name}(\n{items}\n{pad})"
    elif isinstance(pred, Implies):
        a = fmt_pred(pred.antecedent, indent + 1)
        c = fmt_pred(pred.consequent, indent + 1)
        return f"{pad}Implies(\n{a},\n{c}\n{pad})"
    elif isinstance(pred, (ForAll, Exists)):
        name = pred.pred_type
        body = fmt_pred(pred.predicate, indent + 1)
        return f"{pad}{name}({pred.variable},\n{body}\n{pad})"
    return f"{pad}{pred}"


def fmt_formula(formula: RequirementFormula) -> str:
    return f"[{formula.time_type} time]\n{fmt_pred(formula.formula)}"


# --- Build Graph ---

graph = StateGraph(PipelineState)

graph.add_node("ambiguity_review", ambiguity_review)
graph.add_node("entity_extraction", entity_extraction)
graph.add_node("formula_generation", formula_generation)
graph.add_node("validation", validation)

graph.add_edge(START, "ambiguity_review")
graph.add_edge("ambiguity_review", "entity_extraction")
graph.add_edge("entity_extraction", "formula_generation")
graph.add_edge("formula_generation", "validation")
graph.add_conditional_edges("validation", should_retry_formula, {
    "retry": "formula_generation",
    "done": END,
})

pipeline = graph.compile()


# --- Run ---

if __name__ == "__main__":
    from pathlib import Path

    REQ_DIR = Path(__file__).resolve().parent.parent.parent / "req"

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.isdigit() or (len(arg) == 2 and arg.isdigit()):
            path = REQ_DIR / f"{int(arg):02d}.md"
        else:
            path = Path(arg)
        requirement = path.read_text().strip()
    else:
        requirement = TEST_REQUIREMENT

    print("=" * 60)
    print("REQUIREMENT:", requirement)
    print("=" * 60)

    result = pipeline.invoke({
        "requirement": requirement,
        "ambiguity": None,
        "entities": None,
        "formula": None,
        "validation": None,
        "formula_attempts": 0,
        "formula_feedback": "",
    })
