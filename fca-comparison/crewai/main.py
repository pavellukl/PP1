import sys
sys.path.insert(0, "..")

import json
import os

from crewai import Agent, Task, Crew, LLM
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

from prompts import AMBIGUITY_SYSTEM, ENTITY_SYSTEM, FORMULA_SYSTEM

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://172.31.128.1:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:14b")

llm = LLM(
    model=f"ollama/{MODEL_NAME}",
    base_url=OLLAMA_URL,
)


# --- Guardrails ---

def validate_ambiguity(output):
    try:
        if hasattr(output, 'raw'):
            text = output.raw
        else:
            text = str(output)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        data = json.loads(text)
        AmbiguityReviewResult.model_validate(data)
        return (True, output)
    except Exception as e:
        return (False, f"Invalid output: {e}. Return ONLY valid JSON.")


def validate_entities(output):
    try:
        if hasattr(output, 'raw'):
            text = output.raw
        else:
            text = str(output)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        data = json.loads(text)
        EntityExtractionResult.model_validate(data)
        return (True, output)
    except Exception as e:
        return (False, f"Invalid output: {e}. Return ONLY valid JSON.")


def validate_formula(output):
    try:
        if hasattr(output, 'raw'):
            text = output.raw
        else:
            text = str(output)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        data = json.loads(text)
        RequirementFormula.model_validate(data)
        return (True, output)
    except Exception as e:
        return (False, f"Invalid output: {e}. Return ONLY valid JSON.")


def parse_output(text: str, model_class):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    data = json.loads(text)
    return model_class.model_validate(data)


# --- Semantic Validation (rule-based) ---

def validate_semantics(entities: EntityExtractionResult, formula: RequirementFormula) -> ValidationResult:
    errors: list[str] = []
    entity_map = {e.name: e for e in entities.entities}

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

    check_predicate(formula.formula)

    confidence = 0.95 if not errors else 0.3
    return ValidationResult(valid=len(errors) == 0, errors=errors, confidence=confidence)


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


# --- Pipeline ---

def run_pipeline(requirement: str, max_retries: int = 2) -> None:
    print("=" * 60)
    print("REQUIREMENT:", requirement)
    print("=" * 60)

    # --- Agents ---
    ambiguity_agent = Agent(
        role="Ambiguity Reviewer",
        goal="Review requirements against the ambiguity checklist and flag genuine issues",
        backstory=AMBIGUITY_SYSTEM,
        llm=llm,
        verbose=False,
    )

    entity_agent = Agent(
        role="Entity Extractor",
        goal="Extract all entities from a requirement, producing precise and complete results",
        backstory=ENTITY_SYSTEM,
        llm=llm,
        verbose=False,
    )

    formula_agent = Agent(
        role="Requirement Formalizer",
        goal="Translate a requirement and its entities into a formal predicate tree",
        backstory=FORMULA_SYSTEM,
        llm=llm,
        verbose=False,
    )

    # --- Step 1: Ambiguity Review ---
    print("\n--- Step 1: Ambiguity Review ---")
    ambiguity_task = Task(
        description=(
            f"Review this requirement for ambiguity:\n\n{requirement}"
        ),
        expected_output="JSON with ambiguity notes",
        agent=ambiguity_agent,
        guardrails=[validate_ambiguity],
    )

    ambiguity_crew = Crew(agents=[ambiguity_agent], tasks=[ambiguity_task], verbose=False)
    ambiguity_result = ambiguity_crew.kickoff()
    ambiguity = parse_output(ambiguity_result.raw, AmbiguityReviewResult)
    print(ambiguity.model_dump_json(indent=2))

    # --- Step 2: Entity Extraction ---
    print("\n--- Step 2: Entity Extraction ---")
    entity_task = Task(
        description=f"Extract entities from this requirement:\n\n{requirement}",
        expected_output="JSON with extracted entities",
        agent=entity_agent,
        guardrails=[validate_entities],
    )

    entity_crew = Crew(agents=[entity_agent], tasks=[entity_task], verbose=False)
    entity_result = entity_crew.kickoff()
    entities = parse_output(entity_result.raw, EntityExtractionResult)
    print(entities.model_dump_json(indent=2))

    # --- Step 3 + 4: Formula + Validation with retry ---
    feedback = ""
    for attempt in range(1, max_retries + 2):
        print(f"\n--- Step 3: Requirement Formula (attempt {attempt}) ---")

        prompt = (
            f"Requirement: {requirement}\n\n"
            f"Extracted entities:\n{entities.model_dump_json(indent=2)}"
        )
        if feedback:
            prompt += f"\n\nPrevious attempt failed validation:\n{feedback}\nPlease fix these issues."

        formula_task = Task(
            description=prompt,
            expected_output="JSON with formalized requirement as predicate tree",
            agent=formula_agent,
            guardrails=[validate_formula],
        )

        try:
            formula_crew = Crew(agents=[formula_agent], tasks=[formula_task], verbose=False)
            formula_result = formula_crew.kickoff()
            formula = parse_output(formula_result.raw, RequirementFormula)
        except Exception as e:
            print(f"LLM failed to produce valid structure: {type(e).__name__}")
            feedback = "Formula generation failed. Use ONLY constructs from the schema."
            continue

        print(formula.model_dump_json(indent=2))
        print("\nFormula:")
        print(fmt_formula(formula))

        # Step 4: Validation
        print(f"\n--- Step 4: Validation (attempt {attempt}) ---")
        validation = validate_semantics(entities, formula)
        print(validation.model_dump_json(indent=2))

        if validation.valid:
            print("\nPipeline completed successfully.")
            return

        feedback = "\n".join(validation.errors)
        print(f"\nValidation failed, retrying step 3...")

    print(f"\nPipeline failed after {max_retries + 1} attempts.")


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

    run_pipeline(requirement)
