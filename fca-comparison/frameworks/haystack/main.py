import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from haystack import Pipeline, component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from pydantic import ValidationError

from schemas import (
    AmbiguityReviewResult,
    FlavourResult,
    EntityExtractionResult,
    Requirement,
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
    Happening,
    HasHappened,
    Val,
    ValBefore,
    Constant,
    ArithOp,
    Now,
    Prev,
    Next,
    Diff,
    EvtOccCount,
    MkInterval,
    Always,
    Eventually,
    Initial,
    Causes,
    CausesWithin,
    Sequence,
    TraceAllOf,
)

from prompts import AMBIGUITY_SYSTEM, FLAVOUR_SYSTEM, ENTITY_SYSTEM, FORMULA_SYSTEM, task_prompt

TEST_REQUIREMENT = (
    "The software must start to execute the sequence of tasks allocated "
    "to the execution schedule in less than 2 seconds after processor power-up."
)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://172.31.128.1:11434/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "phi4-reasoning:plus")


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def llm_call_with_retry(generator, system_prompt: str, user_prompt: str, model_class, max_retries: int = 3):
    messages = [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(user_prompt),
    ]

    for attempt in range(max_retries):
        result = generator.run(messages=messages)
        reply = result["replies"][0]
        text = reply.text

        try:
            data = parse_json_response(text)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            error_msg = str(e)
            messages.append(ChatMessage.from_assistant(text))
            messages.append(ChatMessage.from_user(
                f"Your response was invalid: {error_msg}\n"
                "Please fix and return ONLY valid JSON matching the schema."
            ))

    raise RuntimeError(f"Failed to get valid output after {max_retries} attempts")


# --- Components ---

@component
class AmbiguityReviewer:
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token("ollama"),
            model=MODEL_NAME,
            api_base_url=OLLAMA_URL,
            generation_kwargs={"response_format": {"type": "json_object"}},
        )
        self.system_prompt = AMBIGUITY_SYSTEM

    @component.output_types(ambiguity=AmbiguityReviewResult, requirement=str)
    def run(self, requirement: str):
        result = llm_call_with_retry(
            self.generator, self.system_prompt,
            task_prompt(f"Review this requirement for ambiguity:\n\n{requirement}"),
            AmbiguityReviewResult,
        )
        return {"ambiguity": result, "requirement": requirement}


@component
class FlavourClassifier:
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token("ollama"),
            model=MODEL_NAME,
            api_base_url=OLLAMA_URL,
            generation_kwargs={"response_format": {"type": "json_object"}},
        )
        self.system_prompt = FLAVOUR_SYSTEM

    @component.output_types(flavour=FlavourResult, requirement=str)
    def run(self, requirement: str, ambiguity_context: str = ""):
        result = llm_call_with_retry(
            self.generator, self.system_prompt,
            task_prompt(f"Determine the time model for this requirement:\n\n{requirement}{ambiguity_context}"),
            FlavourResult,
        )
        return {"flavour": result, "requirement": requirement}


@component
class EntityExtractor:
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token("ollama"),
            model=MODEL_NAME,
            api_base_url=OLLAMA_URL,
            generation_kwargs={"response_format": {"type": "json_object"}},
        )
        self.system_prompt = ENTITY_SYSTEM

    @component.output_types(entities=EntityExtractionResult, requirement=str)
    def run(self, requirement: str, ambiguity_context: str = ""):
        result = llm_call_with_retry(
            self.generator, self.system_prompt,
            task_prompt(
                f"Extract entities from this requirement:\n\n{requirement}"
                f"{ambiguity_context}"
            ),
            EntityExtractionResult,
        )
        return {"entities": result, "requirement": requirement}


@component
class FormulaGenerator:
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token("ollama"),
            model=MODEL_NAME,
            api_base_url=OLLAMA_URL,
            generation_kwargs={"response_format": {"type": "json_object"}},
        )
        self.system_prompt = FORMULA_SYSTEM

    @component.output_types(formula=Requirement, entities=EntityExtractionResult)
    def run(self, requirement: str, entities: EntityExtractionResult,
            flavour: str = "", ambiguity_context: str = "", feedback: str = ""):
        prompt = (
            f"Requirement: {requirement}\n\n"
            f"Time model: {flavour}\n\n"
            f"Extracted entities:\n{entities.model_dump_json(indent=2)}"
            f"{ambiguity_context}"
        )
        if feedback:
            prompt += f"\n\nPrevious attempt failed validation:\n{feedback}\nPlease fix these issues."
        result = llm_call_with_retry(
            self.generator, self.system_prompt, task_prompt(prompt), Requirement,
        )
        return {"formula": result, "entities": entities}


# --- Validation (rule-based) ---

def validate(entities: EntityExtractionResult, req: Requirement) -> ValidationResult:
    errors: list[str] = []
    entity_map = {e.name: e for e in entities.entities}

    def check_predicate(pred, path: str = "constraint"):
        if isinstance(pred, Operation):
            if pred.entity not in entity_map:
                errors.append(f"{path}: entity '{pred.entity}' not declared")
            else:
                entity = entity_map[pred.entity]
                valid_ops = VALID_OPERATIONS.get(entity.type, set())
                if pred.operation not in valid_ops:
                    errors.append(
                        f"{path}: operation '{pred.operation.value}' not valid for "
                        f"entity type {entity.type.value} "
                        f"(valid: {[o.value for o in valid_ops]})"
                    )
        elif isinstance(pred, Happening):
            if pred.entity not in entity_map:
                errors.append(f"{path}: entity '{pred.entity}' not declared")
            elif entity_map[pred.entity].type != EntityType.EVENT:
                errors.append(f"{path}: Happening requires Event entity, got {entity_map[pred.entity].type.value}")
        elif isinstance(pred, HasHappened):
            if pred.entity not in entity_map:
                errors.append(f"{path}: entity '{pred.entity}' not declared")
            elif entity_map[pred.entity].type != EntityType.EVENT:
                errors.append(f"{path}: HasHappened requires Event entity, got {entity_map[pred.entity].type.value}")
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
            check_predicate(pred.predicate, f"{path}.{pred.pred_type}")

    def check_expression(expr, path: str):
        if isinstance(expr, Val):
            if expr.entity not in entity_map:
                errors.append(f"{path}: entity '{expr.entity}' not declared")
        elif isinstance(expr, ValBefore):
            if expr.entity not in entity_map:
                errors.append(f"{path}: entity '{expr.entity}' not declared")
        elif isinstance(expr, EvtOccCount):
            if expr.entity not in entity_map:
                errors.append(f"{path}: entity '{expr.entity}' not declared")
        elif isinstance(expr, ArithOp):
            check_expression(expr.left, f"{path}.left")
            check_expression(expr.right, f"{path}.right")

    def check_constraint(constraint, path: str = "constraint"):
        if isinstance(constraint, Always):
            check_predicate(constraint.predicate, f"{path}.Always")
        elif isinstance(constraint, Eventually):
            check_predicate(constraint.predicate, f"{path}.Eventually")
        elif isinstance(constraint, Initial):
            check_predicate(constraint.predicate, f"{path}.Initial")
        elif isinstance(constraint, Causes):
            check_predicate(constraint.condition, f"{path}.Causes.condition")
            check_predicate(constraint.effect, f"{path}.Causes.effect")
        elif isinstance(constraint, CausesWithin):
            check_predicate(constraint.condition, f"{path}.CausesWithin.condition")
            check_predicate(constraint.effect, f"{path}.CausesWithin.effect")
        elif isinstance(constraint, Sequence):
            for i, step in enumerate(constraint.steps):
                check_predicate(step, f"{path}.Sequence[{i}]")
        elif isinstance(constraint, TraceAllOf):
            for i, c in enumerate(constraint.constructs):
                check_constraint(c, f"{path}.TraceAllOf[{i}]")

    check_constraint(req.constraint)

    confidence = 0.95 if not errors else 0.3
    return ValidationResult(valid=len(errors) == 0, errors=errors, confidence=confidence)


# --- Pretty Printer ---

def fmt_expr(expr) -> str:
    if isinstance(expr, Constant):
        return f"{expr.value}{expr.unit}" if expr.unit else str(expr.value)
    elif isinstance(expr, Val):
        return f"Val({expr.entity}, {fmt_expr(expr.time)})"
    elif isinstance(expr, ValBefore):
        return f"ValBefore({expr.entity}, {fmt_expr(expr.time)})"
    elif isinstance(expr, Now):
        return "Now"
    elif isinstance(expr, Prev):
        return f"Prev({fmt_expr(expr.time)})"
    elif isinstance(expr, Next):
        return f"Next({fmt_expr(expr.time)})"
    elif isinstance(expr, Diff):
        return f"Diff({fmt_expr(expr.t1)}, {fmt_expr(expr.t2)})"
    elif isinstance(expr, ArithOp):
        return f"({fmt_expr(expr.left)} {expr.operator} {fmt_expr(expr.right)})"
    elif isinstance(expr, EvtOccCount):
        return f"EvtOccCount({expr.entity}, {fmt_expr(expr.interval)})"
    elif isinstance(expr, MkInterval):
        l = "(" if expr.left_open else "["
        r = ")" if expr.right_open else "]"
        return f"{l}{fmt_expr(expr.t1)}, {fmt_expr(expr.t2)}{r}"
    return str(expr)


def fmt_pred(pred, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(pred, Operation):
        args_str = f", {', '.join(fmt_expr(a) for a in pred.args)}" if pred.args else ""
        return f"{pad}{pred.operation.value}({pred.entity}{args_str})"
    elif isinstance(pred, Happening):
        return f"{pad}Happening({pred.entity}, {fmt_expr(pred.time)})"
    elif isinstance(pred, HasHappened):
        return f"{pad}HasHappened({pred.entity}, {fmt_expr(pred.time)})"
    elif isinstance(pred, Cmp):
        return f"{pad}Cmp({fmt_expr(pred.left)}, {pred.operator}, {fmt_expr(pred.right)})"
    elif isinstance(pred, Not):
        return f"{pad}Not(\n{fmt_pred(pred.predicate, indent + 1)}\n{pad})"
    elif isinstance(pred, (AllOf, AnyOf)):
        items = ",\n".join(fmt_pred(p, indent + 1) for p in pred.predicates)
        return f"{pad}{pred.pred_type}(\n{items}\n{pad})"
    elif isinstance(pred, Implies):
        a = fmt_pred(pred.antecedent, indent + 1)
        c = fmt_pred(pred.consequent, indent + 1)
        return f"{pad}Implies(\n{a},\n{c}\n{pad})"
    elif isinstance(pred, (ForAll, Exists)):
        body = fmt_pred(pred.predicate, indent + 1)
        return f"{pad}{pred.pred_type}({pred.variable},\n{body}\n{pad})"
    return f"{pad}{pred}"


def fmt_constraint(constraint, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(constraint, Always):
        return f"{pad}Always(\n{fmt_pred(constraint.predicate, indent + 1)}\n{pad})"
    elif isinstance(constraint, Eventually):
        return f"{pad}Eventually(\n{fmt_pred(constraint.predicate, indent + 1)}\n{pad})"
    elif isinstance(constraint, Initial):
        return f"{pad}Initial(\n{fmt_pred(constraint.predicate, indent + 1)}\n{pad})"
    elif isinstance(constraint, Causes):
        cond = fmt_pred(constraint.condition, indent + 1)
        eff = fmt_pred(constraint.effect, indent + 1)
        return f"{pad}Causes(\n{cond},\n{eff}\n{pad})"
    elif isinstance(constraint, CausesWithin):
        cond = fmt_pred(constraint.condition, indent + 1)
        eff = fmt_pred(constraint.effect, indent + 1)
        return f"{pad}CausesWithin(\n{cond},\n{eff},\n{pad}  {constraint.duration}{constraint.duration_unit}\n{pad})"
    elif isinstance(constraint, Sequence):
        steps = ",\n".join(fmt_pred(s, indent + 1) for s in constraint.steps)
        return f"{pad}Sequence(\n{steps}\n{pad})"
    elif isinstance(constraint, TraceAllOf):
        items = ",\n".join(fmt_constraint(c, indent + 1) for c in constraint.constructs)
        return f"{pad}TraceAllOf(\n{items}\n{pad})"
    return f"{pad}{constraint}"


# --- Pipeline with manual retry loop ---

def run_pipeline(requirement: str, max_retries: int = 2) -> None:
    print("=" * 60)
    print("REQUIREMENT:", requirement)
    print("=" * 60)

    # Step 1: Ambiguity Review
    print("\n--- Step 1: Ambiguity Review ---")
    reviewer = AmbiguityReviewer()
    review_result = reviewer.run(requirement=requirement)
    ambiguity = review_result["ambiguity"]
    print(ambiguity.model_dump_json(indent=2))

    ambiguity_context = ""
    if ambiguity.notes:
        ambiguity_context = "\n\nAmbiguity notes from prior review:\n" + "\n".join(
            f"- [{n.severity}] {n.category}: {n.text}" for n in ambiguity.notes
        )

    # Step 2: Flavour (time model)
    print("\n--- Step 2: Flavour Extraction ---")
    classifier = FlavourClassifier()
    flavour_result = classifier.run(requirement=requirement, ambiguity_context=ambiguity_context)
    flavour = flavour_result["flavour"]
    print(f"Flavour: {flavour.flavour}")

    # Step 3: Entity Extraction
    print("\n--- Step 3: Entity Extraction ---")
    extractor = EntityExtractor()
    extract_result = extractor.run(requirement=requirement, ambiguity_context=ambiguity_context)
    entities = extract_result["entities"]
    print(entities.model_dump_json(indent=2))

    # Step 4 + Validation with retry
    formula_gen = FormulaGenerator()
    feedback = ""
    for attempt in range(1, max_retries + 2):
        print(f"\n--- Step 4: Constraint Formalization (attempt {attempt}) ---")

        try:
            gen_result = formula_gen.run(
                requirement=requirement,
                entities=entities,
                flavour=flavour.flavour,
                ambiguity_context=ambiguity_context,
                feedback=feedback,
            )
        except RuntimeError as e:
            print(f"LLM failed to produce valid structure: {e}")
            feedback = "LLM could not produce valid JSON matching the schema. Use ONLY the constructs from the schema."
            continue

        formula = gen_result["formula"]
        print(f"\n[{formula.flavour}]")
        print(fmt_constraint(formula.constraint))

        # Validation
        print(f"\n--- Validation (attempt {attempt}) ---")
        validation = validate(entities, formula)
        print(validation.model_dump_json(indent=2))

        if validation.valid:
            print("\nPipeline completed successfully.")
            return

        feedback = "\n".join(validation.errors)
        print(f"\nValidation failed, retrying step 4...")

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
