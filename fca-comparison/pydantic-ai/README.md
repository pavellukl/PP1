# PydanticAI — Implementation Notes

## Lines of Code
- main.py: 317 lines (pipeline + agents + validation + pretty printer)
- schemas.py: 178 lines (shared across all frameworks)
- Framework-specific: ~140 lines (excluding shared schemas, validation, and pretty printer)

## Setup
- `pip install pydantic-ai` — pulls in many transitive dependencies but works out of the box
- Ollama connection via OpenAI-compatible API required `OpenAIProvider(base_url=..., api_key="ollama")`
- Had to set `OpenAIModelProfile` to use `prompted` output mode instead of tool calling, because Ollama's tool calling support is unreliable

## Structured Output
- Define a Pydantic model, pass it as `output_type` to `Agent` — that's it
- PydanticAI auto-generates JSON schema from the Pydantic model and sends it to the LLM
- Discriminated unions (recursive predicate tree) work correctly
- Field `description` strings are included in the JSON schema sent to the LLM — they double as LLM instructions

## Intermediate Inspection
- Each `agent.run()` returns a result object — full visibility between steps
- Pipeline is just Python async code, so logging/printing between steps is trivial

## Retry Behavior
- **Structural retries** (built-in): `retries=3` on Agent. If LLM returns invalid JSON or wrong types, PydanticAI retries automatically with the validation error in the prompt. Zero code needed.
- **Semantic retries** (manual): Our validation step checks cross-step consistency (entity references, operation-type compatibility). On failure, we re-run step 3 with error feedback appended to the prompt. This is our code.
- On complex requirements (req 9), the 14B model exhausted structural retries — it kept inventing operations not in the schema. PydanticAI raised `UnexpectedModelBehavior` which we catch and handle.

## Model Swapping
- Change `MODEL_NAME` and `OLLAMA_BASE_URL` — two env vars
- For cloud APIs: swap `OpenAIProvider` for the built-in provider (Anthropic, OpenAI, etc.)
- Profile settings (JSON mode vs tool calling) may need to change per provider

## Local Execution
- Works fully local with Ollama
- Tool calling mode doesn't work reliably with Ollama — must use JSON/prompted mode via `OpenAIModelProfile`

## Learning Curve
- Very low. If you know Pydantic, you know 80% of PydanticAI
- Core pattern: define model, create Agent, call `agent.run()` — three concepts
- The Agent/Provider/Profile split is slightly confusing (which class does what) but well-documented

## Extensibility
- Adding a step = define a Pydantic model + create an Agent + add 3-4 lines in the pipeline function
- No framework boilerplate — it's just Python functions calling agents

## Error Clarity
- Pydantic validation errors are detailed (field paths, expected vs actual types)
- PydanticAI wraps them into `ToolRetryError` with all validation errors listed
- When retries are exhausted, the `UnexpectedModelBehavior` exception is clear about what happened

## Schema Swapability
- Change the Pydantic models in schemas.py — everything downstream adapts automatically
- The JSON schema sent to the LLM updates automatically from the Pydantic models
- System prompts may need manual updates if they reference specific fields

## Observations
- The recursive formula schema (discriminated unions) is the hardest part for the local model — simpler requirements (6, 2, 8) work well, complex ones (9) fail
- The example in the system prompt gets copied verbatim by weaker models — using placeholders helps
- Ambiguity review step works but should only flag issues, not rewrite requirements (the LLM adds hallucinated details)
- Validation doesn't check operation argument count/types — only operation-entity type compatibility
- Modifiers are in the schema but unvalidated (formalism says "needs to be parameterized")
