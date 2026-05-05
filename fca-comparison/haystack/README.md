# Haystack — Implementation Notes

## Lines of Code
- main.py: 383 lines (pipeline + components + validation + pretty printer)
- schemas.py: 178 lines (shared)
- Framework-specific: ~205 lines (more than PydanticAI due to manual JSON handling)

## Setup
- `pip install haystack-ai` — clean install, fewer transitive dependencies than PydanticAI
- API keys must be wrapped in `Secret.from_token()` — extra boilerplate
- Uses `OpenAIChatGenerator` with `api_base_url` pointing to Ollama

## Structured Output
- No built-in Pydantic validation for Ollama/OpenAI-compatible endpoints
- Had to write manual JSON parsing + Pydantic validation + retry logic (~25 lines)
- The full Pydantic JSON schema (9670 chars for recursive formula) is too large to include in the system prompt — had to use compact hand-written schema descriptions instead
- PydanticAI handles all of this automatically via the API's structured output mechanism

## Intermediate Inspection
- Components have typed inputs/outputs — you can inspect between steps
- When using the Pipeline graph, intermediate results are accessible via the pipeline result dict
- In practice, we ran components manually (not wired into a Pipeline graph) because the retry loop between formula generation and validation doesn't fit Haystack's DAG model

## Retry Behavior
- No built-in retry on validation failure — entirely manual
- Wrote `llm_call_with_retry()` that appends validation errors to the conversation and retries
- This is structural retry only (bad JSON) — semantic retry (cross-step validation) is a separate manual loop, same as PydanticAI
- More code, same result

## Model Swapping
- Change `model` and `api_base_url` parameters on `OpenAIChatGenerator`
- Haystack has dedicated generators for different providers (OpenAI, Anthropic, etc.) — switching providers may mean switching classes, not just parameters

## Local Execution
- Works with Ollama via `OpenAIChatGenerator` with custom `api_base_url`
- JSON mode (`response_format: json_object`) works with Ollama

## Learning Curve
- Steeper than PydanticAI — need to understand Component decorator pattern, Pipeline graph, Secret types
- The `@component` decorator with `@component.output_types()` is unintuitive at first
- Pipeline graph wiring (`pipeline.connect("a.output", "b.input")`) is powerful for complex DAGs but overkill for a linear pipeline

## Extensibility
- Adding a step = write a new `@component` class + wire it into the pipeline
- More boilerplate per step than PydanticAI (class + decorator + output_types)
- Pipeline is serializable to YAML — useful for deployment but not relevant for this prototype

## Error Clarity
- JSON parsing errors are Python standard — clear enough
- Pydantic validation errors are detailed (same as PydanticAI, since we use the same models)
- Haystack's own errors (Secret type, component wiring) are less helpful

## Schema Swapability
- Same as PydanticAI for the Pydantic models themselves
- But the compact schema descriptions in prompts are hand-written — changing the schema means updating prompt text manually (PydanticAI auto-generates from the model)

## Observations
- Haystack's strength is pipeline-as-graph with YAML serialization — not relevant for our use case
- The component pattern adds boilerplate without benefit for a linear LLM pipeline
- We couldn't use Haystack's Pipeline graph for the full flow because the retry loop between formula generation and validation is a cycle, not a DAG
- Essentially we're writing plain Python with Haystack as just an LLM API wrapper — the framework isn't adding much value here
