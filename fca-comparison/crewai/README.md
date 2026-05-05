# CrewAI — Implementation Notes

## Lines of Code
- main.py: 398 lines (pipeline + agents + tasks + validation + pretty printer)
- schemas.py: 178 lines (shared)
- Framework-specific: ~220 lines (most of any framework)

## Setup
- `pip install crewai` — heaviest install by far. Pulls in chromadb, lancedb, onnxruntime, kubernetes, PDF parsers, etc.
- Dependency conflicts with other frameworks' OpenTelemetry versions (logfire warnings)
- Ollama connection via `LLM(model="ollama/model_name", base_url="...")`

## Structured Output
- No built-in Pydantic validation via the API for Ollama
- CrewAI has `output_pydantic` on Tasks but it didn't work reliably with Ollama
- Used guardrails for structural validation — functions that return (bool, output_or_error)
- Still needed manual JSON parsing on top of guardrails to get Pydantic objects

## Intermediate Inspection
- Each Crew returns a result object with `.raw` containing the LLM's text output
- But tasks in a Crew auto-chain — output of task 1 becomes context for task 2, with no control over formatting
- We ran each step as a separate single-task Crew to maintain control over intermediate outputs

## Retry Behavior
- Guardrails with `guardrail_max_retries` handle structural retry — error is fed back to the agent
- Semantic retry (cross-step validation) is manual, same as Haystack and LangGraph
- The guardrail function interface `(bool, output_or_error)` is simple but limited

## Model Swapping
- Change `model` parameter on `LLM()` — prefix with provider (e.g., `"ollama/model"`, `"openai/gpt-4"`)
- Clean interface for model switching

## Local Execution
- Works with Ollama via the `LLM` class
- Tracing/telemetry tries to phone home by default — prints noisy "Tracing Preference Saved" boxes on first run
- Need to explicitly disable tracing for offline environments

## Learning Curve
- Highest of all four frameworks
- Must understand: Agent (role/goal/backstory), Task (description/expected_output), Crew (orchestration), LLM, guardrails
- The role/goal/backstory pattern is designed for multi-agent conversation, not structured data extraction
- Agent "personas" add prompt overhead that encourages verbose, less precise outputs

## Extensibility
- Adding a step = new Agent + new Task + new Crew (or add to existing Crew)
- Most boilerplate per step of all frameworks
- The Crew abstraction assumes agents collaborate — doesn't match our pipeline's linear data flow

## Error Clarity
- CrewAI adds its own error wrapping — stack traces can be deep
- Guardrail errors are clear (you write the error message yourself)
- Tracing/telemetry warnings clutter the output

## Schema Swapability
- Same as Haystack/LangGraph — Pydantic models easy to change, prompts need manual updates
- The role/goal/backstory fields on Agents also need updating if the formalism changes

## Observations
- CrewAI is designed for multi-agent collaboration (agents discussing, delegating, disagreeing) — our pipeline is a linear data transformation, which is the wrong use case for it
- The role/goal/backstory pattern made the ambiguity reviewer much more verbose (5 notes vs 0-1) — the "agent persona" framing encourages elaboration over precision
- We had to run each step as a separate single-task Crew to maintain control — defeating the purpose of the Crew abstraction
- Heaviest dependency footprint by far (chromadb, onnxruntime, kubernetes) for features we don't use
- Tracing boxes and telemetry warnings make the output noisy
- Slowest of all four implementations due to CrewAI's internal orchestration overhead
