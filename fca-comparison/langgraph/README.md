# LangGraph — Implementation Notes

## Lines of Code
- main.py: 389 lines (graph + nodes + validation + pretty printer)
- schemas.py: 178 lines (shared)
- Framework-specific: ~210 lines (similar to Haystack)

## Setup
- `pip install langgraph langchain-openai` — pulls in the LangChain ecosystem (langchain-core, langsmith, etc.)
- `ChatOpenAI` with `base_url` for Ollama — straightforward, no Secret wrapper needed

## Structured Output
- Same situation as Haystack — no built-in Pydantic validation for Ollama
- Manual JSON parsing + Pydantic validation + retry (same `llm_call_with_retry` pattern)
- LangChain has `.with_structured_output()` but it uses tool calling, which doesn't work reliably with Ollama
- Schema described in prompt text, not sent via API

## Intermediate Inspection
- State is a TypedDict — all intermediate results are in the state dict at any point
- Nodes read from and write to state — full visibility
- Can add print/logging in any node function

## Retry Behavior
- Same as Haystack for structural retry (manual `llm_call_with_retry`)
- But **semantic retry is modeled in the graph itself** via conditional edges:
  ```python
  graph.add_conditional_edges("validation", should_retry_formula, {
      "retry": "formula_generation",
      "done": END,
  })
  ```
- This is LangGraph's key advantage — cycles are first-class, not a manual for-loop
- The retry logic is visible in the graph structure, not hidden in imperative code

## Model Swapping
- Change `model` and `base_url` on `ChatOpenAI` — two parameters
- LangChain has dedicated classes per provider (ChatAnthropic, ChatOpenAI, etc.)
- Switching providers means switching the class

## Local Execution
- Works with Ollama via `ChatOpenAI` with custom `base_url`
- JSON mode via `response_format: {"type": "json_object"}` works

## Learning Curve
- Steeper than both PydanticAI and Haystack
- Must understand: StateGraph, TypedDict state, node functions, edges, conditional edges, compile
- The state management pattern (nodes return partial state dicts that get merged) is unintuitive at first
- LangChain ecosystem complexity — lots of abstractions, multiple message types (SystemMessage, HumanMessage, AIMessage)

## Extensibility
- Adding a step = write a node function + add node + add edge
- The graph structure makes it easy to see the pipeline flow
- Conditional edges make branching/retry logic declarative

## Error Clarity
- LangChain error messages can be verbose and layered (wrapping OpenAI errors in LangChain errors)
- Graph execution errors reference node names — helpful for debugging
- State type errors are caught at graph compile time

## Schema Swapability
- Same as Haystack — Pydantic models are easy to change, prompt text needs manual updates
- State TypedDict also needs updating if output types change

## Observations
- LangGraph is the only framework that models the retry loop in the graph itself (conditional edges with cycles)
- This makes the pipeline flow explicit and inspectable — you can visualize the graph
- But the overhead is significant: TypedDict state, node functions, edge wiring, compile step
- For a linear pipeline, it's overkill. For complex flows with branching and retry, the graph model pays off
- The LangChain ecosystem is heavy — many layers of abstraction for what is ultimately an HTTP call to Ollama
