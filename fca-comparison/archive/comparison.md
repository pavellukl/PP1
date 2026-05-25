# Framework Comparison — Summary

## Overview

Four frameworks evaluated for a multi-step LLM pipeline that formalizes safety-critical avionics requirements (DO-178C context). Pipeline: ambiguity review → entity extraction → formula generation → validation with retry.

Tested with qwen2.5:14b running locally via Ollama. 48 total runs (4 frameworks x 4 requirements x 3 runs each).

## Comparison Table

| Criteria                               | PydanticAI                                                                             | Haystack                                                                 | LangGraph                                                  | CrewAI                                    |
| -------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------- | ----------------------------------------- |
| **Lines of code** (framework-specific) | ~140                                                                                   | ~205                                                                     | ~210                                                       | ~220                                      |
| **Structured output**                  | Built-in (auto-schema, auto-retry)                                                     | Manual (~25 lines)                                                       | Manual (~25 lines)                                         | Guardrails (partial)                      |
| **Intermediate inspection**            | `result.output` between steps                                                          | Dict outputs                                                             | Shared TypedDict state                                     | `result.raw` per Crew                     |
| **Retry — structural**                 | Built-in (`retries=3`)                                                                 | Manual                                                                   | Manual                                                     | Guardrails                                |
| **Retry — semantic**                   | Manual loop                                                                            | Manual loop                                                              | Conditional edge in graph                                  | Manual loop                               |
| **Model swapping**                     | Change provider + model name                                                           | Change generator class + params                                          | Change LLM class + params                                  | Change `LLM()` prefix                     |
| **Local execution**                    | Works (needs JSON mode workaround)                                                     | Works                                                                    | Works                                                      | Works                                     |
| **Learning curve**                     | Low (Pydantic + Agent)                                                                 | Medium (Components, Pipeline, Secret)                                    | Medium (StateGraph, TypedDict, edges, LangChain ecosystem) | Low (Agent/Task/Crew is intuitive)        |
| **Extensibility**                      | Add Agent + 3 lines                                                                    | Add @component class + wiring                                            | Add node function + edges                                  | Add Agent + Task + Crew                   |
| **Error clarity**                      | Good (Pydantic errors + path)                                                          | OK (manual parsing errors)                                               | Verbose (LangChain stack traces)                           | OK (guardrail errors)                     |
| **Schema swapability**                 | Best — change the Pydantic model and the LLM automatically receives the updated schema | Change the Pydantic model + update the prompt text describing the schema | Same as Haystack                                           | Same as Haystack                          |
| **Success rate**                       | 92%                                                                                    | 92%                                                                      | 75%                                                        | 92%                                       |
| **Avg time per run**                   | 65.0s                                                                                  | 65.6s                                                                    | 50.5s                                                      | 25.5s                                     |
| **Formula depth**                      | Shallow (no quantifiers)                                                               | Medium                                                                   | Medium                                                     | Deepest (Exists/ForAll)                   |
| **Install footprint**                  | Medium                                                                                 | Light                                                                    | Medium (LangChain ecosystem)                               | Heavy (chromadb, onnxruntime, kubernetes) |

## Framework Strengths

**PydanticAI** — Least code, best schema propagation. Define a Pydantic model, get structured output with auto-retry. Closest to "just writing Python." Best choice if code simplicity and maintainability are the priority.

**Haystack** — Pipeline-as-graph with YAML serialization. Useful if you need to export/share pipeline configs. DAG-only (no cycles), so retry loops must be manual. Pre-built components for RAG workflows (not relevant here).

**LangGraph** — Only framework that models retry loops as graph edges (conditional edges with cycles). The flow is declarative and visualizable. But lowest reliability in testing and heaviest abstraction overhead (TypedDict state, LangChain ecosystem).

**CrewAI** — Fastest, deepest formulas, designed for multi-agent collaboration. But heaviest install, most boilerplate, and the role/goal/backstory pattern is designed for agent conversation, not structured data pipelines. We could not fully explain its speed/quality advantage.

## Future: Tool Use

The current pipeline doesn't use LLM tool calling — each step is a single LLM call that returns structured JSON. However, as the pipeline matures, tool use could become relevant:

- **Syntax checker tool** — the formula agent calls a tool to verify its formula against the formalism grammar before returning it, catching structural errors the LLM can't self-detect
- **Glossary / domain lookup** — the entity extractor queries a domain glossary to resolve avionics-specific terms and ensure consistent entity naming across requirements
- **Requirements database** — the ambiguity reviewer checks for conflicts or overlaps with already-formalized requirements

If tools are added, "hallucinated tool calls" becomes a real concern — the LLM calling tools with wrong arguments or calling nonexistent tools. How the frameworks handle this:

| Framework | Tool support | Hallucination handling |
|-----------|-------------|----------------------|
| PydanticAI | Built-in with typed tool definitions, auto-validates arguments against schemas | Best — same Pydantic validation as structured output |
| LangGraph | LangChain tool ecosystem, extensive tooling | Good — large ecosystem, well-documented |
| CrewAI | Agent tools with guardrails | OK — guardrails can validate, but less mature |
| Haystack | Component-based, tools are custom components | Manual — you build the validation |

## Decision Questions for the Team

1. **How important is running fully offline?** All four work with Ollama, but the 14B local model is the bottleneck — complex requirements fail across all frameworks. A cloud model would improve quality dramatically but adds a dependency.

2. **Do you need the pipeline to be serializable/exportable?** Haystack supports YAML serialization. The others are code-only.

3. **How complex will the retry/branching logic get?** If the pipeline stays mostly linear with simple retry, PydanticAI's manual loop is fine. If it grows into complex branching (multiple validation paths, conditional steps), LangGraph's graph model becomes more valuable.

4. **How important is formula depth vs reliability?** CrewAI produces deeper formulas but the advantage is unexplained and may not persist with a stronger model. PydanticAI is more predictable.

5. **Will you need multi-agent collaboration?** If future pipeline stages involve agents discussing, delegating, or debating (e.g., ambiguity resolution with multiple reviewers), CrewAI is designed for this. The others treat each step as an independent call.

## Conclusion

### Ranking

1. **CrewAI** — The surprise of this comparison. Initially dismissed as not suitable for serious applications, it consistently outperformed across 48 test runs — fastest execution, deepest formulas, and matched top reliability. We investigated and could not fully explain the advantage. Despite the heavy install footprint, it earned first place on results alone. Needs further investigation to understand whether the advantage holds with stronger models or in production.

2. **PydanticAI** — Least code, best schema propagation, gets out of the way. The safe, predictable choice for a pipeline where the formalism is still evolving. If we prioritize maintainability and simplicity over raw output quality, this is the pick.

3. **LangGraph** — Drop unless the pipeline grows into complex branching with multiple conditional paths. Its graph model is the most expressive for control flow, but it had the lowest reliability (75%) and the LangChain ecosystem adds abstraction overhead that didn't pay off in our tests.

4. **Haystack** — Drop unless YAML serialization or pre-built RAG components become a requirement. Its DAG-only pipeline couldn't model our retry loops, so we ended up calling components manually — at that point it's just an LLM API wrapper with extra boilerplate.
