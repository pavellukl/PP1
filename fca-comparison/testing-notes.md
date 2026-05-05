# Testing Notes

## Test Setup
- Model: qwen2.5:14b via Ollama (local, AMD RX 6800 XT)
- Requirements tested: 2, 3, 6, 8 (48 runs total: 4 frameworks x 4 requirements x 3 runs)
- Prompts standardized across all frameworks via shared `prompts.py`

## Quantitative Results

| Metric | PydanticAI | Haystack | LangGraph | CrewAI |
|--------|-----------|----------|-----------|--------|
| Success rate | 92% (11/12) | 92% (11/12) | 75% (9/12) | 92% (11/12) |
| Avg time per run | 65.0s | 65.6s | 50.5s | **25.5s** |
| Avg attempts | 1.4 | 1.8 | 1.8 | **1.3** |
| Structural failures | 3 | 9 | 7 | 4 |
| Uses Exists/ForAll | 0/12 | 5/12 | 3/12 | **6/12** |
| Avg nesting depth | 4 | 5 | 4 | **6** |

## Key Findings

**On simple requirements, all frameworks produce equivalent results.** Req 6 succeeds 12/12 across all frameworks with structurally identical formulas. On harder requirements (3, 8), differences emerge in both reliability and formula depth, with CrewAI trending better and LangGraph trending worse.

**CrewAI is consistently faster and produces deeper formulas.** We investigated why by intercepting the actual HTTP requests to Ollama. We found that CrewAI sends `response_format: None` while the other three send `response_format: json_object`, and CrewAI skips the extra JSON schema injection message that PydanticAI adds. However, removing `response_format: json_object` from PydanticAI did not reliably reproduce CrewAI's advantage — it produced more sophisticated formulas occasionally but also more failures. The exact cause of CrewAI's edge remains unconfirmed; it may be a combination of factors including message formatting differences at the litellm layer.

**LangGraph had the worst reliability (75%).** Particularly bad on req 8 (1/3 success). No clear technical reason — likely LLM randomness amplified by the framework's message formatting.

**Entity extraction quality determines formula quality.** When entities are mistyped (STATE vs EVENT, wrong STORAGE assignments), the formula step can't recover even with retries.

**Our semantic validation is too shallow.** Both PydanticAI and CrewAI passed validation with clearly wrong formulas (circular logic, `-` as a Cmp operator, invented time variable names). Only entity reference and operation-type compatibility are checked.

## What Doesn't Matter

- **Prompt text:** After standardizing prompts, all frameworks use identical wording from `prompts.py`. CrewAI still slightly outperforms, suggesting the difference is in how the framework delivers the prompt to the LLM, not the prompt content itself.
- **Framework architecture:** On simple requirements (req 6), all frameworks produce structurally identical formulas. On harder requirements, output quality varies — but we could not isolate whether this is framework behavior or LLM randomness amplified by small differences in how each framework formats the API call.
- **Structured output mechanism:** PydanticAI's auto-retry on schema validation is convenient but doesn't improve formula quality — it just saves ~25 lines of manual retry code.

## What Matters

- **Model capability:** The 14B local model is the bottleneck, not the framework. Complex requirements (9, parts of 3) fail across all frameworks.
- **`response_format` setting:** CrewAI doesn't use JSON-constrained decoding, and the other three do. This is a known difference but was not confirmed as the cause of CrewAI's advantage.
- **Code simplicity:** PydanticAI requires the least code for structured output. The others require manual JSON parsing + validation + retry (~25 identical lines each).
