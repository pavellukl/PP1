# Experiment Notes

Knowledge base for Section 3 and beyond. Records all experiments, findings, and observations.

## Semester 1: Framework Comparison (qwen2.5:14b)

Model: qwen2.5:14b via Ollama (local, AMD RX 6800 XT)
Pipeline: ambiguity review → entity extraction → formula generation → validation with retry
Requirements tested: 2, 3, 6, 8 (48 runs: 4 frameworks × 4 requirements × 3 runs)

### Quantitative Results

| Metric | PydanticAI | Haystack | LangGraph | CrewAI |
|--------|-----------|----------|-----------|--------|
| Success rate | 92% (11/12) | 92% (11/12) | 75% (9/12) | 92% (11/12) |
| Avg time | 65.0s | 65.6s | 50.5s | **25.5s** |
| Avg attempts | 1.4 | 1.8 | 1.8 | **1.3** |
| Structural failures | 3 | 9 | 7 | 4 |
| Uses Exists/ForAll | 0/12 | 5/12 | 3/12 | **6/12** |
| Avg nesting depth | 4 | 5 | 4 | **6** |
| Lines of code | ~140 | ~205 | ~210 | ~220 |

### Key Findings
- **On simple requirements, all frameworks produce equivalent results.** Req 6 succeeds 12/12.
- **CrewAI is consistently faster and produces deeper formulas.** Investigated via HTTP interception: CrewAI sends `response_format: None` while others send `json_object`. Could not fully reproduce advantage by changing this setting alone.
- **LangGraph worst reliability (75%).** No clear technical reason.
- **Entity extraction quality determines formula quality.** Mistyped entities can't be recovered.
- **Semantic validation too shallow.** Both PydanticAI and CrewAI passed validation with clearly wrong formulas.
- **Model capability is the bottleneck, not the framework.** Complex requirements fail across all frameworks.
- **PydanticAI: least code, best schema propagation.** Change the Pydantic model and the LLM automatically gets the updated schema.
- **Haystack: DAG-only, can't model retry loops.** Ended up calling components manually.

### Framework Ranking
1. CrewAI — fastest, deepest formulas, unexplained advantage, heavy dependencies
2. PydanticAI — least code, best schema propagation, predictable
3. LangGraph — elegant graph model, worst reliability
4. Haystack — DAG-only, not suited for retry loops

---

## Updated Formalism (May 2026)

Colleague delivered April 2026 formalism with significant changes from prototype version:
- 9 entity types (added EventTrigger, Set)
- 10 operations (added insert, remove, clear; renamed store→write; added EvtPred mechanism)
- 9 temporal constructs (Always, Eventually, Initial, Causes, CausesWithin, Sequence, Precedes, Excludes, Immediately)
- ~20 expression types (arithmetic, intervals, trace functions)
- Requirement structure: (flavour ∈ {Discrete, Continuous}, entities, constraint)
- EventTrigger/Event split: EventTrigger entities fire events; Event entities have target+type modifiers linking back

## Pipeline Architecture (Updated)

4-step pipeline (ambiguity → flavour → entities → constraint), with ambiguity notes passed forward to all subsequent steps.

## Model Selection

- **phi4:latest (14B)** — Microsoft's base model, good at structured reasoning, no thinking overhead
- **phi4-reasoning:plus (14B)** — reasoning variant, wraps output in `<think>` tags that conflict with structured output parsing. Too slow and incompatible with JSON mode for pipeline use.
- **Gemini 2.5 Flash** — Google cloud model, free tier (5 RPM, 20 RPD). Strongest available model.
- qwen2.5:14b and qwen3:14b — used in semester 1 experiments, replaced by phi4.

### Findings: phi4-reasoning:plus issues
- Wraps ALL output in `<think>...</think>` tags, even when told not to
- When told "do not use think tags", produces prose instead of JSON
- When allowed to think, actual JSON comes after `</think>` but PydanticAI can't parse it
- Each call takes 30-120s vs 3-20s for base phi4
- **Conclusion:** reasoning models are incompatible with framework-managed structured output. Would need custom parsing layer.

## Experiment: Python vs JSON Output Format

**Hypothesis:** Based on Danso et al. (2026), expressing formal output as Python code instead of JSON should improve accuracy because (a) LLMs have more Python training data, and (b) Python constructors enforce structural validity.

### Setup
- Same formalism context (FORMALISM_CONTEXT) in both prompts
- JSON prompt: describes schema with discriminators (construct_type, pred_type, expr_type)
- Python prompt: includes full formalism_api.py source code
- Both get 3 examples (CausesWithin, Eventually, Sequence)
- Retry on validation failure (max 2 retries)
- Entity extraction and flavour extraction shared (same for both)

### Preliminary Results (3 requirements, 1 run each, phi4:latest)

| Req | JSON | Python | Notes |
|-----|------|--------|-------|
| 06 (CausesWithin) | SUCCESS, 1 attempt, 6.1s | SUCCESS, 1 attempt, 9.6s | Both correct |
| 02 (Store to memory) | SUCCESS, 1 attempt, 6.5s (Always) | SUCCESS, 2 attempts, 29.4s (Eventually) | Different constraint types — semantic divergence |
| 03 (Sequence) | SUCCESS, 1 attempt, 7.8s | SUCCESS, 1 attempt, 21.2s | Both correct |

### Observations
- JSON is ~2-3x faster (shorter output)
- Python needed retry on 1/3 requirements
- Semantic divergence on Req 02: JSON chose Always, Python chose Eventually — both structurally valid but different meanings
- At phi4 14B capability level, neither format has a decisive structural advantage
- Entity extraction is the most unreliable step — modifier format varies (dict vs list vs string)

### Comprehensive Results (phi4:latest, 10 requirements × 3 runs)

**JSON: 90% success (27/30) vs Python: 70% success (21/30)**

| Req | JSON | Python | Notes |
|-----|------|--------|-------|
| 01 (arithmetic recurrence) | 1/3 | 2/3 | Both struggle — complex initial + always |
| 02 (store to memory) | 3/3 | 3/3 | Both easy |
| 03 (ordered sequence) | 3/3 | 3/3 | Both easy |
| 04 (configure registers) | 3/3 | 2/3 | Python fails on entity references |
| 05 (exception handling) | 3/3 | 1/3 | Python invents functions/wrong args |
| 06 (scheduling deadline) | 3/3 | 3/3 | Both easy |
| 07 (transmit after init) | 3/3 | 3/3 | Both easy |
| 08 (state on receive) | 3/3 | 3/3 | Both easy |
| 09 (multi-condition toggle) | 3/3 | 0/3 | Python completely fails on hardest req |
| 10 (repeated failures) | 2/3 | 1/3 | Both struggle |

JSON avg time: 17.3s, avg attempts: 1.3
Python avg time: 20.2s, avg attempts: 1.5

**Key finding: JSON outperforms Python on phi4 14B.** This contradicts Danso et al.'s finding (Python ~3x better), but Danso tested on frontier models (GPT-4o, Claude, Gemini). At the 14B local model level, the model handles JSON discriminators better than a custom Python API it hasn't seen in training. Python errors include: inventing function names (NameError), wrong argument counts (TypeError), and confusing API levels (using expressions as modifier values).

## Experiment: Gemini 2.5 Flash

Tested Req 09 (hardest — multi-condition timed toggle) and Req 06 (simplest — CausesWithin deadline).

### Results

| Req | JSON | Python | Notes |
|-----|------|--------|-------|
| 09 (hardest) | SUCCESS, 1 attempt, 34.1s (CausesWithin) | SUCCESS, 2 attempts, 39.5s (CausesWithin) | Both produce CausesWithin with AllOf condition! Gemini handles the complex multi-condition requirement. |
| 06 (simplest) | SUCCESS, 1 attempt, 3.9s (CausesWithin) | SUCCESS, 1 attempt, 4.2s (CausesWithin) | Both correct, very fast. |

### Key Observations
- **Gemini 2.5 Flash succeeds on Req 09** — the hardest requirement that phi4 struggles with. This confirms model capability is a major factor.
- Both JSON and Python succeed on both requirements — no format advantage visible here.
- Gemini is faster than phi4 on simple requirements (3.9s vs 6.1s) and produces more complex correct output on hard requirements.
- Entity extraction with Gemini produced 12 entities for Req 09 including appropriate peak detection triggers — more thorough than phi4.
- Event modifier values still come back as None in some cases — entity extraction prompt needs work.
- Python needed 1 retry on Req 09 (same pattern as phi4), JSON didn't.
- **Gemini produced correct CausesWithin structure for Req 09 on first try (JSON)** — this is the requirement that uses EvtOccCount, MkInterval, LastOcc, and multiple conditions. Very promising for cloud model capability.
- Used ~8 of 20 daily requests (12 remaining).

## Experiment: CoT Prompt (REQ2LTL-style) with Gemini

Tested the refined CoT prompt (based on REQ2LTL's BUILDING-ONIONL algorithm) on Req 09 with Gemini 2.5 Flash.

### CoT Prompt Structure (based on REQ2LTL paper)
- Stage 1 (Macro): Determine time model → Identify top-level temporal scope
- Stage 2 (Recursive): Extract entities → Decompose condition → Decompose effect → Normalize atomic propositions
- Critical format rules at end showing discriminator pattern with examples

### Results on Req 09 (hardest requirement)

**CoT produced significantly better output than pipeline on the same requirement:**

| Aspect | Pipeline (Gemini) | CoT (Gemini) | Ground Truth |
|--------|-------------------|--------------|--------------|
| Peak diff > 655 | Separate peak entities ⚠️ | Separate peak entities ⚠️ (non-deterministic — one run used temporal lookups correctly) | Val(d, Start(LastOcc(ev_max_peak))) - Val(d, Start(LastOcc(ev_min_peak))) |
| 2nd write after min peak | EvtOccCount, open interval ✅ | EvtOccCount, open interval ✅ | Same ✅ |
| Signed value > last | ValBefore ⚠️ | ValBefore(d, Now) ⚠️ | Val(d, Start(LastOcc(ev_written_d, Now, 2))) — lookup at previous write event time |
| Toggle effect | Happening only, missing toggle ❌ | write(valid_range, 1 - ValBefore) ✅ | Val(valid_range) = ¬ValBefore(valid_range) |

**Key findings:**
- CoT got the toggle effect correct (1 - ValBefore) where the pipeline missed it entirely.
- Both approaches get the EvtOccCount with open interval correct.
- Both misinterpret "signed value of last d" — they use ValBefore (value just before now) instead of Val at the previous write event time. This is the temporal scoping error Danso identifies.
- Peak diff handling is non-deterministic — one CoT run used correct temporal lookups (Val at LastOcc time), another used simplified separate entities. Same prompt, different outputs.
- The first CoT attempt used wrapper-key JSON format; had to add explicit discriminator format rules to fix. This is a strong argument for Python output which would avoid format issues entirely.

**JSON format note:** First attempt with CoT used wrapper-key format ({"CausesWithin": {...}}) instead of discriminator format. Fixed by adding explicit format rules to the prompt. This is exactly the kind of error the Python output would avoid — the function calls enforce the correct structure.

### Implications
- CoT with hierarchical decomposition improves semantic accuracy, not just structural validity
- The REQ2LTL decomposition algorithm translates well to our formalism
- Format instructions must be very explicit — reasoning quality doesn't guarantee correct serialization
- This is a strong argument for combining CoT reasoning with Python output: the model reasons well with CoT, and Python ensures the output structure is correct

## Experiment: CoT vs Pipeline (Gemini 3.5 Flash)

Tested 4 requirements (06, 02, 03, 09) with Gemini 3.5 Flash. Req 09 hit daily rate limit.

### Results

| Req | Pipeline | CoT | Notes |
|-----|----------|-----|-------|
| 06 (CausesWithin) | SUCCESS, 168.9s | SUCCESS, 16.0s | Same constraint. CoT 10x faster |
| 02 (Store to memory) | SUCCESS, 42.8s (TraceAllOf, 4 entities) | SUCCESS, 16.8s (Always, 2 entities) | Different constructs |
| 03 (Sequence) | SUCCESS, 33.9s (5 entities) | SUCCESS, 30.9s (4 entities) | Same construct. Pipeline more entities |
| 09 (Multi-condition) | FAILED (rate limit) | FAILED (rate limit) | Daily quota exhausted |

Pipeline avg time: 81.9s, CoT avg time: 21.2s

### Ground Truth Comparison

**Req 06 (CausesWithin deadline):** Both semantically correct. Only entity naming differs. ✅/✅

**Req 02 (Store to memory):**
- Ground truth: Eventually(Val = Val) ∧ Always(reset ⇒ value preserved)
- Pipeline: TraceAllOf(Always(Addr check) + Eventually(Happening + Val check)) — has two clauses but Always checks wrong thing (address equality, not reset persistence) ⚠️
- CoT: Always(Val = Val) — too strong, says always equal instead of eventually equal. Misses reset clause entirely. ❌
- Pipeline is closer to ground truth but neither is fully correct.

**Req 03 (Sequence — read then store):**
- Ground truth: Sequence(Happening(read), Happening(write) ∧ Val(DTSCON) = Val(cal_const, Start(LastOcc(read))))
- Pipeline: Sequence correct, but value check uses Val(cal_const, Now) instead of at read time ⚠️
- CoT: Sequence correct, AND value check uses Val(calib, Start(LastOcc(ev_read, Now, 1))) ✅ — matches ground truth's temporal lookup exactly
- **CoT is semantically more correct** — captures that the stored value should be what was read, not the current value.

### Key Findings
- **CoT is ~4x faster** — 1 API call vs 3, plus no rate limit waits between steps
- **Both succeed at the same structural rate** on completed tests (3/3)
- **CoT has better temporal precision on Req 03** — used LastOcc to reference value at read time
- **Pipeline has better structure on Req 02** — attempted both Eventually and Always clauses (CoT oversimplified)
- **Neither fully captures Req 02** — reset persistence clause missed by both
- **CoT produces fewer entities** — reasons about the whole requirement at once, may simplify more
- **CoT advantage for rate-limited APIs** — uses 1/3 the requests, critical for free tier usage

### Rerun with matched decomposition prompt (fresh quota)

All 4 requirements, both approaches now use the same decomposition instructions.

| Req | Pipeline | CoT |
|-----|----------|-----|
| 06 | SUCCESS (30.1s, 4 entities) | SUCCESS (28.2s, 4 entities) |
| 02 | SUCCESS (43.6s, Eventually, 5 entities) | SUCCESS (24.4s, Always, 2 entities) |
| 03 | SUCCESS (51.5s, Sequence, 6 entities) | SUCCESS (34.6s, Sequence, 4 entities) |
| 09 | SUCCESS (47.4s, CausesWithin, 12 entities) | SUCCESS (34.3s, CausesWithin, 7 entities) |

Both: 4/4 (100%). Pipeline avg: 43.1s, CoT avg: 30.4s.

- Pipeline consistently finds more entities (separate extraction step is more thorough)
- CoT is ~30% faster (fewer API calls)
- Req 02 semantic divergence persists (Eventually vs Always)
- **Both succeed on Req 09** — Gemini 3.5 Flash handles the hardest requirement regardless of architecture

### Ground Truth Comparison (updated prompts, Gemini 3.5 Flash)

| Req | Pipeline | CoT | Winner |
|-----|----------|-----|--------|
| 06 | ✅ Correct | ✅ Correct | Tie |
| 02 | ✅ Correct construct (Eventually) | ❌ Wrong construct (Always vs Eventually) | Pipeline |
| 03 | ⚠️ Val(cal_const, Now) — stale value | ✅ Val(cal_const, Start(LastOcc(read))) — correct temporal lookup | CoT |
| 09 | ⚠️ Raw LastOcc as interval (wrong), toggle event only (no value check) | ✅ MkInterval(Start(LastOcc), Now) correct, Val != ValBefore for toggle | CoT |

**CoT wins on temporal precision** (Req 03, 09) — reasons about when events happened and uses LastOcc/Start correctly.
**Pipeline wins on structural correctness** (Req 02) — CoT oversimplifies (Always instead of Eventually).
Neither approach captures the full ground truth on Req 09 (both use separate peak entities instead of Val(d, at_peak_time)).

### Combined with Gemini 2.5 Flash CoT results on Req 09
From earlier testing, CoT with Gemini 2.5 Flash on Req 09:
- Got the toggle effect correct (pipeline missed it)
- Used EvtOccCount with open intervals correctly
- Non-deterministic on peak diff handling (sometimes temporal lookups, sometimes separate entities)
- Required explicit discriminator format rules in prompt

## Experiment: CoT vs Pipeline (phi4:latest)

4 requirements × 1 run each to match the Gemini comparison.

### Results

| Req | Pipeline | CoT |
|-----|----------|-----|
| 02 (Store to memory) | SUCCESS (34.5s) | FAIL — schema validation |
| 03 (Sequence) | SUCCESS (49.6s) | FAIL — schema validation |
| 06 (CausesWithin) | SUCCESS (22.0s) | SUCCESS (23.2s) |
| 09 (Multi-condition) | FAIL | FAIL |

Pipeline: 75% (3/4), CoT: 25% (1/4) — with simple formula prompt

### Rerun with matched decomposition prompt
After updating FORMULA_SYSTEM to use the same decomposition steps as COT_SYSTEM (identify scope → decompose condition → decompose effect → normalize), ensuring only the architecture differs:

Pipeline: 75% (3/4), CoT: 50% (2/4)
Pipeline avg: 28.4s, CoT avg: 21.8s

| Req | Pipeline | CoT (old prompt) | CoT (matched prompt) |
|-----|----------|------------------|---------------------|
| 02 | SUCCESS | FAIL | SUCCESS |
| 03 | SUCCESS | FAIL | FAIL |
| 06 | SUCCESS | SUCCESS | SUCCESS |
| 09 | FAIL | FAIL | FAIL |

### Key Findings
- **Decomposition prompt doubled CoT success rate** (25% → 50%) on phi4, confirming REQ2LTL's finding that structured reasoning instructions matter.
- **Pipeline still wins on phi4** (75% vs 50%) — weaker model benefits from simpler per-call schemas.
- **Gemini 3.5 Flash: Pipeline and CoT tie 3/3.** Stronger model handles the combined task fine.
- **CoT is faster when it works** — 21.8s vs 28.4s avg (fewer API calls).
- **Conclusion: the optimal architecture depends on model capability.** Weaker models benefit from pipeline decomposition. Stronger models benefit from CoT (full context, better temporal reasoning). The decomposition prompt helps both approaches.

## Open Questions for Section 3

- Single CoT call vs multi-call pipeline — not yet tested
- Does the Python advantage appear with stronger models? (Danso tested on GPT-4o, Claude, Gemini)
- How to measure semantic correctness without NuSMV (we have ground truth from formalism doc)
- Framework comparison with new formalism — does CrewAI's advantage persist?
