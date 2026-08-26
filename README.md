# PP1 — LLM Pipeline for Requirement Formalization

## Layout

- `fca-comparison/` — the code: same pipeline in PydanticAI, Haystack, LangGraph, CrewAI (`frameworks/`), shared `schemas.py` / `prompts.py`, `experiments/` scripts, `results/` JSON outputs, `EXPERIMENT_NOTES.md` with findings
- `req/` — test requirements `01.md`–`10.md`
- `report/` — LaTeX semester report
- `presentation/` — LaTeX beamer slides

## Running the pipelines

Requires a running [Ollama](https://ollama.com) server with the model pulled (default: `phi4-reasoning:plus`).

```bash
cd fca-comparison
source .venv/bin/activate   # deps already installed; otherwise: pip install pydantic-ai haystack-ai langgraph crewai python-dotenv

# optional overrides (defaults: WSL host Ollama at 172.31.128.1, phi4-reasoning:plus)
export OLLAMA_BASE_URL=http://localhost:11434/v1
export MODEL_NAME=qwen2.5:14b

python frameworks/pydantic-ai/main.py 2   # run pipeline on req/02.md
python frameworks/langgraph/main.py       # no arg = built-in test requirement
```

Same invocation for `haystack`, `langgraph`, `crewai`. Each prints the intermediate output of every step (ambiguity review → entity extraction → formula generation → validation with retry).

## Experiments

Run from `fca-comparison/` with the venv active; results land in `results/`, findings are written up in `EXPERIMENT_NOTES.md`. Gemini scripts need `GEMINI_API_KEY` in `fca-comparison/.env`; the rest use Ollama.

```bash
python experiments/framework_comparison_runner.py    # batch: 4 frameworks × 4 reqs × 3 runs
python experiments/python_vs_json_preliminary.py     # Python-code vs JSON output, first probe (phi4)
python experiments/python_vs_json_phi4.py            # same comparison, all reqs, multiple runs + retry
python experiments/cot_vs_pipeline_phi4.py           # single CoT prompt vs multi-step pipeline (phi4)
python experiments/cot_vs_pipeline_gemini.py         # same, on Gemini Flash (mind the free-tier rate limit)
python experiments/gemini_cloud_test.py              # hardest req (09) on Gemini, JSON + Python formats
```
