# sample-ai-obs-stack

Trace a LangGraph agent over **OpenTelemetry**. LangChain's own telemetry SDK exports OTel
GenAI-convention spans (`gen_ai.*`) over **OTLP** — no autolog, no third-party instrumentor —
to a local OpenTelemetry Collector, which forwards them to MLflow.

```
LangGraph agent  --(langsmith[otel], gen_ai.* over OTLP)-->  OTel Collector :4318  -->  MLflow /v1/traces
```

OTLP is the only vendor-neutral standard in play: the agent emits it, the collector routes it.
Point the collector at Langfuse, Phoenix, Tempo, or Jaeger and the agent never changes.

## Run

```bash
docker compose up -d                 # MLflow (:5001) + OTel collector (:4317/:4318)

ollama pull llama3.1                 # local, free model (or use a hosted one, below)
pip install -r agent/requirements.txt

python agent/agent.py                # interactive — type a few prompts, blank line to quit
# or one-shot:
python agent/agent.py "weather in Toronto?" "weather in Vancouver?"
```

Then open <http://localhost:5001> → **Experiments → Default → Traces**: each prompt is a
trace — the LLM calls, the `get_weather` tool call, the final answer — as a span tree.

Tracing is pure env config (defaults set at the top of `agent/agent.py`): `LANGSMITH_TRACING`,
`LANGSMITH_OTEL_ENABLED`, `LANGSMITH_OTEL_ONLY` (OTLP only — no LangSmith cloud, no API key),
and `OTEL_EXPORTER_OTLP_ENDPOINT`. The agent code itself never mentions tracing.

**Hosted model:** swap `ChatOllama` for `ChatOpenAI` in `agent/agent.py` and set `OPENAI_API_KEY`.

## Try it: does v2 beat v1?

This is where traces earn their keep.

1. Run the agent on a couple of prompts (above). That's **v1** — its traces are now in MLflow.
2. Change something — edit the system prompt, swap the model, add a tool. That's **v2**. Run the same prompts again.
3. The real question: **how do v1 and v2 differ on quality, groundedness, relevance?** You have no ground-truth answer to diff against, so you reach for a *judge*.

You don't have to invent the judges — use the golden ones already out there. MLflow ships
built-in GenAI scorers (`pip install mlflow` to run them). Because you have **no ground
truth**, lean on the *reference-free* ones — they score against the input or the retrieved
context, not a hidden key:

```python
import mlflow
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety

traces = mlflow.search_traces(experiment_names=["Default"])   # v1 + v2 runs
mlflow.genai.evaluate(
    data=traces,
    scorers=[RelevanceToQuery(), RetrievalGroundedness(), Safety()],
)
```

(`Correctness` also ships, but it needs an *expected* answer — exactly what you don't have.)
Other well-known scorer sets to point at the same traces: **RAGAS** (faithfulness /
answer-relevancy / context-precision), **OpenAI Evals**, **DeepEval**.

The catch worth thinking hardest about: the judge is itself a model with no ground truth.
Which scorer do you trust, and how would you know it's right? That's the open problem —
calibrating the judge — and it's where the real work is. This repo gets you to real traces;
the scoring layer is yours to build on top.

## Notes

- LangChain/LangGraph emit OTel natively via `langsmith[otel]` — the emitter is first-party,
  the spans are the OTel `gen_ai.*` standard, and `LANGSMITH_OTEL_ONLY=true` keeps everything
  local (pure OTLP to your collector, no cloud, no API key).
- If MLflow's rendering of ingested `gen_ai.*` spans looks thin, retarget the collector's
  exporter at Phoenix — one line in `otel-collector.yaml`, no change to the agent.

MIT — see [LICENSE](LICENSE).
