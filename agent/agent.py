"""
A tiny LangGraph agent whose traces are exported over OTLP — no autolog, no third-party
instrumentor. LangChain's own telemetry SDK (langsmith[otel]) emits OpenTelemetry
GenAI-convention spans (gen_ai.*) straight to an OTLP endpoint; here that's the local OTel
collector, which forwards them to MLflow.

Tracing is entirely environment config (defaults set below). The agent code is just
"build the graph, run prompts." Point OTEL_EXPORTER_OTLP_ENDPOINT at anything that speaks
OTLP to switch backends — nothing in this file changes.

Default model is local Ollama (no key, no cost).
"""
import os

# Enable LangChain/LangGraph -> OTLP tracing. MUST be set before importing langchain/langsmith.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")  # OTLP only: no LangSmith cloud, no API key
# langsmith[otel] passes OTEL_EXPORTER_OTLP_ENDPOINT verbatim to the OTLP exporter, which does
# NOT auto-append the signal path — so include /v1/traces explicitly, or the collector 404s.
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
os.environ.setdefault("OTEL_SERVICE_NAME", "langgraph-agent")

import sys

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It's 21C and sunny in {city}."


llm = ChatOllama(
    model=os.getenv("MODEL", "llama3.1"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)
agent = create_react_agent(llm, tools=[get_weather])


def ask(prompt: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    prompts = sys.argv[1:]
    if prompts:
        # one-shot: each CLI arg is a separate prompt (and a separate trace)
        for p in prompts:
            print(f"\n> {p}\n{ask(p)}")
    else:
        # interactive: submit a few prompts; blank line or Ctrl-D to stop
        print('Ask the agent something (blank line / Ctrl-D to quit). e.g. "weather in Toronto?"')
        try:
            while True:
                p = input("\n> ").strip()
                if not p:
                    break
                print(ask(p))
        except EOFError:
            pass
    print("\nTraces -> http://localhost:5001  (Experiments > Default > Traces)")
