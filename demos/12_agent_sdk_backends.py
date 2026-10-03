"""
12 - Agent-SDK orchestrator backends: the OpenAI Agents SDK, AWS Strands
Agents, and Microsoft Agent Framework.

Each backend runs the same `sequential` and `supervisor` strategies as
the native, langgraph and adk backends -- one line of config, same
`.add()`/`.run()` API -- but coordinated by that vendor's own agent SDK
(`sequential` is a real Strands Graph, a real Microsoft WorkflowBuilder
workflow, or chained OpenAI `Runner.run` calls).

They coordinate only. Every actual model call still goes through each
agent's own provider (Gemini here, via the shared rate limiter), so
`use_openai_agents()` never calls OpenAI and `use_strands()` never calls
Bedrock.

Each sequential workflow is run twice on the same agents, since a
workflow and its agents are meant to be reused across calls.

Requires: pip install "requisite-ai[openai_agents,strands,agent_framework]"

Run with:
    python demos/12_agent_sdk_backends.py
"""

from _shared import gemini_api_key, make_agent
from requisite import Workflow

BACKENDS = [
    ("openai_agents", "use_openai_agents"),
    ("strands", "use_strands"),
    ("agent_framework", "use_agent_framework"),
]


def run_on(backend: str, use_backend: str) -> None:
    print(f"\n=== {backend} ===")
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")
    coordinator = make_agent(
        "Coordinator", "You coordinate a small team, delegating one subtask at a time."
    )

    sequential = getattr(Workflow().sequential(), use_backend)()
    sequential.add(researcher).add(writer)
    result = sequential.run("Research AI agent frameworks and summarize the landscape.")
    print(f"sequential: {result.content}")
    print(f"(orchestrator={result.orchestrator}, steps={[s.agent_name for s in result.steps]})")

    # Same workflow, same agents, second synchronous run.
    again = sequential.run("Research quantum computing and write one sentence.")
    print(f"sequential, run again: {again.content}")

    supervisor = getattr(Workflow().supervisor(), use_backend)()
    supervisor.add(coordinator).add(researcher).add(writer)
    result = supervisor.run(
        "Research what an AI agent framework is and write a short explainer.", max_rounds=6
    )
    print(f"supervisor: {result.content}")
    print(f"(orchestrator={result.orchestrator}, delegated to: {[s.agent_name for s in result.steps]})")


def main() -> None:
    if not gemini_api_key():
        print("Skipping 12_agent_sdk_backends.py: no GEMINI_API_KEY configured.")
        return
    for backend, use_backend in BACKENDS:
        run_on(backend, use_backend)


if __name__ == "__main__":
    main()
