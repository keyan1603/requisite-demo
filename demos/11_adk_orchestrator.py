"""
11 - ADK orchestrator backend: delegating multi-agent coordination to
Google's Agent Development Kit (google-adk) instead of the native or
langgraph backends.

Covers `sequential` (a custom BaseAgent running each step's LlmAgent in
order) and `supervisor` (a custom BaseAgent reusing the exact same
decision protocol the native/langgraph supervisor already use). Every
actual model call still goes through each agent's own configured
provider -- ADK handles coordination only, never the LLM call itself.

Requires: pip install google-adk (a materially heavier install than the
other backends -- see the main repo's docs/adr/0039-adk-orchestrator-backend.md).

Run with:
    python demos/11_adk_orchestrator.py
"""

from _shared import gemini_api_key, make_agent
from requisite import Workflow


def sequential_on_adk() -> None:
    print("=== sequential (adk) ===")
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().sequential().use_adk()
    workflow.add(researcher).add(writer)
    result = workflow.run("Research AI agent frameworks and summarize the landscape.")
    print(result.content)
    print(f"(orchestrator={result.orchestrator}, strategy={result.strategy})")


def supervisor_on_adk() -> None:
    print("\n=== supervisor (adk) ===")
    coordinator = make_agent(
        "Coordinator", "You coordinate a small team, delegating one subtask at a time."
    )
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().supervisor().use_adk()
    workflow.add(coordinator).add(researcher).add(writer)
    result = workflow.run(
        "Research what Google's Agent Development Kit (ADK) for building AI "
        "agents is and write a short explainer.",
        max_rounds=6,
    )
    print(result.content)
    print(f"(delegated to: {[s.agent_name for s in result.steps]})")


def main() -> None:
    if not gemini_api_key():
        print("Skipping 11_adk_orchestrator.py: no GEMINI_API_KEY configured.")
        return
    sequential_on_adk()
    supervisor_on_adk()


if __name__ == "__main__":
    main()
