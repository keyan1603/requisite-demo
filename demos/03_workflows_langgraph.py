"""
03 - The same workflow API, switched to the langgraph execution backend.

The point: `.use_langgraph()` is a one-line configuration change --
`.add()`/`.run()` never change. supervisor/hierarchical build a real
`add_conditional_edges` graph on langgraph instead of a Python loop; see
docs/adr/0016, 0028, 0029 in the requisite-ai repo for the design.

Run with:
    python demos/03_workflows_langgraph.py
"""

from _shared import make_agent
from requisite import Workflow


def sequential_on_langgraph() -> None:
    print("=== sequential (langgraph) ===")
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().use_langgraph()
    workflow.add(researcher).add(writer)
    result = workflow.run("Research AI agent frameworks and summarize the landscape.")
    print(result.content)
    print(f"(orchestrator={result.orchestrator}, strategy={result.strategy})")


def supervisor_on_langgraph() -> None:
    print("\n=== supervisor (langgraph -- a real conditional graph, not a Python loop) ===")
    coordinator = make_agent(
        "Coordinator", "You coordinate a small team, delegating one subtask at a time."
    )
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().supervisor().use_langgraph()
    workflow.add(coordinator).add(researcher).add(writer)
    result = workflow.run("Research what LangGraph is and write a short summary.", max_rounds=6)
    print(result.content)
    print(f"(delegated to: {[s.agent_name for s in result.steps]})")


def reflection_on_langgraph() -> None:
    print("\n=== reflection (langgraph -- a real 3-node cycle: draft/critique/revise) ===")
    writer = make_agent("Writer", "You write short, punchy taglines.")
    workflow = Workflow().reflection().use_langgraph()
    workflow.add(writer)
    result = workflow.run(
        "Write a one-sentence tagline for an open-source AI framework.", max_rounds=3
    )
    print(result.content)
    print(f"(rounds of self-critique/revision: {len(result.steps)})")


def hierarchical_on_langgraph() -> None:
    print("\n=== hierarchical (langgraph) ===")
    team_lead = make_agent(
        "TeamLead", "You coordinate a small research team, delegating one subtask at a time."
    )
    team_researcher = make_agent("TeamResearcher")
    research_team = Workflow(name="ResearchTeam").supervisor()
    research_team.add(team_lead).add(team_researcher)

    director = make_agent(
        "Director",
        "You coordinate a small organization, delegating to either the Writer "
        "or the ResearchTeam, one subtask at a time.",
    )
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().hierarchical().use_langgraph()
    workflow.add(director).add(writer).add(research_team)
    result = workflow.run(
        "Research what the Model Context Protocol (MCP) is, then write a two-sentence summary.",
        max_rounds=6,
    )
    print(result.content)


def graph_on_langgraph() -> None:
    print("\n=== graph (langgraph -- reuses native's own routing logic verbatim) ===")
    triage = make_agent(
        "Triage",
        "You triage incoming requests. If answering well requires factual research, "
        "respond with exactly: NEEDS_RESEARCH: <one-sentence reason>. Otherwise respond "
        "with exactly: DIRECT_ANSWER: <your answer>.",
    )
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().graph().use_langgraph()
    workflow.add(triage).add(researcher).add(writer)
    workflow.add_edge(
        "Triage", "Researcher", condition=lambda c: c.strip().startswith("NEEDS_RESEARCH")
    )
    workflow.add_edge(
        "Triage", "Writer", condition=lambda c: c.strip().startswith("DIRECT_ANSWER")
    )
    workflow.add_edge("Researcher", "Writer")

    direct = workflow.run("What is 2 + 2?")
    print(f"direct path: {[s.agent_name for s in direct.steps]} -> {direct.content}")


def main() -> None:
    sequential_on_langgraph()
    supervisor_on_langgraph()
    reflection_on_langgraph()
    hierarchical_on_langgraph()
    graph_on_langgraph()


if __name__ == "__main__":
    main()
