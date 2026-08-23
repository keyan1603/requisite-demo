"""
02 - Multi-agent workflows on the native (default, dependency-free) backend.

Covers the core execution strategies: sequential, parallel, supervisor,
hierarchical, reflection, and graph (developer-declared routing).

Run with:
    python demos/02_workflows_native.py
"""

from _shared import make_agent
from requisite import END, Workflow


def sequential_and_parallel() -> None:
    print("=== sequential ===")
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow()
    workflow.add(researcher).add(writer)
    result = workflow.run("Research AI agent frameworks and summarize the landscape.")
    print(result.content)

    print("\n=== parallel ===")
    workflow.parallel()
    result = workflow.run("What is retrieval-augmented generation?")
    print(result.content)


def supervisor() -> None:
    print("\n=== supervisor ===")
    coordinator = make_agent(
        "Coordinator", "You coordinate a small team, delegating one subtask at a time."
    )
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().supervisor()
    workflow.add(coordinator).add(researcher).add(writer)
    result = workflow.run("Research what MCP is and write a short explainer.", max_rounds=6)
    print(result.content)
    print(f"(delegated to: {[s.agent_name for s in result.steps]})")


def hierarchical() -> None:
    print("\n=== hierarchical (a delegate can be a nested Workflow 'team') ===")
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

    workflow = Workflow().hierarchical()
    workflow.add(director).add(writer).add(research_team)
    result = workflow.run(
        "Research what the Model Context Protocol (MCP) is, then write a two-sentence summary.",
        max_rounds=6,
    )
    print(result.content)
    print(
        "(delegated to: "
        f"{[getattr(s, 'agent_name', 'ResearchTeam (nested Workflow)') for s in result.steps]})"
    )


def reflection() -> None:
    print("\n=== reflection (an agent critiques and revises its own output) ===")
    writer = make_agent("Writer", "You write short, punchy taglines.")
    workflow = Workflow().reflection()
    workflow.add(writer)
    result = workflow.run(
        "Write a one-sentence tagline for an open-source AI framework.", max_rounds=3
    )
    print(result.content)
    print(f"(rounds of self-critique/revision: {len(result.steps)})")


def graph_with_conditional_routing() -> None:
    print("\n=== graph (routing decided by developer-declared conditions, not the model) ===")
    triage = make_agent(
        "Triage",
        "You triage incoming requests. If answering well requires factual research, "
        "respond with exactly: NEEDS_RESEARCH: <one-sentence reason>. Otherwise respond "
        "with exactly: DIRECT_ANSWER: <your answer>.",
    )
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().graph()
    workflow.add(triage).add(researcher).add(writer)
    workflow.add_edge(
        "Triage", "Researcher", condition=lambda c: c.strip().startswith("NEEDS_RESEARCH")
    )
    workflow.add_edge(
        "Triage", "Writer", condition=lambda c: c.strip().startswith("DIRECT_ANSWER")
    )
    workflow.add_edge("Researcher", "Writer")
    # Writer has no outgoing edges -> the graph terminates implicitly there.
    # (END is available too, for an explicit terminating edge -- see add_edge's docstring.)
    _ = END

    direct = workflow.run("What is 2 + 2?")
    print(f"direct path:   {[s.agent_name for s in direct.steps]} -> {direct.content}")

    researched = workflow.run("Research the current state of quantum error correction.")
    print(f"research path: {[s.agent_name for s in researched.steps]} -> {researched.content}")


def main() -> None:
    sequential_and_parallel()
    supervisor()
    hierarchical()
    reflection()
    graph_with_conditional_routing()


if __name__ == "__main__":
    main()
