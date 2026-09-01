"""
03 - The same workflow API, switched to the langgraph execution backend.

The point: `.use_langgraph()` is a one-line configuration change --
`.add()`/`.run()` never change. supervisor/hierarchical/reflection/critic
build a real `add_conditional_edges` graph (a loop-back cycle) on
langgraph instead of a Python loop; parallel/consensus/map_reduce fan out
to concurrent nodes in one superstep joined by an aggregator; debate/
tree_of_thoughts unroll their fixed-shape rounds/levels into a static
sequence at graph-build time; planner runs one upfront structured-output
call, then a bounded loop-back cycle over the plan's own length;
reflexion is a 3-node attempt/evaluate/reflect cycle, structurally close
to reflection/critic except the loop-back condition is a pluggable
evaluator's success signal. See docs/adr/0016, 0028, 0029, 0032, 0033,
0034, 0035, 0037 in the requisite-ai repo for the design behind each.

Run with:
    python demos/03_workflows_langgraph.py
"""

from _shared import make_agent
from requisite import EvaluationResult, Workflow


def sequential_on_langgraph() -> None:
    print("=== sequential (langgraph) ===")
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().use_langgraph()
    workflow.add(researcher).add(writer)
    result = workflow.run("Research AI agent frameworks and summarize the landscape.")
    print(result.content)
    print(f"(orchestrator={result.orchestrator}, strategy={result.strategy})")


def parallel_on_langgraph() -> None:
    print("\n=== parallel (langgraph -- concurrent nodes in one superstep, joined by an aggregator) ===")
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().parallel().use_langgraph()
    workflow.add(researcher).add(writer)
    result = workflow.run("What is retrieval-augmented generation?")
    print(result.content)


def consensus_on_langgraph() -> None:
    print("\n=== consensus (langgraph -- concurrent participants joined into one synthesizer node) ===")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")
    writer_a = make_agent("WriterA")
    writer_b = make_agent("WriterB")

    workflow = Workflow().consensus().use_langgraph()
    workflow.add(writer).add(writer_a).add(writer_b)
    result = workflow.run(
        "In one sentence, what is the single biggest benefit of retrieval-augmented generation?"
    )
    print(result.content)
    print(f"(synthesized from: {[s.agent_name for s in result.steps[:-1]]})")


def map_reduce_on_langgraph() -> None:
    print("\n=== map_reduce (langgraph -- one node per item, joined into one reducer node) ===")
    mapper_a = make_agent("MapperA")
    mapper_b = make_agent("MapperB")
    reducer = make_agent("Reducer", "You combine several short summaries into one coherent overview.")

    workflow = Workflow().map_reduce().use_langgraph()
    workflow.add(reducer).add(mapper_a).add(mapper_b)
    result = workflow.run(
        "Summarize the key idea of each note in one sentence each.",
        map_items=[
            "RAG grounds LLM answers in retrieved documents to reduce hallucination.",
            "MCP standardizes how AI applications connect to external tools and data.",
            "LangGraph models agent workflows as graphs instead of linear chains.",
        ],
    )
    print(result.content)
    print(f"(mapped by: {[s.agent_name for s in result.steps[:-1]]})")


def planner_on_langgraph() -> None:
    print(
        "\n=== planner (langgraph -- one upfront structured-output call produces the whole "
        "plan, then a bounded loop-back cycle executes each step) ==="
    )
    planner_agent = make_agent(
        "Planner", "You break tasks into an ordered plan of subtasks for your team."
    )
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().planner().use_langgraph()
    workflow.add(planner_agent).add(researcher).add(writer)
    result = workflow.run("Research what MCP (Model Context Protocol) is and write a short explainer.")
    print(result.content)
    print(f"(plan had {len(result.steps)} step(s): {[s.agent_name for s in result.steps]})")


def critic_on_langgraph() -> None:
    print("\n=== critic (langgraph -- reuses reflection's graph builder, generalized to two agents) ===")
    writer = make_agent("Writer", "You write short, punchy taglines.")
    critic_agent = make_agent(
        "Critic",
        "You critique short taglines for clarity and punch. If a tagline is "
        "already excellent, respond with exactly NO_CHANGES_NEEDED.",
    )

    workflow = Workflow().critic().use_langgraph()
    workflow.add(writer).add(critic_agent)
    result = workflow.run(
        "Write a one-sentence tagline for an open-source AI framework.", max_rounds=3
    )
    print(result.content)
    print(f"(rounds of generator/critic exchange: {len(result.steps)})")


def debate_on_langgraph() -> None:
    print("\n=== debate (langgraph -- max_rounds unrolled into a static sequence at build time) ===")
    debater_for = make_agent(
        "ForMonolith", "You argue FOR building a new backend as a monolith. Be concise."
    )
    debater_against = make_agent(
        "ForMicroservices", "You argue FOR building a new backend as microservices. Be concise."
    )
    moderator = make_agent(
        "Moderator", "You judge technical debates and deliver a balanced final verdict."
    )

    workflow = Workflow().debate().use_langgraph()
    workflow.add(moderator).add(debater_for).add(debater_against)
    result = workflow.run(
        "Should a small team building a new product start with a monolith or microservices?",
        max_rounds=2,
    )
    print(result.content)
    print(f"(debate steps + verdict: {len(result.steps)})")


def tree_of_thoughts_on_langgraph() -> None:
    print(
        "\n=== tree_of_thoughts (langgraph -- breadth/beam_width/max_depth fully determine "
        "each level's fan-out width at graph-build time) ==="
    )
    evaluator = make_agent(
        "Evaluator",
        "You evaluate candidate reasoning steps for a math word problem, scoring "
        "how promising and correct each one is.",
    )
    thinker = make_agent(
        "Thinker", "You propose the next reasoning step to solve a math word problem."
    )

    workflow = Workflow().tree_of_thoughts().use_langgraph()
    workflow.add(evaluator).add(thinker)
    result = workflow.run(
        "A train travels 60 miles in the first hour and 90 miles in the "
        "second hour. What is its average speed over the two hours?",
        breadth=3,
        beam_width=2,
        max_depth=3,
    )
    print(result.content)
    print(f"(generated {len(result.steps)} candidate thoughts)")


def reflexion_on_langgraph() -> None:
    print(
        "\n=== reflexion (langgraph -- attempt/evaluate/reflect cycle, loop-back condition "
        "is the evaluator's success signal) ==="
    )
    solver = make_agent("Solver", "You solve arithmetic word problems, showing your work briefly.")

    def check_contains_391(task: str, attempt: str) -> EvaluationResult:
        if "391" in attempt:
            return EvaluationResult(success=True, feedback="Correct.")
        return EvaluationResult(
            success=False,
            feedback="The final numeric answer is wrong -- recompute 17 * 23 carefully.",
        )

    workflow = Workflow().reflexion().use_langgraph()
    workflow.add(solver)
    result = workflow.run("What is 17 * 23?", evaluator=check_contains_391, max_trials=3)
    print(result.content)
    print(f"(succeeded: {result.succeeded}, {len(result.steps)} step(s) across up to 3 trials)")


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
    parallel_on_langgraph()
    consensus_on_langgraph()
    map_reduce_on_langgraph()
    supervisor_on_langgraph()
    planner_on_langgraph()
    reflection_on_langgraph()
    critic_on_langgraph()
    debate_on_langgraph()
    tree_of_thoughts_on_langgraph()
    reflexion_on_langgraph()
    hierarchical_on_langgraph()
    graph_on_langgraph()


if __name__ == "__main__":
    main()
