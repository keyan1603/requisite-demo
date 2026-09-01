"""
02 - Multi-agent workflows on the native (default, dependency-free) backend.

Covers all thirteen execution strategies: sequential, parallel, consensus,
map_reduce, supervisor, planner, hierarchical, reflection, critic,
debate, tree_of_thoughts, reflexion, and graph (developer-declared
routing), plus what happens when a hierarchical delegation graph
references itself.

Run with:
    python demos/02_workflows_native.py
"""

from _shared import make_agent
from requisite import Agent, END, EvaluationResult, Workflow
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse
from requisite.orchestrators.native import _SupervisorDecision
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry


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


def consensus() -> None:
    print("\n=== consensus (several agents answer independently, then one synthesizes) ===")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")
    writer_a = make_agent("WriterA")
    writer_b = make_agent("WriterB")

    workflow = Workflow().consensus()
    workflow.add(writer).add(writer_a).add(writer_b)
    result = workflow.run(
        "In one sentence, what is the single biggest benefit of retrieval-augmented generation?"
    )
    print(result.content)
    print(f"(synthesized from: {[s.agent_name for s in result.steps[:-1]]})")


def map_reduce() -> None:
    print("\n=== map_reduce (each mapper handles one item, then a reducer combines them) ===")
    mapper_a = make_agent("MapperA")
    mapper_b = make_agent("MapperB")
    reducer = make_agent("Reducer", "You combine several short summaries into one coherent overview.")

    workflow = Workflow().map_reduce()
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


def planner() -> None:
    print("\n=== planner (the first agent decomposes the task into a plan for named workers) ===")
    planner_agent = make_agent(
        "Planner", "You break tasks into an ordered plan of subtasks for your team."
    )
    researcher = make_agent("Researcher", "You research topics and produce concise bullet points.")
    writer = make_agent("Writer", "You turn research notes into one polished paragraph.")

    workflow = Workflow().planner()
    workflow.add(planner_agent).add(researcher).add(writer)
    result = workflow.run("Research what MCP (Model Context Protocol) is and write a short explainer.")
    print(result.content)
    print(f"(plan had {len(result.steps)} step(s): {[s.agent_name for s in result.steps]})")


def critic() -> None:
    print("\n=== critic (a separate agent reviews the writer's draft, unlike reflection's self-review) ===")
    writer = make_agent("Writer", "You write short, punchy taglines.")
    critic_agent = make_agent(
        "Critic",
        "You critique short taglines for clarity and punch. If a tagline is "
        "already excellent, respond with exactly NO_CHANGES_NEEDED.",
    )

    workflow = Workflow().critic()
    workflow.add(writer).add(critic_agent)
    result = workflow.run(
        "Write a one-sentence tagline for an open-source AI framework.", max_rounds=3
    )
    print(result.content)
    print(f"(rounds of generator/critic exchange: {len(result.steps)})")


def debate() -> None:
    print("\n=== debate (a moderator judges two agents arguing opposite sides of a trade-off) ===")
    debater_for = make_agent(
        "ForMonolith", "You argue FOR building a new backend as a monolith. Be concise."
    )
    debater_against = make_agent(
        "ForMicroservices", "You argue FOR building a new backend as microservices. Be concise."
    )
    moderator = make_agent(
        "Moderator", "You judge technical debates and deliver a balanced final verdict."
    )

    workflow = Workflow().debate()
    workflow.add(moderator).add(debater_for).add(debater_against)
    result = workflow.run(
        "Should a small team building a new product start with a monolith or microservices?",
        max_rounds=2,
    )
    print(result.content)
    print(f"(debate steps + verdict: {len(result.steps)})")


def tree_of_thoughts() -> None:
    print("\n=== tree_of_thoughts (candidate reasoning steps scored and pruned each level) ===")
    evaluator = make_agent(
        "Evaluator",
        "You evaluate candidate reasoning steps for a math word problem, scoring "
        "how promising and correct each one is.",
    )
    thinker = make_agent(
        "Thinker", "You propose the next reasoning step to solve a math word problem."
    )

    workflow = Workflow().tree_of_thoughts()
    workflow.add(evaluator).add(thinker)
    result = workflow.run(
        "A train travels 60 miles in the first hour and 90 miles in the "
        "second hour. What is its average speed over the two hours?",
        breadth=3,
        beam_width=2,
        max_depth=3,
    )
    print(result.content)
    print(f"(generated {len(result.steps)} candidate thoughts across up to 3 levels)")


def reflexion() -> None:
    print("\n=== reflexion (attempt, evaluate, and reflect before retrying from scratch) ===")
    solver = make_agent("Solver", "You solve arithmetic word problems, showing your work briefly.")

    def check_contains_391(task: str, attempt: str) -> EvaluationResult:
        if "391" in attempt:
            return EvaluationResult(success=True, feedback="Correct.")
        return EvaluationResult(
            success=False,
            feedback="The final numeric answer is wrong -- recompute 17 * 23 carefully.",
        )

    workflow = Workflow().reflexion()
    workflow.add(solver)
    result = workflow.run("What is 17 * 23?", evaluator=check_contains_391, max_trials=3)
    print(result.content)
    print(f"(succeeded: {result.succeeded}, {len(result.steps)} step(s) across up to 3 trials)")


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


class _AlwaysDelegateProvider(BaseProvider):
    """A scripted provider that always asks to delegate to the same
    named worker, no matter what it's asked -- used below to show what
    happens when a hierarchical delegation graph references itself. No
    live model call, so this runs instantly and needs no API key."""

    def __init__(self, *, worker: str, **kwargs) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self._worker = worker

    @property
    def name(self) -> str:
        return "always-delegate"

    def chat(self, messages, **kwargs) -> ChatResponse:
        decision = _SupervisorDecision(action="delegate", worker=self._worker, task="keep going")
        return ChatResponse(content="", model=self._model, provider=self.name, parsed=decision)

    async def achat(self, messages, **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(self, messages, **kwargs):
        raise NotImplementedError

    async def astream(self, messages, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover


def self_referential_delegation_is_caught() -> None:
    print(
        "\n=== hierarchical: a team that delegates to itself raises a clean error, "
        "not an infinite loop ==="
    )
    registry = ProviderRegistry()
    provider = _AlwaysDelegateProvider(worker="Team")
    registry.register("scripted", lambda **kwargs: provider)
    settings = Settings(default_provider="scripted", model="fake-model", rate_limit_rpm=None)
    coordinator = Agent(
        name="Coordinator", provider="scripted", settings=settings, registry=registry
    )

    team = Workflow(name="Team").hierarchical()
    team.add(coordinator).add(team)  # Team delegates to itself

    try:
        team.run("go", max_rounds=1_000_000)
        raise AssertionError("expected ConfigurationException")
    except ConfigurationException as exc:
        assert "delegation cycle detected" in str(exc)
        print(f"  raised cleanly: {exc}")


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
    consensus()
    map_reduce()
    supervisor()
    planner()
    hierarchical()
    self_referential_delegation_is_caught()
    reflection()
    critic()
    debate()
    tree_of_thoughts()
    reflexion()
    graph_with_conditional_routing()


if __name__ == "__main__":
    main()
