"""
10 - Verifies the behavior fixed in the ADR-0031 code review pass,
against the real published package (not the framework's own internal
test suite). Each check prints PASS/FAIL rather than raising, so one
regression doesn't stop the rest from being checked.

See: https://github.com/requisite-ai/requisite-ai/blob/main/docs/adr/0031-code-review-fixes.md

Run with:
    python demos/10_recent_fixes.py
"""

import asyncio
import sys

from requisite import Agent, Workflow
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse, ToolCall
from requisite.orchestrators.native import _SupervisorDecision
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry
from requisite.tools import tool

_checks_passed = 0
_checks_failed = 0


def check(name: str, condition: bool) -> None:
    global _checks_passed, _checks_failed
    if condition:
        _checks_passed += 1
        print(f"  PASS: {name}")
    else:
        _checks_failed += 1
        print(f"  FAIL: {name}")


# --- A scripted (no live model call) provider, so the two structural
# fixes below (recursion-cycle detection, hallucinated-tool recovery)
# can be verified deterministically and instantly, the same way the
# framework's own test suite does it -- these aren't about model
# quality, they're about the framework's own control flow.


class AlwaysDelegateProvider(BaseProvider):
    """Always asks to delegate to the same named worker -- used to prove
    a self-referential Workflow cycle is caught, not left to recurse."""

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


class HallucinatedToolProvider(BaseProvider):
    """First turn: asks for a tool that was never registered. Second
    turn: gives a final answer -- proving the agent recovered instead of
    crashing on the first bad call."""

    def __init__(self, **kwargs) -> None:
        super().__init__(api_key="fake-key", model="fake-model")
        self._call_count = 0

    @property
    def name(self) -> str:
        return "hallucinate-then-answer"

    def chat(self, messages, **kwargs) -> ChatResponse:
        self._call_count += 1
        if self._call_count == 1:
            return ChatResponse(
                content="",
                model=self._model,
                provider=self.name,
                tool_calls=[ToolCall(id="call_1", name="totally_made_up_tool", arguments={})],
            )
        return ChatResponse(content="recovered", model=self._model, provider=self.name)

    async def achat(self, messages, **kwargs) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(self, messages, **kwargs):
        raise NotImplementedError

    async def astream(self, messages, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover


def make_scripted_agent(name: str, provider: BaseProvider) -> Agent:
    registry = ProviderRegistry()
    registry.register("scripted", lambda **kwargs: provider)
    from requisite.config.settings import Settings

    settings = Settings(default_provider="scripted", model="fake-model", rate_limit_rpm=None)
    return Agent(name=name, provider="scripted", settings=settings, registry=registry)


def workflow_self_reference_is_caught() -> None:
    print("=== Fix #1: self-referential Workflow delegation is caught, not a RecursionError ===")
    coordinator = make_scripted_agent("Coordinator", AlwaysDelegateProvider(worker="Team"))
    team = Workflow(name="Team").hierarchical()
    team.add(coordinator).add(team)  # Team delegates to itself

    try:
        team.run("go", max_rounds=1_000_000)  # would never terminate without the fix
        check("raises ConfigurationException", False)
    except ConfigurationException as exc:
        check("raises ConfigurationException", "delegation cycle detected" in str(exc))
    except RecursionError:
        check("raises ConfigurationException (got RecursionError instead -- regression!)", False)


def hallucinated_tool_recovers() -> None:
    print("\n=== Fix #2: a hallucinated tool call no longer aborts the whole run ===")
    agent = make_scripted_agent("Recovering", HallucinatedToolProvider())
    result = agent.run("do something", max_iterations=5)
    check("recovers and reaches a final answer", result.content == "recovered")
    check("iterations used the recovery turn", result.iterations == 2)


async def concurrent_tool_failure_does_not_orphan_siblings() -> None:
    print("\n=== Fix #3: one failing concurrent tool call doesn't abandon the others ===")

    @tool
    def good_tool(x: int) -> int:
        return x * 2

    class PartialFailureProvider(BaseProvider):
        def __init__(self, **kwargs) -> None:
            super().__init__(api_key="fake-key", model="fake-model")
            self._call_count = 0

        @property
        def name(self) -> str:
            return "partial-failure"

        def chat(self, messages, **kwargs) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content="",
                    model=self._model,
                    provider=self.name,
                    tool_calls=[
                        ToolCall(id="c1", name="totally_made_up_tool", arguments={}),
                        ToolCall(id="c2", name="good_tool", arguments={"x": 21}),
                    ],
                )
            return ChatResponse(content="done", model=self._model, provider=self.name)

        async def achat(self, messages, **kwargs) -> ChatResponse:
            return self.chat(messages, **kwargs)

        def stream(self, messages, **kwargs):
            raise NotImplementedError

        async def astream(self, messages, **kwargs):
            raise NotImplementedError
            yield  # pragma: no cover

    registry = ProviderRegistry()
    provider = PartialFailureProvider()
    registry.register("scripted", lambda **kwargs: provider)
    from requisite.config.settings import Settings

    settings = Settings(default_provider="scripted", model="fake-model", rate_limit_rpm=None)
    agent = Agent(
        name="PartialFailure",
        provider="scripted",
        settings=settings,
        registry=registry,
        tools=[good_tool],
    )
    result = await agent.arun("do something")
    check("reaches a final answer despite one bad tool call", result.content == "done")
    check(
        "both tool calls were attempted",
        set(result.tool_calls_executed) == {"totally_made_up_tool", "good_tool"},
    )


def tool_schema_survives_unresolvable_annotation() -> None:
    print("\n=== Fix #4: @tool no longer crashes on an unresolvable type hint ===")
    try:

        @tool
        def weird_tool(x: "SomeTypeThatDoesNotExistAnywhere") -> str:  # noqa: F821
            """A tool with a forward-reference annotation that can never resolve."""
            return str(x)

        check(
            "registers with a permissive fallback schema",
            weird_tool.tool.parameters_schema["properties"]["x"] == {"type": "string"},
        )
    except NameError:
        check("registers with a permissive fallback schema (got NameError instead)", False)


def main() -> int:
    """Returns an exit code (0/1) rather than calling sys.exit() directly,
    so run_all.py can import this module and call main() in-process
    without a failing check tearing down the whole run_all.py process --
    sys.exit()/SystemExit would otherwise propagate straight out of an
    in-process call the same way it does for a standalone run."""
    workflow_self_reference_is_caught()
    hallucinated_tool_recovers()
    asyncio.run(concurrent_tool_failure_does_not_orphan_siblings())
    tool_schema_survives_unresolvable_annotation()

    print(f"\n{_checks_passed} passed, {_checks_failed} failed")
    return 1 if _checks_failed else 0


if __name__ == "__main__":
    sys.exit(main())
