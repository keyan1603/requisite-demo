"""
01 - Quickstart: the AI facade and a tool-calling Agent.

Covers:
  - requisite.AI: the simplest way to send one chat message
  - requisite.Agent: a tool-calling loop around a provider
  - @tool: turning a plain Python function into something the model can call
  - what the tool-calling loop does when things don't go perfectly: a
    hallucinated tool call, one failing call in a concurrent batch, and
    a tool whose type hint can't be resolved

The first two sections need a real provider (see the "Add your API key"
step in the README). The rest use a scripted provider that returns a
fixed response instead of calling a real model, so they run instantly
and need no API key.

Run with:
    python demos/01_quickstart.py
"""

import asyncio

from _shared import make_agent, make_ai
from requisite import Agent
from requisite.config.settings import Settings
from requisite.core.interfaces import ChatResponse, ToolCall
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry
from requisite.tools import tool


def quickstart_ai_facade() -> None:
    print("=== AI facade: one-shot chat ===")
    ai = make_ai()  # reads DEFAULT_PROVIDER / MODEL / <PROVIDER>_API_KEY from .env
    reply = ai.chat("In one sentence, what is retrieval-augmented generation?")
    print(reply)


@tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (fake, for this demo)."""
    return f"It's sunny and 22C in {city}."


def quickstart_agent_with_tools() -> None:
    print("\n=== Agent: tool-calling loop ===")
    assistant = make_agent(
        "Assistant",
        "You are a helpful assistant. Use tools when they help answer the question.",
        tools=[add, get_weather],
    )

    result = assistant.run("What's 47 plus 55, and what's the weather like in Tokyo?")
    print(result.content)
    print(f"(tool calls made: {result.tool_calls_executed}, iterations: {result.iterations})")


# --- Scripted providers below: a fixed, pre-written response instead of
# a real model call, which makes it possible to demonstrate exactly how
# the tool-calling loop handles a specific situation on demand.


def _make_scripted_agent(name: str, provider: BaseProvider, tools: list | None = None) -> Agent:
    registry = ProviderRegistry()
    registry.register("scripted", lambda **kwargs: provider)
    settings = Settings(default_provider="scripted", model="fake-model", rate_limit_rpm=None)
    return Agent(
        name=name, provider="scripted", settings=settings, registry=registry, tools=tools or []
    )


class _HallucinatedToolProvider(BaseProvider):
    """First turn: asks for a tool that was never registered. Second
    turn: gives a final answer."""

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


def hallucinated_tool_call_recovers() -> None:
    print("\n=== A hallucinated tool call doesn't abort the run ===")
    agent = _make_scripted_agent("Recovering", _HallucinatedToolProvider())
    result = agent.run("do something", max_iterations=5)
    assert result.content == "recovered"
    assert result.iterations == 2
    print(f"  recovered and reached a final answer in {result.iterations} iterations")


class _PartialFailureProvider(BaseProvider):
    """Asks for two tool calls at once in a single turn, one of which
    doesn't exist."""

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
                    ToolCall(id="c2", name="add", arguments={"a": 20, "b": 22}),
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


async def _concurrent_tool_call_partial_failure() -> None:
    agent = _make_scripted_agent("PartialFailure", _PartialFailureProvider(), tools=[add])
    result = await agent.arun("do something")
    assert result.content == "done"
    assert set(result.tool_calls_executed) == {"totally_made_up_tool", "add"}
    print(f"  both calls were attempted: {result.tool_calls_executed}")


def concurrent_tool_call_partial_failure() -> None:
    print("\n=== One failing call in a concurrent batch doesn't abandon the others ===")
    asyncio.run(_concurrent_tool_call_partial_failure())


def tool_with_unresolvable_type_hint() -> None:
    print("\n=== @tool on a function with an unresolvable type hint ===")

    @tool
    def weird_tool(x: "SomeTypeThatDoesNotExistAnywhere") -> str:  # noqa: F821
        """A tool with a forward-reference annotation that can never resolve."""
        return str(x)

    schema = weird_tool.tool.parameters_schema["properties"]["x"]
    assert schema == {"type": "string"}
    print(f"  registered with a permissive fallback schema: {schema}")


def main() -> None:
    quickstart_ai_facade()
    quickstart_agent_with_tools()
    hallucinated_tool_call_recovers()
    concurrent_tool_call_partial_failure()
    tool_with_unresolvable_type_hint()


if __name__ == "__main__":
    main()
