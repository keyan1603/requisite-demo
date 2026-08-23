"""
01 - Quickstart: the AI facade and a tool-calling Agent.

Covers:
  - requisite.AI: the simplest way to send one chat message
  - requisite.Agent: a tool-calling loop around a provider
  - @tool: turning a plain Python function into something the model can call

Run with:
    python demos/01_quickstart.py
"""

from _shared import make_agent, make_ai
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


def main() -> None:
    quickstart_ai_facade()
    quickstart_agent_with_tools()


if __name__ == "__main__":
    main()
