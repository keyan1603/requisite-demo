"""
06 - A self-contained MCP server, exposing Requisite tools/agents over
stdio. Not meant to be run directly -- demos/07_mcp_client.py spawns
this as a subprocess and connects to it.

This is the *reverse* direction of MCPClient: any MCP client (Claude
Desktop, Claude Code, or Requisite's own MCPClient) can use this
server's tools/agent.
"""

from requisite import Agent
from requisite.core.interfaces import Message, Role
from requisite.mcp import MCPServer
from requisite.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (fake, for this demo)."""
    return f"It's sunny and 22C in {city}."


def main() -> None:
    assistant = Agent(
        name="assistant",
        system_prompt="You are a helpful assistant. Answer in one sentence.",
    )

    server = MCPServer(name="requisite-demo", tools=[add, get_weather], agents=[assistant])

    server.add_resource(
        "memo://readme",
        name="readme",
        description="A short note about this demo server.",
        mime_type="text/plain",
        content="This MCP server is served by Requisite -- see demos/06_mcp_server.py.",
    )

    server.add_prompt(
        "weather_report",
        description="Ask for a one-sentence weather report for a city.",
        render=lambda args: [
            Message(role=Role.USER, content=f"What's the weather like in {args['city']}?")
        ],
    )

    server.run_stdio()


if __name__ == "__main__":
    main()
