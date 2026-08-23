"""
07 - MCP client: connecting to demos/06_mcp_server.py, both in the
default per-call-reconnect mode and in the opt-in persistent-session
mode (ADR-0030) -- and a timing comparison between the two.

Run with:
    python demos/07_mcp_client.py
"""

import asyncio
import sys
import time
from pathlib import Path

from _shared import make_agent
from requisite.capabilities import default_registry as capabilities
from requisite.mcp import MCPClient

SERVER_SCRIPT = str(Path(__file__).parent / "06_mcp_server.py")


def default_mode_example() -> None:
    print("=== Default mode: each call reconnects, calls, disconnects ===")
    client = MCPClient.stdio(name="requisite-demo", command=sys.executable, args=[SERVER_SCRIPT])

    tools = client.discover_tools()
    print(f"Discovered {len(tools)} tool(s): {[t.name for t in tools]}")

    resources = client.discover_resources()
    print(f"Discovered {len(resources)} resource(s): {[r.uri for r in resources]}")
    print(f"  readme -> {client.read_resource('memo://readme')}")

    prompts = client.discover_prompts()
    print(f"Discovered {len(prompts)} prompt(s): {[p.name for p in prompts]}")

    # get_prompt() returns Message objects, directly usable by an agent:
    messages = client.get_prompt("weather_report", {"city": "Paris"})
    agent = make_agent("Assistant")
    print(f"  weather_report(city='Paris') -> {agent.run(messages).content}")

    # Bridge a tool into the capability system:
    client.register_as_capability(capabilities, capability="add")
    agent2 = make_agent("Assistant2")
    agent2.requires("add")
    print(f"  agent using the bridged 'add' capability -> {agent2.run('What is 47 + 55?').content}")


async def persistent_mode_and_timing_example() -> None:
    print("\n=== Persistent-session mode (ADR-0030): connect once, reuse across calls ===")
    n = 10

    # Baseline: default mode, N calls, each reconnecting.
    default_client = MCPClient.stdio(
        name="requisite-demo", command=sys.executable, args=[SERVER_SCRIPT]
    )
    start = time.perf_counter()
    for _ in range(n):
        await default_client.adiscover_tools()
    default_elapsed = time.perf_counter() - start
    print(f"  default mode:    {n} calls in {default_elapsed * 1000:.1f}ms")

    # Persistent mode: one connection, reused for all N calls.
    async with MCPClient.stdio(
        name="requisite-demo", command=sys.executable, args=[SERVER_SCRIPT]
    ) as persistent_client:
        start = time.perf_counter()
        for _ in range(n):
            await persistent_client.adiscover_tools()
        persistent_elapsed = time.perf_counter() - start
    print(f"  persistent mode: {n} calls in {persistent_elapsed * 1000:.1f}ms")
    print(f"  speedup: ~{default_elapsed / max(persistent_elapsed, 1e-6):.0f}x")


def main() -> None:
    default_mode_example()
    asyncio.run(persistent_mode_and_timing_example())


if __name__ == "__main__":
    main()
