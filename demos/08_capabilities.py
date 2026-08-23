"""
08 - Capability resolution: agent.requires(...) declares *what* an
agent needs, not which specific tool implements it.

Run with:
    python demos/08_capabilities.py
"""

import os

from _shared import make_agent
from requisite.capabilities import default_registry


def builtin_capabilities() -> None:
    print("=== Built-in capabilities (zero setup, free/keyless public APIs) ===")
    agent = make_agent("Assistant")
    agent.requires("weather", "internet_search", "github")

    result = agent.run("What's the weather in Tokyo right now?")
    print(result.content)
    print(f"Tools used: {result.tool_calls_executed}")

    github_result = agent.run("Find a popular Python framework for building AI agents on GitHub.")
    print(github_result.content)
    print(f"Tools used: {github_result.tool_calls_executed}")


def overriding_a_capability() -> None:
    print("\n=== Overriding a capability with a higher-priority implementation ===")

    def paid_weather_api(city: str) -> str:
        """Paid, more accurate weather provider."""
        return f"[precise forecast for {city}]"

    default_registry.register(
        "weather",
        paid_weather_api,
        provider_name="acme-weather",
        priority=10,
        is_available=lambda: bool(os.environ.get("ACME_WEATHER_API_KEY")),
    )

    agent = make_agent("Assistant2")
    agent.requires("weather")  # uses acme-weather if ACME_WEATHER_API_KEY is set, else falls back
    print(agent.run("Weather in Paris?").content)


def main() -> None:
    builtin_capabilities()
    overriding_a_capability()


if __name__ == "__main__":
    main()
