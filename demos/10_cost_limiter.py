"""
10 - CostLimiter: capping real dollar spend, independent of and
composable with RateLimiter (one paces call rate, the other caps call
cost). Pricing is a caller-supplied cost_fn -- no price table is
shipped or maintained by the framework.

The budget below is set deliberately low so the exhaustion path
actually triggers against real usage numbers, not just a scripted one.

Run with:
    python demos/10_cost_limiter.py
"""

from _shared import gemini_api_key
from requisite import Agent, CostLimitException, CostLimiter, RateLimiter, cost_per_token

# Illustrative Gemini 2.5 Flash-ish rates, not guaranteed current --
# always confirm against your provider's own current pricing page.
cost_fn = cost_per_token(prompt_rate_per_1k=0.000075, completion_rate_per_1k=0.0003)


def exhausts_a_tiny_budget() -> None:
    print("=== CostLimiter: budget exhaustion on real usage ===")

    # Deliberately tiny -- not shared via _shared.py's shared_rate_limit on
    # purpose, so this demo's own tiny budget_usd is exercised in isolation
    # without competing with every other demo's calls in run_all.py.
    budget = CostLimiter(budget_usd=0.000005, cost_fn=cost_fn)
    agent = Agent(
        name="Assistant",
        provider="gemini",
        system_prompt="You are a concise assistant. Answer in one short sentence.",
        rate_limiter=RateLimiter(requests_per_minute=8, max_wait_seconds=180),
        cost_limiter=budget,
    )

    prompts = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Peru?",
    ]
    for prompt in prompts:
        try:
            result = agent.run(prompt)
        except CostLimitException as exc:
            print(f"\nBudget exhausted, call blocked before reaching the provider: {exc}")
            break
        print(f"\n{prompt}\n{result.content}")
        print(f"(spent so far: ${budget.spent_usd:.6f}, remaining: ${budget.remaining_usd:.6f})")

    print(f"\nFinal: spent ${budget.spent_usd:.6f}, remaining ${budget.remaining_usd:.6f}")

    budget.reset()
    print(f"After reset(): spent ${budget.spent_usd:.6f}")


def main() -> None:
    if not gemini_api_key():
        print("Skipping 10_cost_limiter.py: no GEMINI_API_KEY configured.")
        return
    exhausts_a_tiny_budget()


if __name__ == "__main__":
    main()
