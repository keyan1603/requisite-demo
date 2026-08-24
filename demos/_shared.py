"""
Shared helper for the demo scripts: one RateLimiter for the whole run,
so every Agent/AI a demo builds draws on one real, coherent budget
instead of each auto-building its own from Settings' RATE_LIMIT_RPM.

Why a single shared instance: Agent()/AI() built *without* an explicit
rate_limiter= each construct their own private RateLimiter from
RATE_LIMIT_RPM (see AI._build_default_rate_limiter() in the framework).
Several separately-paced limiters can each individually honor their own
"N/min" while still combining to exceed the *real* Gemini free-tier
quota, since they are all drawing on the same API key without
coordinating with each other. Passing this one shared instance to every
Agent/AI in the process gives it complete, accurate visibility into the
real call rate, which is what run_all.py relies on by running every demo
in one continuous process rather than as separate subprocesses.
"""

from requisite import AI, Agent, RateLimiter
from requisite.config.settings import Settings

# Deliberately more conservative than Gemini's typical free-tier
# 15/minute -- leaves real headroom rather than aiming exactly at the
# limit, since several agents in the same run share this one budget.
shared_rate_limit = RateLimiter(requests_per_minute=8, max_wait_seconds=180)

_settings = Settings()


def make_agent(name: str, system_prompt: str = "", **kwargs: object) -> Agent:
    return Agent(name=name, system_prompt=system_prompt, rate_limiter=shared_rate_limit, **kwargs)


def make_ai(**kwargs: object) -> AI:
    return AI(rate_limiter=shared_rate_limit, **kwargs)


def gemini_api_key() -> str | None:
    """GeminiEmbeddingProvider() (unlike Agent()/AI()) reads its key from
    the real OS environment directly, not from .env via Settings() --
    pydantic-settings parses .env into Settings' own typed fields without
    mutating os.environ. Pass this explicitly to GeminiEmbeddingProvider(
    api_key=...) instead of relying on its bare-constructor fallback,
    which silently sees nothing when a key only lives in .env."""
    return _settings.api_key_for("gemini")
