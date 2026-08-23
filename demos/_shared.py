"""
Shared helper for the demo scripts: one conservative RateLimiter per
process, so every Agent/AI a demo builds shares one real budget instead
of each auto-building its own from Settings' RATE_LIMIT_RPM.

Why this matters: Agent()/AI() built *without* an explicit rate_limiter=
each construct their own private RateLimiter from RATE_LIMIT_RPM (see
AI._build_default_rate_limiter() in the framework). Several uncoordinated
limiters, each individually honoring "12/min", can still combine to
exceed the *real* Gemini free-tier quota (typically 15/min) when they're
all drawing on the same API key -- confirmed live: running several demos
back to back hit real 429s from Gemini even though each demo's own
agents looked fine in isolation. Sharing one RateLimiter instance across
every Agent/AI in a process closes that within-process gap.

This can't fully close the gap *across* run_all.py's demos, though --
each demo runs as its own OS subprocess (so a real quota conflict can
still happen if one demo's calls are still in flight moments before the
next starts) -- run_all.py adds a short pause between demos for that
reason. See README.md's rate-limiting note for the full picture.
"""

from requisite import AI, Agent, RateLimiter
from requisite.config.settings import Settings

# Deliberately more conservative than Gemini's typical free-tier
# 15/minute -- leaves real headroom for the imperfect cross-process
# coordination described above, rather than aiming exactly at the limit.
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
