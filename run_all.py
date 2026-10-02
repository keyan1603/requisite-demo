"""
Runs every demo in demos/ in one continuous process and reports a
pass/fail summary.

Deliberately in-process, not one subprocess per demo: demos/_shared.py's
module-level `shared_rate_limit` is a single RateLimiter instance.
Importing every demo module into this same interpreter means that one
instance genuinely accumulates the full call history across all of
them, so its own wait/backoff logic has complete, correct visibility
into the real API quota for the whole run.

demos/06_mcp_server.py is skipped -- it's a server script meant to be
spawned as a real subprocess by demos/07_mcp_client.py's own
MCPClient.stdio(...) calls (an inherent part of how the MCP stdio
transport works, unrelated to how this runner invokes demos).
demos/_shared.py is a helper module, not a demo.

demos/11_adk_orchestrator.py is skipped for now -- it needs
requisite-ai>=0.38.0 (workflow.use_adk()), which this repo's venv can't
install yet since 0.38.0 isn't published to PyPI. Remove this skip once
it is and the venv is upgraded -- see README.md's "Verified" section.

Run with:
    python run_all.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path
from types import ModuleType

DEMOS_DIR = Path(__file__).parent / "demos"
SKIP = {"06_mcp_server.py", "_shared.py", "11_adk_orchestrator.py"}


def _load_module(script: Path) -> ModuleType:
    """Import a demo script as a module without adding demos/ as a real
    package -- mirrors how `python demos/X.py` resolves demos/_shared.py
    via the script's own directory on sys.path, so `from _shared import
    ...` inside each demo keeps working unchanged."""
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[script.stem] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    # demos/_shared.py imports functions like from_ imports (`from
    # _shared import make_agent`) that resolve relative to demos/ being
    # on sys.path -- true for `python demos/X.py` automatically, and
    # needs doing explicitly here since run_all.py itself lives one
    # directory up.
    sys.path.insert(0, str(DEMOS_DIR))

    scripts = sorted(p for p in DEMOS_DIR.glob("*.py") if p.name not in SKIP)
    results: dict[str, bool] = {}

    for script in scripts:
        print(f"\n{'=' * 70}\nRunning {script.name}\n{'=' * 70}", flush=True)
        try:
            module = _load_module(script)
            exit_code = module.main()
            results[script.name] = exit_code in (None, 0)
        except Exception:  # noqa: BLE001 - a crashing demo is a FAIL, not a run_all.py crash
            traceback.print_exc()
            results[script.name] = False

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    failures = [name for name, ok in results.items() if not ok]
    if failures:
        print(f"\n{len(failures)} of {len(results)} demo(s) failed: {failures}")
        return 1

    print(f"\nAll {len(results)} demo(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
