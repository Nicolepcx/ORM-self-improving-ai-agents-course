# sandbox_bridge.py
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Sequence

from mcp_run_python import code_sandbox

@asynccontextmanager
async def sandbox(dependencies: Optional[Sequence[str]] = None, log_handler=None):
    """
    Shared sandbox context. Use in a long-lived scope to avoid re-downloading wheels.
    """
    async with code_sandbox(dependencies=list(dependencies or []), log_handler=log_handler) as sb:
        yield sb

async def sandbox_eval(
    sb,
    code: str,
    vars: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """
    Run a code string inside the sandbox with a timeout.
    Returns the dict from code_sandbox: {status, stdout, stderr, return_value, error}
    """
    return await asyncio.wait_for(sb.eval(code, vars or {}), timeout=timeout)
