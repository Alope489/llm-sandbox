"""Runtime driver for delegating materials prompts to the AtomAgent script."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_EXAMPLE_PATH = _REPO_ROOT / ".env.example"
_ENV_PATH = _REPO_ROOT / ".env"


def _load_environment() -> None:
    """Load template and local environment files for AtomAgent configuration."""
    # Load template defaults first, then local overrides from .env if present.
    if _ENV_EXAMPLE_PATH.exists():
        load_dotenv(dotenv_path=_ENV_EXAMPLE_PATH, override=False)
    load_dotenv(dotenv_path=_ENV_PATH, override=False)


_load_environment()


class AtomAgentDriver:
    """Wrapper around the AtomAgents subprocess entrypoint."""

    def __init__(self, model: str | None = None, temp: float = 0.1) -> None:
        """
        Initialize runtime settings for the AtomAgent invocation.

        Args:
            model: Optional model override. Defaults to `ANTHROPIC_MODEL`.
            temp: Sampling temperature retained for compatibility.
        """
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.temp = temp
        self._atomagents_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "AtomAgents", "AtomAgents.py")
        )

    def run_materials_sim(self, prompt: str) -> Dict[str, str | float]:
        """
        Run a materials simulation prompt through AtomAgent.

        Args:
            prompt: Natural-language materials request.

        Returns:
            Dict containing subprocess output text and measured latency.
        """
        step_start = time.time()

        output = ""
        try:
            env = os.environ.copy()
            # Keep subprocess credentials model-driven and environment-based.
            env["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")
            env["ANTHROPIC_MODEL"] = self.model
            env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
            env["CHROMA_OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
            completed = subprocess.run(
                [sys.executable, self._atomagents_script, "--prompt", prompt],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            output = completed.stdout or ""
            if completed.returncode != 0 and completed.stderr:
                # Append stderr to preserve root cause details from the tool chain.
                output = f"{output}\n{completed.stderr}".strip()
        except Exception as exc:
            output = f"Materials simulation failed: {exc}"

        step_end = time.time()
        latency_seconds = step_end - step_start

        return {
            "output": output,
            "latency_seconds": latency_seconds,
        }
