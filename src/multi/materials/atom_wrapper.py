"""High-level wrapper for running AtomAgent materials simulations.

This module provides a clean integration point for teammates who need to call
AtomAgent from the multi-agent pipeline without managing environment bootstraps
or low-level driver setup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from src.multi.materials.atom_driver import AtomAgentDriver


class AtomWrapper:
    """Thin orchestration wrapper around ``AtomAgentDriver``.

    The wrapper is responsible for:
    - loading environment variables from repository-level dotenv files,
    - validating that Anthropic credentials are available, and
    - exposing a simple ``run(prompt)`` API for callers.
    """

    def __init__(self, model: str | None = None, temp: float = 0.1) -> None:
        """Initialize AtomAgent runtime dependencies.

        Args:
            model: Optional Anthropic model override. If omitted, the wrapper
                resolves ``ANTHROPIC_MODEL`` from the environment and falls back
                to Sonnet 4.6 alias (``claude-sonnet-4-6``).
            temp: Sampling temperature forwarded to ``AtomAgentDriver``.

        Raises:
            RuntimeError: If ``ANTHROPIC_API_KEY`` is missing after dotenv load.
        """
        self._repo_root = Path(__file__).resolve().parents[3]
        self._load_environment()

        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._ensure_anthropic_api_key()
        self._driver = AtomAgentDriver(model=self.model, temp=temp)

    def _load_environment(self) -> None:
        """Load `.env.example` defaults and `.env` local overrides.

        Loading order is intentional:
        1. `.env.example` seeds documented defaults for local development.
        2. `.env` overrides with machine-specific secrets and model choices.
        """
        env_example_path = self._repo_root / ".env.example"
        env_path = self._repo_root / ".env"

        if env_example_path.exists():
            load_dotenv(dotenv_path=env_example_path, override=False)
        load_dotenv(dotenv_path=env_path, override=False)

    def _ensure_anthropic_api_key(self) -> None:
        """Validate that an Anthropic API key exists in environment variables."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Missing ANTHROPIC_API_KEY. Please add it to your repository root "
                ".env file (based on .env.example) and rerun."
            )

    def run(self, prompt: str) -> Dict[str, str | float]:
        """Execute a materials request through AtomAgent.

        Args:
            prompt: Natural-language materials prompt to send to AtomAgent.

        Returns:
            Driver metrics payload with:
            - ``output``: text returned by AtomAgent subprocess.
            - ``latency_seconds``: elapsed runtime for the call.
        """
        # The wrapper intentionally delegates execution details to AtomAgentDriver
        # so callers only need to provide a prompt.
        return self._driver.run_materials_sim(prompt)
