import os
import subprocess
import sys
import time
from typing import Dict


class AtomAgentDriver:
    def __init__(self, model: str = "gpt-4o-mini", temp: float = 0.1) -> None:
        self.model = model
        self.temp = temp
        self._atomagents_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "AtomAgents", "AtomAgents.py")
        )

    def run_materials_sim(self, prompt: str) -> Dict[str, str | float]:
        step_start = time.time()

        output = ""
        try:
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
            env["CHROMA_OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
            completed = subprocess.run(
                [sys.executable, self._atomagents_script, "--prompt", prompt],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            output = completed.stdout or ""
            if completed.returncode != 0 and completed.stderr:
                output = f"{output}\n{completed.stderr}".strip()
        except Exception as exc:
            output = f"Materials simulation failed: {exc}"

        step_end = time.time()
        latency_seconds = step_end - step_start

        return {
            "output": output,
            "latency_seconds": latency_seconds,
        }
