"""
Root test configuration.

Loads .env before any test module is imported so module-level skipif guards
and isolation runs work reliably. Uses absolute path resolution — immune to
cwd changes by IDEs, scripts, or debuggers. If .env is absent (e.g. CI
injects keys via secrets), the file is simply not loaded.

Pillar compliance:
  - Pillar 4 (maintainability): single source of truth for .env loading.
  - Pillar 5 (security): override=False respects CI-injected secrets over .env.
  - Pillar 7 (robustness): explicit path + existence guard prevents silent failures.
"""
from pathlib import Path

from dotenv import load_dotenv

# tests/conftest.py → tests/ → project_root (where .env lives)
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)  # override=False (default) — respects shell vars and CI secrets
