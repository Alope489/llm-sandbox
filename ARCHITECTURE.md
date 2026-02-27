# Architecture

## Layout

- All application code lives under **`src/`**.
- Config and env stay at project root: `.env`, `requirements.txt`, `ARCHITECTURE.md`.

## LLM Wrapper

- **`src/wrapper.py`**: Single module that exposes a unified chat interface and switches between LLM providers via environment variable.
- **Interface**: `complete(messages: list[dict]) -> str`
  - **Input**: List of message dicts with `role` (`"user"`, `"assistant"`, `"system"`) and `content` (string).
  - **Output**: Assistant reply as a single string.
- **Default provider**: OpenAI. Set `LLM_PROVIDER=anthropic` in `.env` to use Anthropic.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `openai` (default) or `anthropic` |
| `OPENAI_API_KEY` | OpenAI API key (required when provider is openai) |
| `ANTHROPIC_API_KEY` | Anthropic API key (required when provider is anthropic) |
| `OPENAI_MODEL` | OpenAI model (default: gpt-4o-mini) |
| `ANTHROPIC_MODEL` | Anthropic model (default: claude-3-5-sonnet-20241022) |
| `MAX_TOKENS` | Max tokens for Anthropic (default: 1024) |

Load from `.env` via `python-dotenv` (called in `src/wrapper.py` on import).

## Linear pipeline

- **`src/linear/`**: Linear LLM pipeline. First step is the **extractor**.
- **`src/linear/extractor.py`**: Structured extraction from task descriptions.
  - **Interface**: `extract(text: str) -> dict`
  - **Input**: Raw task description (e.g. material/simulation prompt).
  - **Output**: Parsed dict matching the material/simulation schema (top-level keys: `material_system`, `processing_conditions`, `simulation_parameters`, `computed_properties`, `uncertainty_estimates`).
  - **Provider**: Uses `LLM_PROVIDER` (same as wrapper). **OpenAI**: Structured Outputs via `response_format` and `json_schema`. **Anthropic**: Tool use (single tool with `input_schema`, optional `strict`). Reuses `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `MAX_TOKENS`; no new env vars or dependencies.
- **`src/linear/__init__.py`**: Exposes `extract` (e.g. `from src.linear import extract`).
