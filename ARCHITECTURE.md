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
