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

## Multi / Knowledge-Base

- **`src/multi/`**: Provider-agnostic RAG module for document indexing, retrieval, and augmented completion.
- Files:
  - **`knowledge_base.py`**: Core logic for indexing, chunking, embedding, storage, and search.
  - **`wrapper.py`**: Augmented completion entry point.
  - **`__init__.py`**: Re-exports `index`, `search`, `clear`, `store_size`, `complete_with_knowledge`.
- **Public API**: `from src.multi import index, search, clear, store_size, complete_with_knowledge`
- **Interface**:
  - `index(paths: list[str]) -> None`: Ingest documents or raw text.
  - `search(query: str, top_k: int = 5) -> list[dict]`: Retrieve top-k relevant chunks.
  - `complete_with_knowledge(messages: list[dict], query: str, top_k: int = 5) -> str`: Augment and complete.
  - `clear() -> None`: Reset in-memory store.
  - `store_size() -> int`: Get current number of stored chunks (for testing).
- **Storage**: In-memory list; no persistent DB.
- **Embeddings**: Uses OpenAI API (requires `OPENAI_API_KEY`).
- **New Environment Variables** (optional):

| Variable          | Purpose                                                        |
| ----------------- | -------------------------------------------------------------- |
| `KB_DATA_DIR`     | Default directory to scan for documents (optional convenience) |
| `EMBEDDING_MODEL` | OpenAI embedding model (default: `text-embedding-3-small`)     |

- **Data Flow**:

```mermaid
graph TD
    A[User] -->|index(paths)| B[Ingest & Chunk]
    B --> C[Embed]
    C --> D[Store in _STORE]
    A -->|search(query)| E[Embed query]
    E --> F[Cosine similarity on _STORE]
    F --> G[Top-k results]
    A -->|complete_with_knowledge(messages, query)| H[search]
    H --> I[Augment messages<br/>(provider-specific)]
    I --> J[complete()]
```

## KB Agent

- **`src/multi/file_store.py`**: Provider-aware file storage layer.
  - **OpenAI**: Creates and manages an OpenAI vector store (`client.vector_stores`) and an Assistants API assistant (`client.beta.assistants`) with `file_search` enabled. Files are uploaded via `vector_stores.file_batches.upload_and_poll`. Queries run as assistant threads; `file_citation` annotations signal a successful retrieval.
  - **Anthropic**: Delegates directly to `knowledge_base.index()` (in-memory vector store).
  - **Interface**:
    - `upload_files(paths: list[str]) -> None`: Route file upload to the active provider.
    - `query_openai(query: str) -> str`: Query the OpenAI assistant; returns response text if citations found, `""` otherwise.
    - `clear_openai() -> None`: Reset OpenAI module-level store/assistant IDs.

- **`src/multi/kb_agent.py`**: Orchestration layer — KB first, web search fallback.
  - **Interface**: `ask(query: str) -> str`
  - **Provider dispatch**:

| `LLM_PROVIDER` | KB search | Fallback |
|---|---|---|
| `openai` | `query_openai()` — file_citation present? | `OpenAI().responses.create` with `web_search_preview` |
| `anthropic` | `search()` from `knowledge_base` — non-empty? | `Anthropic().messages.create` with `web_search_20250305` |

  - **Fallback trigger**: OpenAI — `query_openai()` returns `""`; Anthropic — `search()` returns `[]`.
  - Both web search mechanisms are **first-party** (no third-party service).

- **New Environment Variables** (optional):

| Variable | Purpose |
|---|---|
| `OPENAI_VECTOR_STORE_ID` | Pre-existing OpenAI vector store ID (created at runtime if absent) |
| `OPENAI_ASSISTANT_ID` | Pre-existing OpenAI assistant ID (created at runtime if absent) |
