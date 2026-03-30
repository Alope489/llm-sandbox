# KB Agent Token Throughput Notes

## Why input tokens can be larger than output tokens

In `sandbox/kb_agent.py`, `ask()` sends a short user query, but the model call also enables `file_search` on the vector store.  
That means `input_tokens` often includes more than the visible query text:

- Query text from the user.
- Tool schema and orchestration tokens for `file_search`.
- Retrieved knowledge-base chunks injected into context.
- Internal system/tool-routing tokens from the Responses API.

The model can still answer concisely, so `output_tokens` may stay small while `input_tokens` is much larger.

## What this means when reading metrics

- `input_tokens > output_tokens` is expected for retrieval-heavy calls.
- A short query does not imply low total input tokens.
- Citation count helps interpret whether retrieval likely happened.

## Current implementation context

`ask()` currently records:

- `query`
- `answer`
- `input_tokens`
- `output_tokens`
- `citation_file_ids`
- `vector_store_id`

`output_prompt` was removed because it duplicated `answer`.

## Practical guidance

- Compare token usage across similar query types, not only query length.
- Track `citation_file_ids` alongside `input_tokens` for better diagnostics.
- Expect larger token throughput as corpus size and retrieval depth grow.
