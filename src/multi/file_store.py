"""Provider-aware file storage: OpenAI Assistants vector store or Anthropic in-memory KB."""
import os
from dotenv import load_dotenv

load_dotenv()

_VECTOR_STORE_ID: str = None
_ASSISTANT_ID: str = None


def clear_openai() -> None:
    global _VECTOR_STORE_ID, _ASSISTANT_ID
    _VECTOR_STORE_ID = None
    _ASSISTANT_ID = None


def upload_files(paths: list) -> None:
    if os.environ.get("LLM_PROVIDER", "openai") == "anthropic":
        from .knowledge_base import index
        index(paths)
    else:
        _upload_openai(paths)


def _upload_openai(paths: list) -> None:
    from openai import OpenAI
    global _VECTOR_STORE_ID, _ASSISTANT_ID
    client = OpenAI()
    if _VECTOR_STORE_ID is None:
        _VECTOR_STORE_ID = client.vector_stores.create(name="kb").id
    file_streams = [open(p, "rb") for p in paths]
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=_VECTOR_STORE_ID,
        files=file_streams,
    )
    for f in file_streams:
        f.close()
    if _ASSISTANT_ID is None:
        _ASSISTANT_ID = client.beta.assistants.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [_VECTOR_STORE_ID]}},
        ).id
    else:
        client.beta.assistants.update(
            _ASSISTANT_ID,
            tool_resources={"file_search": {"vector_store_ids": [_VECTOR_STORE_ID]}},
        )


def query_openai(query: str) -> str:
    from openai import OpenAI
    if _ASSISTANT_ID is None:
        return ""
    client = OpenAI()
    thread_id = client.beta.threads.create(
        messages=[{"role": "user", "content": query}]
    ).id
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=_ASSISTANT_ID,
    )
    if run.status != "completed":
        return ""
    for msg in client.beta.threads.messages.list(thread_id=thread_id).data:
        if msg.role == "assistant":
            for block in msg.content:
                if hasattr(block, "text"):
                    if any(
                        getattr(a, "type", None) == "file_citation"
                        for a in block.text.annotations
                    ):
                        return block.text.value
    return ""
