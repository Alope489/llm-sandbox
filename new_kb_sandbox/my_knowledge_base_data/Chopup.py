"""Build fixed-size KB chunk files from the solid-state synthesis JSON corpus.

Splits UTF-8 text into chunks of at most ``CHUNK_SIZE_BYTES`` (default 5 KiB),
preferring paragraph boundaries (``\\n\\n``), without dropping bytes. Writes
up to ``TARGET_OUTPUT_BYTES`` (default 5 MiB) of chunk data as numbered ``.txt``
files plus one concatenated ``kb_single.txt``.

Dependencies: json, pathlib, os.

Configuration: ``CHOPUP_DATASET_PATH``, ``CHOPUP_OUTPUT_DIR``, ``CHUNK_SIZE_BYTES``,
``TARGET_OUTPUT_BYTES`` (optional environment overrides).

Pillar alignment: no hardcoded paths without env fallbacks; single-purpose
functions; explicit byte semantics for chunk sizing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# UTF-8 leading-byte ranges (RFC 3629) for manual span calculation.
_U8_ASCII_LIMIT = 0x80
_U8_2_MIN = 0xC0
_U8_2_MAX = 0xDF
_U8_3_MIN = 0xE0
_U8_3_MAX = 0xEF
_U8_4_MIN = 0xF0
_U8_4_MAX = 0xF7

_SELF_CHECK_CHUNK_BYTES = 64


def _align_utf8_chunk_end(data: bytes, start: int, end: int) -> int:
    """Shrink ``end`` so ``data[start:end]`` is valid UTF-8.

    Args:
        data: Full corpus as UTF-8 bytes.
        start: Inclusive start index.
        end: Exclusive end index (may land inside a multibyte code point).

    Returns:
        Largest ``e`` with ``start < e <= end`` such that ``data[start:e]``
        decodes as UTF-8, or ``start`` if none (caller must advance).

    Complexity:
        O(end - start) worst case per call.
    """
    while end > start:
        try:
            data[start:end].decode("utf-8")
            return end
        except UnicodeDecodeError:
            end -= 1
    return start


def _advance_past_one_codepoint(data: bytes, start: int, limit: int) -> int:
    """Return index after the first UTF-8 code point starting at ``start``.

    Args:
        data: UTF-8 byte string.
        start: Index of first byte of a code point.
        limit: Maximum index (exclusive) for scanning.

    Returns:
        Index immediately after one decoded code point, capped by ``limit``.

    Raises:
        UnicodeDecodeError: If no valid code point starts at ``start``.
    """
    if start >= limit:
        return start
    first = data[start]
    if first < _U8_ASCII_LIMIT:
        return start + 1
    if _U8_2_MIN <= first <= _U8_2_MAX:
        span = 2
    elif _U8_3_MIN <= first <= _U8_3_MAX:
        span = 3
    elif _U8_4_MIN <= first <= _U8_4_MAX:
        span = 4
    else:
        raise UnicodeDecodeError("utf-8", data, start, start + 1, "invalid start byte")
    end = min(start + span, limit)
    data[start:end].decode("utf-8")
    return end


def create_chunks_utf8(text: str, chunk_size: int) -> list[str]:
    """Split ``text`` into UTF-8 chunks of at most ``chunk_size`` bytes.

    Paragraph boundaries (``\\n\\n``) are preferred when the break falls in the
    second half of the window. Every byte of ``text`` appears in exactly one chunk;
    joining chunks reproduces ``text``.

    Args:
        text: Full corpus string.
        chunk_size: Maximum UTF-8 byte length per chunk (must be positive).

    Returns:
        List of chunk strings whose concatenation equals ``text``.

    Raises:
        ValueError: If ``chunk_size`` is not positive.

    Complexity:
        O(len(text.encode('utf-8'))) with small per-chunk backward scans.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    data = text.encode("utf-8")
    n = len(data)
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = data[start:end]
            rel_break = window.rfind(b"\n\n")
            if rel_break != -1 and rel_break >= chunk_size // 2:
                end = start + rel_break
        if end < n:
            end = _align_utf8_chunk_end(data, start, end)
        if end == start:
            end = _advance_past_one_codepoint(data, start, n)
        chunk_bytes = data[start:end]
        chunks.append(chunk_bytes.decode("utf-8"))
        start = end
    return chunks


def _select_chunks_up_to_byte_budget(chunks: list[str], max_bytes: int) -> list[str]:
    """Return a prefix of ``chunks`` whose UTF-8 sizes sum to at most ``max_bytes``.

    Args:
        chunks: Chunk strings from ``create_chunks_utf8``.
        max_bytes: Non-negative total byte budget.

    Returns:
        Longest initial sublist such that the sum of ``len(c.encode('utf-8'))``
        does not exceed ``max_bytes``.

    Raises:
        ValueError: If ``max_bytes`` is negative.

    Complexity:
        O(k) where k is the number of selected chunks.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    out: list[str] = []
    used = 0
    for ch in chunks:
        b = len(ch.encode("utf-8"))
        if used + b > max_bytes:
            break
        out.append(ch)
        used += b
    return out


def _self_check() -> None:
    """Sanity check: paragraph split preserves round-trip and byte cap.

    Raises:
        RuntimeError: If round-trip or per-chunk byte cap is violated.
    """
    # Paragraph breaks + long run: ASCII-only to avoid ambiguous-unicode lint noise.
    s = "ab\n\n" + ("c" * 2000) + "\n\nend"
    cs = create_chunks_utf8(s, _SELF_CHECK_CHUNK_BYTES)
    joined = "".join(cs)
    if joined != s:
        raise RuntimeError("chunk round-trip mismatch")
    for c in cs:
        if len(c.encode("utf-8")) > _SELF_CHECK_CHUNK_BYTES:
            raise RuntimeError("chunk exceeds byte cap")


def _json_compact(value: object) -> str:
    """Serialize ``value`` as compact UTF-8 JSON for plain-text corpus lines.

    Returns:
        Single-line JSON string (no ASCII escaping of non-ASCII).
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _field_as_text(value: object) -> str:
    """Turn an arbitrary JSON-like field into a single-line string.

    Returns:
        ``"N/A"`` for ``None``, else ``str``/JSON text suitable for a corpus line.
    """
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return _json_compact(value)


def _format_target(entry: dict) -> str:
    """Prefer formula / name from nested ``target`` objects in the solid-state schema.

    Returns:
        Human-readable target summary string.
    """
    t = entry.get("target")
    if isinstance(t, dict):
        return (
            t.get("material_formula")
            or t.get("material_string")
            or _json_compact(t)
        )
    return _field_as_text(t)


def _format_precursors(entry: dict) -> str:
    """Join precursor formulas or strings; supports dict entries in the dataset.

    Returns:
        Comma-separated precursor labels, or ``"N/A"`` when absent.
    """
    raw = entry.get("precursors")
    if raw is None:
        return "N/A"
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return _field_as_text(raw)
    if not raw:
        return "N/A"
    parts: list[str] = []
    for p in raw:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict):
            parts.append(
                p.get("material_formula")
                or p.get("material_string")
                or _json_compact(p)
            )
        else:
            parts.append(_json_compact(p))
    return ", ".join(parts)


def _format_operations(entry: dict) -> str:
    """Summarize operation list (``string`` / ``type`` per op).

    Returns:
        Semicolon-separated operation labels, or ``"N/A"`` when absent.
    """
    raw = entry.get("operations")
    if raw is None:
        return "N/A"
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return _field_as_text(raw)
    if not raw:
        return "N/A"
    parts: list[str] = []
    for op in raw:
        if isinstance(op, dict):
            parts.append(
                op.get("string") or op.get("type") or _json_compact(op)
            )
        else:
            parts.append(_field_as_text(op))
    return "; ".join(parts)


def main() -> None:
    """Load dataset, chunk, write multi-file and single-file outputs under budget."""
    chunk_size = int(os.environ.get("CHUNK_SIZE_BYTES", "5120"))
    target_bytes = int(os.environ.get("TARGET_OUTPUT_BYTES", str(5 * 1024 * 1024)))
    dataset_path = os.environ.get(
        "CHOPUP_DATASET_PATH",
        "solid-state_dataset_2019-06-27_upd.json",
    )
    output_dir = Path(os.environ.get("CHOPUP_OUTPUT_DIR", "materials_science_kb"))

    _self_check()

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = ""
    seen_dois: set[str] = set()
    for entry in data:
        doi = entry.get("doi") or f"entry_{len(seen_dois)}"
        if doi in seen_dois:
            continue
        seen_dois.add(doi)
        para = f"### Synthesis Record {doi}\n"
        para += f"Target: {_format_target(entry)}\n"
        para += f"Precursors: {_format_precursors(entry)}\n"
        para += f"Operations: {_format_operations(entry)}\n"
        para += f"Conditions: {_field_as_text(entry.get('conditions'))}\n\n"
        corpus += para

    all_chunks = create_chunks_utf8(corpus, chunk_size)
    to_write = _select_chunks_up_to_byte_budget(all_chunks, target_bytes)

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(to_write):
        path = output_dir / f"kb_chunk_{i:04d}.txt"
        path.write_text(chunk, encoding="utf-8")

    single_path = output_dir / "kb_single.txt"
    single_path.write_text("".join(to_write), encoding="utf-8")

    written_bytes = sum(len(c.encode("utf-8")) for c in to_write)
    print(
        f"Wrote {len(to_write)} chunks (~{written_bytes:,} B UTF-8) "
        f"to {output_dir!s} (budget {target_bytes:,} B; chunk max {chunk_size} B). "
        f"Full corpus split into {len(all_chunks)} chunks "
        f"(~{len(corpus):,} chars)."
    )


if __name__ == "__main__":
    main()
