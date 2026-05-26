"""
Book structure-based chunking with quick stats for Gutenberg-style text.

New standalone file. Does not overwrite any existing file.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from lightrag.utils import TiktokenTokenizer, Tokenizer


SECTION_PATTERNS = [
    re.compile(r"^(STAVE\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN))\b", re.IGNORECASE),
    re.compile(r"^(CHAPTER\s+(?:\d+|[IVXLCDM]+))\b", re.IGNORECASE),
    re.compile(r"^(PART\s+(?:\d+|[IVXLCDM]+))\b", re.IGNORECASE),
    re.compile(r"^(BOOK\s+(?:\d+|[IVXLCDM]+))\b", re.IGNORECASE),
]


def _normalize(content: str) -> list[str]:
    return content.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _strip_gutenberg_wrapper(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)

    for i, line in enumerate(lines):
        if line.strip().startswith("*** START OF THE PROJECT GUTENBERG EBOOK"):
            start = i + 1
            break

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("*** END OF THE PROJECT GUTENBERG EBOOK"):
            end = i
            break

    return lines[start:end]


def _noise(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if t == "[Illustration]" or t.startswith("[Illustration"):
        return True
    if t in {"CONTENTS", "LIST OF ILLUSTRATIONS", "IN BLACK AND WHITE", "IN COLOUR"}:
        return True
    if re.search(r"\s{2,}\d{1,4}$", t) and ("STAVE" in t or "CHAPTER" in t or "PART" in t):
        return True
    return False


def _detect_section_heading(line: str) -> str | None:
    t = line.strip()
    for p in SECTION_PATTERNS:
        m = p.match(t)
        if m:
            return m.group(1).upper()
    return None


def _split_sections(content: str) -> list[tuple[str, str]]:
    lines = _strip_gutenberg_wrapper(_normalize(content))

    sections: list[tuple[str, str]] = []
    title = "FRONT_MATTER"
    body_lines: list[str] = []

    def flush() -> None:
        nonlocal title, body_lines
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((title, body))
        body_lines = []

    for line in lines:
        if _noise(line):
            continue

        heading = _detect_section_heading(line)
        if heading:
            flush()
            title = heading
            continue

        body_lines.append(line)

    flush()

    if not sections:
        return [("DOCUMENT", content)]

    return sections


def _token_window_split(
    tokenizer: Tokenizer,
    text: str,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> list[str]:
    toks = tokenizer.encode(text)
    if len(toks) <= chunk_token_size:
        return [text]

    step = max(1, chunk_token_size - chunk_overlap_token_size)
    parts: list[str] = []
    for i in range(0, len(toks), step):
        piece = tokenizer.decode(toks[i : i + chunk_token_size]).strip()
        if piece:
            parts.append(piece)
    return parts


def gutenberg_chunking_with_stats(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 120,
    chunk_token_size: int = 900,
) -> list[dict[str, Any]]:
    """LightRAG-compatible chunking_func with section-aware logic for book.txt style."""
    if split_by_character:
        units = [u.strip() for u in content.split(split_by_character) if u.strip()]
        if split_by_character_only:
            sections = [("UNIT", u) for u in units]
        else:
            sections: list[tuple[str, str]] = []
            for u in units:
                sections.extend(_split_sections(u))
    else:
        sections = _split_sections(content)

    chunks: list[dict[str, Any]] = []
    idx = 0

    for sec_title, sec_body in sections:
        text = f"{sec_title}\n\n{sec_body}".strip()
        for part in _token_window_split(
            tokenizer, text, chunk_token_size=chunk_token_size, chunk_overlap_token_size=chunk_overlap_token_size
        ):
            chunks.append(
                {
                    "tokens": len(tokenizer.encode(part)),
                    "content": part,
                    "chunk_order_index": idx,
                }
            )
            idx += 1

    return chunks


def print_chunk_stats(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        print("No chunks generated")
        return

    token_counts = [int(c["tokens"]) for c in chunks]
    print(f"chunks_total={len(chunks)}")
    print(f"tokens_avg={sum(token_counts) / len(token_counts):.2f}")
    print(f"tokens_min={min(token_counts)}")
    print(f"tokens_max={max(token_counts)}")

    by_section: dict[str, int] = defaultdict(int)
    for c in chunks:
        first_line = str(c["content"]).splitlines()[0].strip() if c["content"] else ""
        if first_line.startswith("STAVE ") or first_line.startswith("CHAPTER ") or first_line.startswith("PART "):
            by_section[first_line] += 1

    if by_section:
        print("\nchunks_by_section:")
        for name, count in sorted(by_section.items()):
            print(f"  {name}: {count}")


if __name__ == "__main__":
    source = Path("book.txt")
    if not source.exists():
        raise FileNotFoundError("book.txt not found in current directory")

    text = source.read_text(encoding="utf-8", errors="ignore")
    tokenizer = TiktokenTokenizer("gpt-4o-mini")

    chunks = gutenberg_chunking_with_stats(
        tokenizer,
        text,
        chunk_token_size=900,
        chunk_overlap_token_size=120,
    )

    print_chunk_stats(chunks)
    print("\npreview_first_2_chunks:")
    for c in chunks[:2]:
        preview = c["content"][:280].replace("\n", " ")
        print(f"- idx={c['chunk_order_index']} tokens={c['tokens']} text={preview}...")
