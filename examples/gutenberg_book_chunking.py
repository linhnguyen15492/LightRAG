"""
Book-aware structure chunking for Gutenberg-like plain text files.

This file is standalone and does not modify existing project code.
It is designed for inputs similar to book.txt (A Christmas Carol style).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from lightrag import LightRAG
from lightrag.utils import Tokenizer


@dataclass
class Section:
    title: str
    body: str


_MAIN_SECTION_PATTERNS = [
    re.compile(r"^STAVE\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b", re.IGNORECASE),
    re.compile(r"^CHAPTER\s+([0-9]+|[IVXLCDM]+)\b", re.IGNORECASE),
    re.compile(r"^PART\s+([0-9]+|[IVXLCDM]+)\b", re.IGNORECASE),
    re.compile(r"^BOOK\s+([0-9]+|[IVXLCDM]+)\b", re.IGNORECASE),
]

_PRELUDE_HEADING_PATTERNS = [
    re.compile(r"^PREFACE$", re.IGNORECASE),
    re.compile(r"^INTRODUCTION$", re.IGNORECASE),
    re.compile(r"^PROLOGUE$", re.IGNORECASE),
]

_NOISE_HEADINGS = {
    "CONTENTS",
    "LIST OF ILLUSTRATIONS",
    "IN BLACK AND WHITE",
    "IN COLOUR",
    "CHARACTERS",
}


def _normalize_lines(content: str) -> list[str]:
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    return [line.rstrip() for line in text.split("\n")]


def _is_main_section_heading(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    return any(p.match(text) for p in _MAIN_SECTION_PATTERNS)


def _is_prelude_heading(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if any(p.match(text) for p in _PRELUDE_HEADING_PATTERNS):
        return True
    return text.isupper() and 3 <= len(text) <= 60 and text not in _NOISE_HEADINGS


def _is_noise_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False

    if text.startswith("*** START OF THE PROJECT GUTENBERG EBOOK"):
        return True
    if text.startswith("*** END OF THE PROJECT GUTENBERG EBOOK"):
        return True

    if text.startswith("[Illustration") or text == "[Illustration]":
        return True

    # TOC-ish lines: "STAVE TWO--...   37"
    if re.search(r"\s{2,}[0-9]{1,4}$", text) and ("STAVE" in text or "CHAPTER" in text or "PART" in text):
        return True

    if text.upper() in _NOISE_HEADINGS:
        return True

    return False


def _trim_gutenberg_front_matter(lines: list[str]) -> list[str]:
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


def _split_book_sections(content: str) -> list[Section]:
    lines = _trim_gutenberg_front_matter(_normalize_lines(content))

    sections: list[Section] = []
    current_title = ""
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_body
        body = "\n".join(current_body).strip()
        if current_title or body:
            sections.append(Section(title=current_title.strip(), body=body))
        current_title = ""
        current_body = []

    for line in lines:
        stripped = line.strip()

        if _is_noise_line(stripped):
            continue

        if _is_main_section_heading(stripped):
            flush()
            current_title = stripped
            continue

        # Capture one heading block before first chapter as a prelude section.
        if not sections and not current_title and _is_prelude_heading(stripped):
            flush()
            current_title = stripped
            continue

        current_body.append(line)

    flush()

    # Fallback: if heading split failed, use paragraph groups.
    if len(sections) <= 1:
        plain = "\n".join(lines)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", plain) if p.strip()]
        if len(paragraphs) > 1:
            return [Section(title=f"Section {i + 1}", body=p) for i, p in enumerate(paragraphs)]

    return [s for s in sections if s.body.strip()]


def _split_oversized(
    tokenizer: Tokenizer,
    text: str,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> list[str]:
    tokens = tokenizer.encode(text)
    if len(tokens) <= chunk_token_size:
        return [text]

    step = max(1, chunk_token_size - chunk_overlap_token_size)
    parts: list[str] = []
    for start in range(0, len(tokens), step):
        part = tokenizer.decode(tokens[start : start + chunk_token_size]).strip()
        if part:
            parts.append(part)
    return parts


def gutenberg_structure_chunking(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """
    Structure-first chunking for book-like plain text.

    Output format is compatible with LightRAG chunking_func.
    """
    # Optional caller override: force split first then parse per fragment.
    if split_by_character:
        pieces = [p.strip() for p in content.split(split_by_character) if p.strip()]
        if split_by_character_only:
            sections = [Section(title="", body=p) for p in pieces]
        else:
            sections = []
            for p in pieces:
                sections.extend(_split_book_sections(p))
    else:
        sections = _split_book_sections(content)

    chunks: list[dict[str, Any]] = []
    idx = 0

    for sec in sections:
        base_text = f"{sec.title}\n\n{sec.body}".strip() if sec.title else sec.body.strip()
        if not base_text:
            continue

        for part in _split_oversized(
            tokenizer,
            base_text,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
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


def build_rag_for_books(**kwargs: Any) -> LightRAG:
    return LightRAG(chunking_func=gutenberg_structure_chunking, **kwargs)


if __name__ == "__main__":
    from pathlib import Path
    from lightrag.utils import TiktokenTokenizer

    sample_path = Path("book.txt")
    if sample_path.exists():
        text = sample_path.read_text(encoding="utf-8", errors="ignore")
    else:
        text = "STAVE ONE\nMarley was dead: to begin with.\n\nSTAVE TWO\nAnother section..."

    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    out = gutenberg_structure_chunking(tokenizer, text, chunk_token_size=450, chunk_overlap_token_size=60)
    print(f"chunks={len(out)}")
    for c in out[:3]:
        print(f"\n[{c['chunk_order_index']}] tokens={c['tokens']}\n{c['content'][:250]}...")
