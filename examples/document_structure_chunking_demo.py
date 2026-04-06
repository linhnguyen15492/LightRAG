"""
Document structure-based chunking demo for LightRAG.

This file is standalone and does not modify existing project code.
You can import `structure_based_chunking` and pass it to LightRAG(chunking_func=...).
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


_HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+$"),  # Markdown headings
    re.compile(r"^\d+(?:\.\d+)*\s+.+$"),  # 1, 1.2, 2.3.4 style headings
    re.compile(r"^[A-Z][A-Z\s\-]{3,80}$"),  # ALL CAPS titles
]


def _is_heading(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    return any(pattern.match(text) for pattern in _HEADING_PATTERNS)


def _split_into_sections(content: str) -> list[Section]:
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: list[Section] = []
    current_title = ""
    current_body: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if _is_heading(line):
            if current_title or current_body:
                sections.append(
                    Section(title=current_title.strip(), body="\n".join(current_body).strip())
                )
            current_title = line.strip()
            current_body = []
        else:
            current_body.append(line)

    if current_title or current_body:
        sections.append(Section(title=current_title.strip(), body="\n".join(current_body).strip()))

    # Fallback for plain text documents: split by paragraph groups.
    if len(sections) == 1 and not sections[0].title:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sections[0].body) if p.strip()]
        if len(paragraphs) > 1:
            return [Section(title=f"Section {i + 1}", body=p) for i, p in enumerate(paragraphs)]

    return sections


def _tail_overlap(tokenizer: Tokenizer, text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    tokens = tokenizer.encode(text)
    if not tokens:
        return ""
    return tokenizer.decode(tokens[-overlap_tokens:]).strip()


def _split_oversized_text(
    tokenizer: Tokenizer,
    text: str,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> list[str]:
    tokens = tokenizer.encode(text)
    if len(tokens) <= chunk_token_size:
        return [text]

    window = max(1, chunk_token_size - chunk_overlap_token_size)
    parts: list[str] = []
    for start in range(0, len(tokens), window):
        piece = tokenizer.decode(tokens[start : start + chunk_token_size]).strip()
        if piece:
            parts.append(piece)
    return parts


def structure_based_chunking(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """
    Chunk text by document structure first, then enforce token size limits.

    Compatible with LightRAG's `chunking_func` signature.
    """
    if split_by_character:
        # Respect caller override if explicitly requested.
        units = [u.strip() for u in content.split(split_by_character) if u.strip()]
        if split_by_character_only:
            sections = [Section(title="", body=u) for u in units]
        else:
            sections = []
            for u in units:
                sections.extend(_split_into_sections(u))
    else:
        sections = _split_into_sections(content)

    chunks: list[dict[str, Any]] = []
    current_text = ""
    current_tokens = 0
    chunk_index = 0

    def flush_current() -> None:
        nonlocal current_text, current_tokens, chunk_index
        text = current_text.strip()
        if not text:
            current_text = ""
            current_tokens = 0
            return
        chunks.append(
            {
                "tokens": current_tokens,
                "content": text,
                "chunk_order_index": chunk_index,
            }
        )
        chunk_index += 1
        current_text = ""
        current_tokens = 0

    for sec in sections:
        section_text = f"{sec.title}\n{sec.body}".strip() if sec.title else sec.body.strip()
        if not section_text:
            continue

        section_parts = _split_oversized_text(
            tokenizer,
            section_text,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
        )

        for part in section_parts:
            part_tokens = len(tokenizer.encode(part))

            if part_tokens > chunk_token_size:
                # Defensive fallback. Should be rare due to _split_oversized_text.
                flush_current()
                forced_parts = _split_oversized_text(
                    tokenizer,
                    part,
                    chunk_token_size=chunk_token_size,
                    chunk_overlap_token_size=chunk_overlap_token_size,
                )
                for forced in forced_parts:
                    forced_tokens = len(tokenizer.encode(forced))
                    chunks.append(
                        {
                            "tokens": forced_tokens,
                            "content": forced,
                            "chunk_order_index": chunk_index,
                        }
                    )
                    chunk_index += 1
                continue

            if current_tokens + part_tokens <= chunk_token_size:
                current_text = (current_text + "\n\n" + part).strip() if current_text else part
                current_tokens = len(tokenizer.encode(current_text))
                continue

            # Finalize current chunk and start a new one with overlap context.
            overlap_text = _tail_overlap(tokenizer, current_text, chunk_overlap_token_size)
            flush_current()
            current_text = (overlap_text + "\n\n" + part).strip() if overlap_text else part
            current_tokens = len(tokenizer.encode(current_text))

    flush_current()

    return chunks


def build_rag_with_structure_chunking(**kwargs: Any) -> LightRAG:
    """Convenience factory for quick experiments."""
    return LightRAG(chunking_func=structure_based_chunking, **kwargs)


if __name__ == "__main__":
    sample = """# Introduction
LightRAG supports multiple insertion pipelines.

## Problem
Token-only chunking may break section boundaries.

## Approach
Use heading-aware chunking first, then token safety split.

## Result
Better semantic coherence per chunk.
"""

    from lightrag.utils import TiktokenTokenizer

    tokenizer = TiktokenTokenizer("gpt-4o-mini")
    demo_chunks = structure_based_chunking(tokenizer, sample, chunk_token_size=80, chunk_overlap_token_size=10)
    for item in demo_chunks:
        print(f"[{item['chunk_order_index']}] tokens={item['tokens']}\n{item['content']}\n")
