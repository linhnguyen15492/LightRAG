from __future__ import annotations

from typing import Any, Sequence

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from lightrag.exceptions import ChunkTokenLimitExceededError
from lightrag.utils import Tokenizer, logger


DEFAULT_MARKDOWN_HEADERS: list[tuple[str, str]] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def chunking_by_document(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    use_markdown_structure: bool = True,
    markdown_headers_to_split_on: Sequence[tuple[str, str]] | None = None,
    strict_token_limit: bool = False,
) -> list[dict[str, Any]]:
    """Split a document into token-bounded chunks using LangChain splitters.

    Strategy:
    1. Optionally split markdown by headers into semantic sections.
    2. Split each section by token-aware recursive splitter.

    Returns chunk dictionaries compatible with LightRAG chunk schema.
    The first six parameters preserve LightRAG's chunking_func signature.
    """

    if chunk_token_size <= 0:
        raise ValueError("chunk_token_size must be > 0")
    if chunk_overlap_token_size < 0:
        raise ValueError("chunk_overlap_token_size must be >= 0")
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError(
            "chunk_overlap_token_size must be smaller than chunk_token_size"
        )

    if not content:
        return []

    if split_by_character:
        raw_chunks = content.split(split_by_character)
    else:
        raw_chunks = [content]

    if split_by_character and split_by_character_only:
        for chunk in raw_chunks:
            token_count = len(tokenizer.encode(chunk))
            if token_count > chunk_token_size:
                logger.warning(
                    "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                    token_count,
                    chunk_token_size,
                )
                raise ChunkTokenLimitExceededError(
                    chunk_tokens=token_count,
                    chunk_token_limit=chunk_token_size,
                    chunk_preview=chunk[:120],
                )

    def _token_len(text: str) -> int:
        return len(tokenizer.encode(text))

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_token_size,
        chunk_overlap=chunk_overlap_token_size,
        length_function=_token_len,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
    )

    split_docs = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if not raw_chunk:
            continue

        should_use_markdown = use_markdown_structure and "#" in raw_chunk

        if should_use_markdown:
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=list(
                    markdown_headers_to_split_on or DEFAULT_MARKDOWN_HEADERS
                ),
                strip_headers=False,
            )
            structured_docs = header_splitter.split_text(raw_chunk)
            split_docs.extend(
                recursive_splitter.split_documents(structured_docs)
                if structured_docs
                else recursive_splitter.create_documents([raw_chunk])
            )
        else:
            split_docs.extend(recursive_splitter.create_documents([raw_chunk]))

    results: list[dict[str, Any]] = []
    for index, doc in enumerate(split_docs):
        chunk_text = doc.page_content.strip()
        if not chunk_text:
            continue

        token_count = _token_len(chunk_text)
        if token_count > chunk_token_size:
            message = (
                "Document chunk exceeds token limit after splitting: "
                f"len={token_count} limit={chunk_token_size}"
            )
            logger.warning(message)
            if strict_token_limit:
                raise ChunkTokenLimitExceededError(
                    chunk_tokens=token_count,
                    chunk_token_limit=chunk_token_size,
                    chunk_preview=chunk_text[:120],
                )

        results.append(
            {
                "tokens": token_count,
                "content": chunk_text,
                "chunk_order_index": index,
                "metadata": doc.metadata or {},
            }
        )

    return results


__all__ = ["chunking_by_document", "DEFAULT_MARKDOWN_HEADERS"]
