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

    Returns a list of chunks compatible with the existing chunk schema used by
    ``chunking_by_token_size`` and includes document metadata when available.
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

    def _token_len(text: str) -> int:
        return len(tokenizer.encode(text))

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_token_size,
        chunk_overlap=chunk_overlap_token_size,
        length_function=_token_len,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
    )

    should_use_markdown = use_markdown_structure and "#" in content

    if should_use_markdown:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(
                markdown_headers_to_split_on or DEFAULT_MARKDOWN_HEADERS
            ),
            strip_headers=False,
        )
        structured_docs = header_splitter.split_text(content)
        split_docs = (
            recursive_splitter.split_documents(structured_docs)
            if structured_docs
            else recursive_splitter.create_documents([content])
        )
    else:
        split_docs = recursive_splitter.create_documents([content])

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
