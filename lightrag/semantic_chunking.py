from __future__ import annotations

import os
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from lightrag.exceptions import ChunkTokenLimitExceededError
from lightrag.utils import Tokenizer, logger


def _validate_chunk_params(
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> None:
    if chunk_token_size <= 0:
        raise ValueError("chunk_token_size must be > 0")
    if chunk_overlap_token_size < 0:
        raise ValueError("chunk_overlap_token_size must be >= 0")
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError(
            "chunk_overlap_token_size must be smaller than chunk_token_size"
        )


def _resolve_langchain_embeddings(langchain_embeddings: Any | None) -> Any:
    """Resolve a LangChain Embeddings implementation for SemanticChunker.

    Priority:
    1. Explicit `langchain_embeddings` parameter.
    2. OpenAI embeddings from environment (if dependencies and API key are present).
    """
    if langchain_embeddings is not None:
        return langchain_embeddings

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return None

    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception:
        return None

    model_name = os.getenv("SEMANTIC_CHUNK_EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = os.getenv("EVAL_EMBEDDING_BINDING_HOST") or os.getenv(
        "EVAL_LLM_BINDING_HOST"
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": openai_api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIEmbeddings(**kwargs)


def _build_semantic_chunker(
    embeddings: Any,
    breakpoint_threshold_type: str,
    breakpoint_threshold_amount: float,
):
    """Build LangChain SemanticChunker lazily to avoid hard dependency errors."""
    try:
        from langchain_experimental.text_splitter import (  # type: ignore[reportMissingImports]
            SemanticChunker,
        )
    except Exception as exc:
        raise ImportError(
            "Semantic chunking requires langchain-experimental. "
            "Install with: pip install langchain-experimental"
        ) from exc

    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )


def _split_oversized_text_by_tokens(
    tokenizer: Tokenizer,
    text: str,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> list[str]:
    """Split oversized text by token window while preserving overlap."""
    tokens = tokenizer.encode(text)
    step = chunk_token_size - chunk_overlap_token_size

    pieces: list[str] = []
    for start in range(0, len(tokens), step):
        piece = tokenizer.decode(tokens[start : start + chunk_token_size]).strip()
        if piece:
            pieces.append(piece)
    return pieces


def _merge_small_chunks(
    tokenizer: Tokenizer,
    chunk_texts: list[str],
    min_chunk_token_size: int,
    chunk_token_size: int,
) -> list[str]:
    """Merge neighboring tiny chunks produced by semantic boundaries."""
    if min_chunk_token_size <= 0:
        return chunk_texts

    merged: list[str] = []
    buffer_text = ""
    buffer_tokens = 0

    for text in chunk_texts:
        text_tokens = len(tokenizer.encode(text))

        if not buffer_text:
            buffer_text = text
            buffer_tokens = text_tokens
            continue

        if (
            buffer_tokens < min_chunk_token_size
            and buffer_tokens + text_tokens <= chunk_token_size
        ):
            buffer_text = f"{buffer_text} {text}".strip()
            buffer_tokens += text_tokens
            continue

        merged.append(buffer_text)
        buffer_text = text
        buffer_tokens = text_tokens

    if buffer_text:
        merged.append(buffer_text)

    return merged


def chunking_by_semantic_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    semantic_break_threshold: float = 0.15,
    min_chunk_token_size: int = 300,
    langchain_embeddings: Any | None = None,
    semantic_breakpoint_threshold_type: str = "percentile",
    semantic_breakpoint_threshold_amount: float | None = None,
) -> list[dict[str, Any]]:
    """LangChain-based semantic chunking compatible with LightRAG chunk schema.

    Returns chunk objects with keys: tokens, content, chunk_order_index.
    The first six parameters preserve LightRAG's chunking_func signature.
    """
    _validate_chunk_params(chunk_overlap_token_size, chunk_token_size)

    if not 0 <= semantic_break_threshold <= 1:
        raise ValueError("semantic_break_threshold must be in [0, 1]")
    if min_chunk_token_size < 0:
        raise ValueError("min_chunk_token_size must be >= 0")
    if min_chunk_token_size > chunk_token_size:
        raise ValueError("min_chunk_token_size must be <= chunk_token_size")

    # Backward-compatible mapping from previous threshold semantics.
    # A lower semantic_break_threshold means stricter splitting;
    # map it to higher percentile breakpoints when explicit amount is not provided.
    if semantic_breakpoint_threshold_amount is None:
        semantic_breakpoint_threshold_amount = max(
            1.0,
            min(99.0, (1.0 - semantic_break_threshold) * 100.0),
        )

    results: list[dict[str, Any]] = []
    if not content:
        return results

    if split_by_character:
        raw_chunks = content.split(split_by_character)
    else:
        raw_chunks = [content]

    # Match legacy behavior: strict split mode raises if any split piece is oversized.
    if split_by_character and split_by_character_only:
        for index, chunk in enumerate(raw_chunks):
            _tokens = tokenizer.encode(chunk)
            if len(_tokens) > chunk_token_size:
                logger.warning(
                    "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                    len(_tokens),
                    chunk_token_size,
                )
                raise ChunkTokenLimitExceededError(
                    chunk_tokens=len(_tokens),
                    chunk_token_limit=chunk_token_size,
                    chunk_preview=chunk[:120],
                )
            results.append(
                {
                    "tokens": len(_tokens),
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
        return results

    embeddings = _resolve_langchain_embeddings(langchain_embeddings)
    if embeddings is None:
        logger.warning(
            "No LangChain embeddings available for semantic chunking. "
            "Set OPENAI_API_KEY and install langchain-openai, or pass langchain_embeddings explicitly. "
            "Falling back to token-aware recursive splitting."
        )
        semantic_chunker = None
    else:
        semantic_chunker = _build_semantic_chunker(
            embeddings=embeddings,
            breakpoint_threshold_type=semantic_breakpoint_threshold_type,
            breakpoint_threshold_amount=semantic_breakpoint_threshold_amount,
        )

    token_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_token_size,
        chunk_overlap=chunk_overlap_token_size,
        length_function=lambda text: len(tokenizer.encode(text)),
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
    )

    candidates: list[str] = []
    for raw_chunk in raw_chunks:
        raw_chunk = raw_chunk.strip()
        if not raw_chunk:
            continue

        semantic_candidates: list[str] = []
        if semantic_chunker is not None:
            try:
                docs = semantic_chunker.create_documents([raw_chunk])
                semantic_candidates = [
                    doc.page_content.strip()
                    for doc in docs
                    if getattr(doc, "page_content", "").strip()
                ]
            except Exception as exc:
                logger.warning(
                    "SemanticChunker failed on a document, fallback to token splitter: %s",
                    exc,
                )

        if not semantic_candidates:
            semantic_candidates = [raw_chunk]

        for semantic_text in semantic_candidates:
            if len(tokenizer.encode(semantic_text)) > chunk_token_size:
                split_docs = token_splitter.create_documents([semantic_text])
                for split_doc in split_docs:
                    split_text = split_doc.page_content.strip()
                    if split_text:
                        candidates.append(split_text)
                continue

            candidates.append(semantic_text)

    candidates = _merge_small_chunks(
        tokenizer=tokenizer,
        chunk_texts=candidates,
        min_chunk_token_size=min_chunk_token_size,
        chunk_token_size=chunk_token_size,
    )

    for index, chunk_text in enumerate(candidates):
        if not chunk_text:
            continue
        token_count = len(tokenizer.encode(chunk_text))
        if token_count > chunk_token_size:
            logger.warning(
                "Semantic chunk exceeds token limit after split: len=%d limit=%d",
                token_count,
                chunk_token_size,
            )
            # Force final safety split to preserve contract.
            fallback_pieces = _split_oversized_text_by_tokens(
                tokenizer=tokenizer,
                text=chunk_text,
                chunk_token_size=chunk_token_size,
                chunk_overlap_token_size=chunk_overlap_token_size,
            )
            for piece in fallback_pieces:
                piece_tokens = len(tokenizer.encode(piece))
                results.append(
                    {
                        "tokens": piece_tokens,
                        "content": piece,
                        "chunk_order_index": len(results),
                    }
                )
            continue

        results.append(
            {
                "tokens": token_count,
                "content": chunk_text,
                "chunk_order_index": len(results),
            }
        )

    return results


__all__ = ["chunking_by_semantic_token_size"]
