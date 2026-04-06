import asyncio
import os
from typing import Any, TYPE_CHECKING

from dotenv import load_dotenv

from lightrag import QueryParam
from lightrag.document_chunking import chunking_by_document
from lightrag.lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

if TYPE_CHECKING:
    from lightrag.utils import Tokenizer

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./rag_storage_document_chunking"
INPUT_FILE = "./book.txt"


def document_based_chunking_adapter(
    tokenizer: "Tokenizer",
    content: str,
    split_by_character: str | None,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict[str, Any]]:
    """Adapter for LightRAG chunking_func signature."""
    _ = split_by_character
    _ = split_by_character_only
    return chunking_by_document(
        tokenizer=tokenizer,
        content=content,
        chunk_overlap_token_size=chunk_overlap_token_size,
        chunk_token_size=chunk_token_size,
        use_markdown_structure=True,
        strict_token_limit=False,
    )


async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        chunking_func=document_based_chunking_adapter,
    )
    await rag.initialize_storages()
    return rag


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set")
        print("Run: export OPENAI_API_KEY='your-openai-api-key'")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"Error: input file not found: {INPUT_FILE}")
        return

    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = None
    try:
        rag = await initialize_rag()

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        query = "What are the top themes?"

        print("\nNaive Search:")
        print(await rag.aquery(query, param=QueryParam(mode="naive")))

        print("\nLocal Search:")
        print(await rag.aquery(query, param=QueryParam(mode="local")))

        print("\nGlobal Search:")
        print(await rag.aquery(query, param=QueryParam(mode="global")))

        print("\nHybrid Search:")
        print(await rag.aquery(query, param=QueryParam(mode="hybrid")))

    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
