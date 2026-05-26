import asyncio
import os

from dotenv import load_dotenv

from lightrag import QueryParam
from lightrag.document_chunking import chunking_by_document
from lightrag.lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./rag_storage_document_chunking"
INPUT_FILE = "./book.txt"


async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        chunking_func=chunking_by_document,
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
