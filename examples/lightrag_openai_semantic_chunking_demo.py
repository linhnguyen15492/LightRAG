import os
import asyncio
import logging
import logging.config
import json
from pathlib import Path

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.semantic_chunking import chunking_by_semantic_token_size
from lightrag.utils import logger, set_verbose_debug

load_dotenv(dotenv_path='.env', override=False)

WORKING_DIR = './rag_storage_semantic_chunking'
INPUT_DIR = './lightrag/evaluation/sample_documents'
EVAL_DATASET_FILE = './lightrag/evaluation/sample_dataset.json'


def configure_logging() -> None:
    """Configure logging for the semantic chunking demo."""

    for logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error', 'lightrag']:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir = os.getenv('LOG_DIR', os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, 'lightrag_semantic_demo.log'))

    print(f"\nLightRAG semantic demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv('LOG_MAX_BYTES', 10485760))
    log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', 5))

    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {'format': '%(levelname)s: %(message)s'},
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
            },
            'handlers': {
                'console': {
                    'formatter': 'default',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stderr',
                },
                'file': {
                    'formatter': 'detailed',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': log_file_path,
                    'maxBytes': log_max_bytes,
                    'backupCount': log_backup_count,
                    'encoding': 'utf-8',
                },
            },
            'loggers': {
                'lightrag': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': False,
                }
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true')


async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        chunking_func=chunking_by_semantic_token_size,
    )

    await rag.initialize_storages()
    return rag


async def main() -> None:
    if not os.getenv('OPENAI_API_KEY'):
        print('Error: OPENAI_API_KEY environment variable is not set.')
        print("Run: export OPENAI_API_KEY='your-openai-api-key'")
        return

    input_dir = Path(INPUT_DIR)
    eval_dataset_file = Path(EVAL_DATASET_FILE)

    if not input_dir.exists():
        print(f'Error: input directory not found: {input_dir}')
        return

    if not eval_dataset_file.exists():
        print(f'Error: eval dataset not found: {eval_dataset_file}')
        return

    supported_suffixes = {'.md', '.markdown', '.txt'}
    input_files = [
        path
        for path in sorted(input_dir.rglob('*'))
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    if not input_files:
        print(f'Error: no supported files found in {input_dir}')
        return

    os.makedirs(WORKING_DIR, exist_ok=True)

    rag = None
    try:
        files_to_delete = [
            'graph_chunk_entity_relation.graphml',
            'kv_store_doc_status.json',
            'kv_store_full_docs.json',
            'kv_store_text_chunks.json',
            'vdb_chunks.json',
            'vdb_entities.json',
            'vdb_relationships.json',
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f'Deleting old file:: {file_path}')

        rag = await initialize_rag()
        if rag is None:
            print('Error: failed to initialize LightRAG')
            return
        if rag.embedding_func is None:
            print('Error: embedding function is not configured')
            return

        test_text = ['This is a test string for embedding.']
        embedding = await rag.embedding_func(test_text)
        print('\n=======================')
        print('Semantic Chunking Demo')
        print('=======================')
        print(f'Test text: {test_text}')
        print(f'Detected embedding dimension: {embedding.shape[1]}')

        print(f'\nIndexing {len(input_files)} files from {input_dir} ...')
        for file_path in input_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            if not content:
                continue
            await rag.ainsert(content, file_paths=[str(file_path)])
            print(f'Indexed: {file_path.name}')

        with open(eval_dataset_file, 'r', encoding='utf-8') as file:
            eval_data = json.load(file)

        test_cases = eval_data.get('test_cases', [])
        if not test_cases:
            print(f'Error: no test_cases found in {eval_dataset_file}')
            return

        print('\n=====================')
        print('Evaluation on Indexed Sample Docs (mode=mix)')
        print('=====================')

        for idx, case in enumerate(test_cases, start=1):
            question = case.get('question', '').strip()
            ground_truth = case.get('ground_truth', '').strip()
            if not question:
                continue

            query_param = QueryParam(
                mode='mix',
                top_k=int(os.getenv('EVAL_QUERY_TOP_K', '10')),
                response_type='Multiple Paragraphs',
            )

            answer = await rag.aquery(question, param=query_param)
            retrieved_data = await rag.aquery_data(question, param=query_param)
            contexts = retrieved_data.get('data', {}).get('chunks', [])

            print(f'\n[{idx}] Question: {question}')
            print(f'Ground Truth: {ground_truth}')
            print(f'Retrieved Chunks: {len(contexts)}')
            print('Answer:')
            print(answer)

    except Exception as error:
        print(f'An error occurred: {error}')
    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == '__main__':
    configure_logging()
    asyncio.run(main())
    print('\nDone!')
