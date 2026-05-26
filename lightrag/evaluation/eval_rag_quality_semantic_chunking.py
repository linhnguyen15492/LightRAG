#!/usr/bin/env python3
"""
RAGAS Evaluation Script for LightRAG using Semantic Chunking (local pipeline).

This script keeps the evaluation structure from eval_rag_quality.py, but replaces
API-based querying with a local LightRAG instance configured to use
chunking_by_semantic_token_size.

Usage:
    python lightrag/evaluation/eval_rag_quality_semantic_chunking.py
    python lightrag/evaluation/eval_rag_quality_semantic_chunking.py -d my_test.json
    # Reuse existing indexed data (default behavior)
    python lightrag/evaluation/eval_rag_quality_semantic_chunking.py --working-dir ./rag_storage_semantic_chunking
    # Reindex from sample docs when needed
    python lightrag/evaluation/eval_rag_quality_semantic_chunking.py --reindex
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightrag import LightRAG, QueryParam
from lightrag.evaluation.eval_rag_quality import RAGEvaluator
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.semantic_chunking import chunking_by_semantic_token_size
from lightrag.utils import logger

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path='.env', override=False)


class SemanticChunkingRAGEvaluator(RAGEvaluator):
    """RAGAS evaluator that uses local LightRAG + semantic chunking."""

    def __init__(
        self,
        test_dataset_path: str | None = None,
        input_dir: str | None = None,
        working_dir: str | None = None,
        clean_working_dir: bool = False,
        reindex_from_input: bool = False,
    ):
        self.input_dir = Path(input_dir) if input_dir else Path(__file__).parent / 'sample_documents'
        self.clean_working_dir = clean_working_dir
        self.reindex_from_input = reindex_from_input

        if working_dir:
            self.working_dir = Path(working_dir)
        else:
            self.working_dir = Path(__file__).parent / 'rag_storage_semantic_eval'

        self.rag: LightRAG | None = None

        super().__init__(
            test_dataset_path=test_dataset_path,
            rag_api_url='local-semantic-chunking',
        )

        # Keep semantic evaluation results separate from API-based evaluation outputs.
        self.results_dir = Path(__file__).parent / 'results_semantic'
        self.results_dir.mkdir(exist_ok=True)

    def _display_configuration(self):
        """Display semantic-eval specific configuration."""
        logger.info('Evaluation Models:')
        logger.info('  • LLM Model:            %s', self.eval_model)
        logger.info('  • Embedding Model:      %s', self.eval_embedding_model)
        logger.info('Semantic Chunking Configuration:')
        logger.info('  • Chunking Method:      chunking_by_semantic_token_size')
        logger.info('  • Input Directory:      %s', self.input_dir)
        logger.info('  • Working Directory:    %s', self.working_dir)
        logger.info('  • Reindex from Input:   %s', self.reindex_from_input)

        logger.info('Concurrency & Rate Limiting:')
        query_top_k = int(os.getenv('EVAL_QUERY_TOP_K', '10'))
        logger.info('  • Query Top-K:          %s Entities/Relations', query_top_k)
        logger.info('  • LLM Max Retries:      %s', self.eval_max_retries)
        logger.info('  • LLM Timeout:          %s seconds', self.eval_timeout)

        logger.info('Test Configuration:')
        logger.info('  • Total Test Cases:     %s', len(self.test_cases))
        logger.info('  • Test Dataset:         %s', self.test_dataset_path.name)
        logger.info('  • Results Directory:    %s', self.results_dir.name)

    def _collect_input_files(self) -> list[Path]:
        """Collect source files to index for local evaluation."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f'Input directory not found: {self.input_dir}')

        allowed_suffixes = {'.txt', '.md', '.markdown'}
        files = [
            path
            for path in sorted(self.input_dir.rglob('*'))
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        ]

        if not files:
            raise FileNotFoundError(
                f'No supported documents found in {self.input_dir}. '
                'Expected at least one .txt/.md/.markdown file.'
            )

        return files

    async def _initialize_local_rag(self) -> None:
        """Create local LightRAG instance and optionally index evaluation documents."""
        if self.clean_working_dir and self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
            chunking_func=chunking_by_semantic_token_size,
        )
        await self.rag.initialize_storages()

        if not self.reindex_from_input:
            # Reuse previously indexed data in working_dir by default.
            logger.info(
                'Using existing indexed data from working directory: %s',
                self.working_dir,
            )
            logger.info(
                'Skip re-indexing. Use --reindex to rebuild index from %s',
                self.input_dir,
            )
            return

        files = self._collect_input_files()
        logger.info('Indexing %s files into local LightRAG...', len(files))

        for path in files:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if content.strip():
                await self.rag.ainsert(content, file_paths=[str(path)])

        logger.info('Indexing complete.')

    async def _finalize_local_rag(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()
            self.rag = None

    async def generate_rag_response(
        self,
        question: str,
        client: Any = None,
    ) -> Dict[str, Any]:
        """Generate answer and retrieved contexts from local semantic-chunked LightRAG."""
        _ = client

        if self.rag is None:
            raise RuntimeError('LightRAG is not initialized. Call _initialize_local_rag first.')

        query_top_k = int(os.getenv('EVAL_QUERY_TOP_K', '10'))
        query_mode = os.getenv('EVAL_QUERY_MODE', 'mix')

        query_param = QueryParam(
            mode=query_mode,
            top_k=query_top_k,
            response_type='Multiple Paragraphs',
        )

        answer = await self.rag.aquery(question, param=query_param)

        data_response = await self.rag.aquery_data(
            question,
            param=QueryParam(
                mode=query_mode,
                top_k=query_top_k,
                response_type='Multiple Paragraphs',
            ),
        )

        chunks = data_response.get('data', {}).get('chunks', [])
        contexts = [
            chunk.get('content', '').strip()
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get('content')
        ]

        if isinstance(answer, str):
            answer_text = answer
        else:
            answer_text = ''.join([part async for part in answer])

        return {
            'answer': answer_text or 'No response generated',
            'contexts': contexts,
        }

    async def run(self) -> Dict[str, Any]:
        """Initialize local semantic RAG, run inherited evaluation, then clean up."""
        await self._initialize_local_rag()
        try:
            return await super().run()
        finally:
            await self._finalize_local_rag()


async def main() -> None:
    """Main entry point for semantic chunking RAGAS evaluation."""
    try:
        parser = argparse.ArgumentParser(
            description='RAGAS Evaluation Script for LightRAG (Semantic Chunking)',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Examples:
    # Use existing indexed data in default working dir
  python lightrag/evaluation/eval_rag_quality_semantic_chunking.py

    # Use existing indexed data from a specific working dir
    python lightrag/evaluation/eval_rag_quality_semantic_chunking.py --working-dir ./rag_storage_semantic_chunking

  # Specify custom dataset
  python lightrag/evaluation/eval_rag_quality_semantic_chunking.py --dataset my_test.json

    # Reindex from input docs directory
    python lightrag/evaluation/eval_rag_quality_semantic_chunking.py --reindex --input-dir ./my_docs

    # Force clean old index before reindexing
    python lightrag/evaluation/eval_rag_quality_semantic_chunking.py --reindex --force-reindex

    # Reindex from custom docs without deleting existing working dir first
    python lightrag/evaluation/eval_rag_quality_semantic_chunking.py --reindex --input-dir ./my_docs
            ''',
        )

        parser.add_argument(
            '--dataset',
            '-d',
            type=str,
            default=None,
            help='Path to test dataset JSON file (default: sample_dataset.json in evaluation directory)',
        )

        parser.add_argument(
            '--input-dir',
            type=str,
            default=None,
            help='Directory of source documents for indexing (used when --reindex is enabled)',
        )

        parser.add_argument(
            '--working-dir',
            type=str,
            default=None,
            help='Working directory for semantic evaluation storage',
        )

        parser.add_argument(
            '--reindex',
            action='store_true',
            help='Rebuild index from input documents before evaluation',
        )

        parser.add_argument(
            '--force-reindex',
            action='store_true',
            help='When reindexing, remove old working-dir index files first',
        )

        args = parser.parse_args()

        logger.info('%s', '=' * 70)
        logger.info('🔍 RAGAS Evaluation - Local LightRAG with Semantic Chunking')
        logger.info('%s', '=' * 70)

        evaluator = SemanticChunkingRAGEvaluator(
            test_dataset_path=args.dataset,
            input_dir=args.input_dir,
            working_dir=args.working_dir,
            clean_working_dir=args.force_reindex,
            reindex_from_input=args.reindex,
        )
        await evaluator.run()

    except Exception as e:
        logger.exception('❌ Error: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
