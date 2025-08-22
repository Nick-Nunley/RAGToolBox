"""
RAGToolBox Index module.

Provides the Indexer class for embedding and storing (indexing) chunked strings.

Additionally, this script provides a CLI entry point for execution as a standalone python module.
"""

import argparse
import os
import logging
import sys
import subprocess
import re
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
from RAGToolBox.embeddings import Embeddings
from RAGToolBox.chunk import Chunker, HierarchicalChunker, SectionAwareChunker, SlidingWindowChunker
from RAGToolBox.vector_store import VectorStoreFactory
from RAGToolBox.logging import RAGTBLogger

__all__ = ['IndexerConfig', 'ParallelConfig', 'Indexer']
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class IndexerConfig:
    """
    Holds all the optional config settings for Indexer class.

    Attributes:
        vector_store_backend: Identifier for the vector store backend.
            Currently supported: ``"sqlite"``, ``"chroma"``
        vector_store_config: Backend-specific options. For example, for
            SQLite: ``{"db_path": Path(".../embeddings.db")}``; for Chroma:
            ``{"collection_name": "...", "persist_directory": "...", "chroma_client_url": "..."}``
        output_dir: Directory where embedding artifacts (e.g., SQLite DB) are written
    """
    vector_store_backend: str = "sqlite"
    vector_store_config: Optional[dict] = None
    output_dir: Path = Path("assets/kb/embeddings/")

@dataclass(frozen=True)
class ParallelConfig:
    """
    Holds all config settings for parallel processing.

    Attributes:
        parallel_embed: If ``True``, embed in parallel using a process pool
        num_workers: Number of **processes** to spawn when ``parallel_embed`` is enabled
            this uses :class:`concurrent.futures.ProcessPoolExecutor`
    """
    parallel_embed: bool = False
    num_workers: int = 3


class Indexer:
    """
    Indexer class for loading (optional), chunking, and embedding content.

    Attributes:
        chunker: Chunker used to split documents into chunks
        embedding_model: Name of the embedding backend; must be one of
            :py:meth:`RAGToolBox.embeddings.Embeddings.supported_models`
        output_dir: Directory where embedding artifacts are written
        vector_store: Initialized vector store backend instance

    Note:
        The effective vector store configuration is derived from ``config``
        For the SQLite backend, ``db_path`` defaults to ``output_dir / "embeddings.db"``
    """

    def __init__(
        self, chunker: Chunker, embedding_model: str, config: Optional[IndexerConfig] = None
        ):
        """
        Initializes an Indexer object.

        Args:
            chunker: Chunker to use for indexing
            embedding_model: Name of the embedding backend. See
                :py:meth:`Embeddings.supported_models` for the current list
            config: IndexerConfig to provide indexing paramters such as
                the vector store backend, backend-specific configuration, and the output directory.
        """
        self.chunker = chunker
        Embeddings.validate_embedding_model(embedding_model)
        self.embedding_model = embedding_model
        if config is None:
            logger.info("No Indexer config supplied. Using default 'IndexerConfig'")
            config = IndexerConfig()
        self.output_dir = config.output_dir

        # Initialize vector store backend
        self.vector_store_config = config.vector_store_config or {}
        if config.vector_store_backend == 'sqlite':
            # For SQLite, use the output_dir to determine db_path
            db_path = self.output_dir / 'embeddings.db'
            self.vector_store_config['db_path'] = db_path

        self.vector_store = VectorStoreFactory.create_backend(
            config.vector_store_backend,
            **self.vector_store_config
            )
        self.vector_store.initialize()

    def pre_chunk(self, text: str) -> dict:
        """
        Parse simple front-matter and references from Markdown.

        - Treat a top block before the first line equal to ``---`` as front-matter
        (lines formatted as ``key: value``).
        - If a ``## References`` section exists, include its text in metadata
        under ``"references"`` (the main text is left intact).

        Args:
            text: Raw Markdown document

        Returns:
            A dict with:
                - ``"metadata"``: ``dict[str, str]`` key/value pairs from front-matter and
                optional ``"references"``
                - ``"text"``: ``str`` main body (without front-matter)
        """
        # Split at the first '---' line
        parts = re.split(r'^---$', text, maxsplit=1, flags=re.MULTILINE)
        if len(parts) == 2:
            meta_block, main_text = parts
            metadata = {}
            for line in meta_block.strip().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            try:
                refs_match = re.search(
                    r'^## References\s*\n(.+)',
                    main_text,
                    flags=re.MULTILINE | re.DOTALL
                    )
                if refs_match:
                    metadata['references'] = refs_match.group(1).strip()
            except Exception: # pylint: disable=broad-exception-caught
                pass
            return {'metadata': metadata, 'text': main_text.strip()}

        metadata = {}
        main_text = text.strip()
        try:
            refs_match = re.search(
                r'^## References\s*\n(.+)',
                main_text,
                flags=re.MULTILINE | re.DOTALL
                )
            if refs_match:
                metadata['references'] = refs_match.group(1).strip()
        except Exception: # pylint: disable=broad-exception-caught
            pass
        return {'metadata': metadata, 'text': main_text}

    def chunk(self, doc_args: Tuple[str, str]) -> Tuple[str, dict, List[str]]:
        """
        Chunk a document after extracting metadata.

        Args:
            doc_args: ``(name, text)`` tuple

        Returns:
            ``(name, metadata, chunks)`` where:
                - ``name`` is the document name,
                - ``metadata`` is ``dict[str, str]``,
                - ``chunks`` is ``List[str]``
        """
        name, text = doc_args
        parsed = self.pre_chunk(text)
        metadata = parsed['metadata']
        main_text = parsed['text']
        return (name, metadata, self.chunker.chunk(main_text))

    def embed(self, chunks: List[str], max_retries: int = 5) -> List[List[float]]:
        """
        Embed a list of chunks using the configured embedding backend.

        Retries with exponential backoff on rate-limit errors for remote backends.

        Args:
            chunks: Chunk texts to embed
            max_retries: Max retry attempts (only relevant for remote backends)

        Returns:
            List of embedding vectors (aligned to ``chunks``)

        Raises:
            ImportError: If the selected backend is not installed
            RuntimeError: If the backend call fails after retries
            ValueError: If the configured backend is unsupported
        """
        return Embeddings.embed_texts(self.embedding_model, chunks, max_retries)

    def _insert_embeddings_to_db(self, chunked_results: list, embeddings: list) -> None:
        """
        Insert chunk, embedding, and metadata into the vector store backend.
        """
        self.vector_store.insert_embeddings(chunked_results, embeddings)

    def _embed_and_save_in_batch(self, batch: List[str], batch_entries: List[dict]) -> None:
        """
        Embed a batch of chunks and insert their embeddings into the SQLite database.
        """
        embeddings = self.embed(batch)
        self._insert_embeddings_to_db(batch_entries, embeddings)

    def index(
        self, chunked_results: list[dict], parallel_config: Optional[ParallelConfig] = None
        ) -> None:
        """
        Embed the provided chunks and store results in the configured vector store.

        Args:
            chunked_results: List of entries with at least:
                - ``"chunk"``: ``str`` chunk text,
                - ``"metadata"``: ``dict[str, str]`` (optional keys allowed)
            parallel_config: Controls process-based parallel embedding

        Raises:
            RuntimeError: If embedding or persistence fails

        Note:
            Parallel mode uses a **process** pool; speedups depend on the backend:
            CPU-bound local embedding can scale, while network-bound remote calls
            may benefit more from batching than from additional processes.
        """
        if parallel_config is None:
            parallel_config = ParallelConfig()
        chunk_texts = [entry['chunk'] for entry in chunked_results]
        if not chunk_texts:
            logger.warning("No chunks to embed.")
            return
        if parallel_config.parallel_embed:
            # heuristic: 2 batches per worker
            batch_size = max(1, len(chunk_texts) // (parallel_config.num_workers * 2))
            batches = [chunk_texts[i:i+batch_size] for i in range(0, len(chunk_texts), batch_size)]
            from concurrent.futures import ProcessPoolExecutor, as_completed
            logger.debug(
                "Embedding %d chunks "
                "using %d workers...",
                len(chunk_texts), parallel_config.num_workers
                )
            with ProcessPoolExecutor(max_workers=parallel_config.num_workers) as executor:
                futures = []
                for i, batch in enumerate(batches):
                    batch_entries = chunked_results[i*batch_size:(i+1)*batch_size]
                    futures.append(executor.submit(
                        self._embed_and_save_in_batch,
                        batch,
                        batch_entries
                        ))
                for future in as_completed(futures):
                    future.result()
        else:
            embeddings = self.embed(chunk_texts)
            self._insert_embeddings_to_db(chunked_results, embeddings)
        logger.info("Indexing complete.")

    def _optional_loading_and_kb_init(self, cli_args: argparse.Namespace) -> Path:
        """Helper method for handling optional loading and initializing kb path in main"""
        if getattr(cli_args, 'command', None) == 'load':
            cmd = [
                sys.executable, "-m", "RAGToolBox.loader",
                *cli_args.urls,
                "--output-dir", cli_args.output_dir,
                ]
            if getattr(cli_args, "email", None):
                cmd += ["--email", cli_args.email]
            if getattr(cli_args, "use_readability", False):
                cmd += ["--use-readability"]
            # Pass through logging flags to child process
            if hasattr(cli_args, "log_level") and cli_args.log_level:
                cmd += ["--log-level", cli_args.log_level]
            if hasattr(cli_args, "log_file") and cli_args.log_file:
                cmd += ["--log-file", cli_args.log_file]
                if hasattr(cli_args, "log_file_level") and cli_args.log_file_level:
                    cmd += ["--log-file-level", cli_args.log_file_level]

            subprocess.run(cmd, check=True)
            # After loading, continue to chunking and indexing
            return Path(cli_args.output_dir)
        return Path(getattr(cli_args, 'kb_dir', 'assets/kb'))

    def _concurrent_chunker(self, docs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """Helper method for implementing concurrent chunking"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.debug("Chunking %d documents using %d processes...", len(docs), os.cpu_count())
        chunked_results = []
        with ProcessPoolExecutor() as executor:
            future_to_name = {executor.submit(self.chunk, doc): doc[0] for doc in docs}
            for future in as_completed(future_to_name):
                name, metadata, chunks = future.result()
                for chunk in chunks:
                    chunked_results.append({
                        'name': name,
                        'metadata': metadata,
                        'chunk': chunk
                    })
                logger.debug("Chunked %s: %d chunks", name, len(chunks))
        return chunked_results

    def main(self, cli_args: argparse.Namespace) -> None:
        """
        CLI entrypoint for indexing a knowledge base directory.

        Reads ``*.txt`` files, chunks them, and embeds/persists to the chosen vector store.
        May spawn a subprocess to run the loader when the ``load`` subcommand is used.
        """
        kb_dir = self._optional_loading_and_kb_init(cli_args=cli_args)

        # 1. Gather all .txt files in the knowledge base directory
        txt_files = list(kb_dir.glob('*.txt'))
        if not txt_files:
            logger.warning("No .txt files found in %s. Skipping chunking and indexing.", kb_dir)
            return

        # 2. Read all documents
        docs = []
        for file in txt_files:
            with open(file, 'r', encoding='utf-8') as f:
                docs.append((file.name, f.read()))

        # 3. Concurrent chunking using ProcessPoolExecutor
        chunked_results = self._concurrent_chunker(docs=docs)

        # 4. Embedding and indexing (optionally parallelized)
        self.index(
            chunked_results,
            parallel_config=ParallelConfig(
                parallel_embed=getattr(cli_args, 'parallel_embed', False),
                num_workers=getattr(cli_args, 'num_workers', 3)
                )
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Indexing pipeline with optional loading"
    )
    parser.add_argument(
        '--kb-dir', '-k', default='assets/kb',
        help='Directory where knowledge base is stored'
    )
    parser.add_argument(
        '--embedding-model', '-e', default='fastembed',
        help='Embedding model to use'
    )
    parser.add_argument(
        '--parallel-embed', '-p', action='store_true',
        help='Enable parallel embedding using multiple workers'
    )
    parser.add_argument(
        '--num-workers', '-n', type=int, default=3,
        help='Number of worker processes for embedding (default: 3)'
    )
    parser.add_argument(
        '--vector-store', '-v', default='sqlite', choices=['sqlite', 'chroma'],
        help='Vector store backend to use (default: sqlite)'
    )
    parser.add_argument(
        '--chroma-url', type=str,
        help='Chroma server URL (e.g., http://localhost:8000) for remote Chroma'
    )
    parser.add_argument(
        '--chroma-persist-dir', type=str,
        help='Directory to persist Chroma data locally'
    )
    parser.add_argument(
        '--collection-name', type=str, default='rag_collection',
        help='Collection name for Chroma (default: rag_collection)'
    )
    RAGTBLogger.add_logging_args(parser=parser)
    # Load subcommand
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    load_parser = subparsers.add_parser('load', help='Load documents from URLs')
    load_parser.add_argument('urls', nargs='+', help='URLs to load')
    load_parser.add_argument('--output-dir', '-o', default='assets/kb', help='Output directory')
    load_parser.add_argument('--email', '-e', help='Email for NCBI E-utilities')
    load_parser.add_argument(
        '--use-readability',
        action='store_true', help='Use readability fallback'
        )
    args = parser.parse_args()

    RAGTBLogger.configure_logging_from_args(args=args)

    # Prepare vector store configuration
    vector_store_config = {}
    if args.vector_store == 'chroma':
        if args.chroma_url:
            vector_store_config['chroma_client_url'] = args.chroma_url
        if args.chroma_persist_dir:
            vector_store_config['persist_directory'] = args.chroma_persist_dir
        vector_store_config['collection_name'] = args.collection_name

    indexer = Indexer(
        chunker = HierarchicalChunker([SectionAwareChunker(), SlidingWindowChunker()]),
        embedding_model = args.embedding_model,
        config = IndexerConfig(
            vector_store_backend = args.vector_store,
            vector_store_config = vector_store_config,
            output_dir = Path(Path(args.kb_dir) / 'embeddings')
            )
        )

    indexer.main(args)
