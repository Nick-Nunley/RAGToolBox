"""
RAGToolBox Index module.

Provides the Indexer class for embedding and storing (indexing) chunked strings.

Additionally, this script provides a CLI entry point for execution as a standalone python module.
"""

import argparse
import os
import subprocess
import re
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
from RAGToolBox.chunk import Chunker, HierarchicalChunker, SectionAwareChunker, SlidingWindowChunker
from RAGToolBox.vector_store import VectorStoreFactory

@dataclass(frozen=True)
class IndexerConfig:
    """
    Holds all the optional config settings for Indexer class.
    """
    vector_store_backend: str = "sqlite"
    vector_store_config: Optional[dict] = None
    output_dir: Path = Path("assets/kb/embeddings/")

@dataclass(frozen=True)
class ParallelConfig:
    """
    Holds all config settings for parallel processing
    """
    parallel_embed: bool = False
    num_workers: int = 3


class Indexer:
    """Indexer class for loading (optional), chunking, and embedding content."""

    def __init__(
        self, chunker: Chunker, embedding_model: str, config: Optional[IndexerConfig] = None
        ):
        self.chunker = chunker
        supported_embedding_models = ['openai', 'fastembed']
        if embedding_model not in supported_embedding_models:
            raise ValueError(
                f"Unsupported embedding model: {embedding_model}. "
                f"Embedding model must be one of: {supported_embedding_models}"
                )
        self.embedding_model = embedding_model
        if config is None:
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
        Parse a markdown document with metadata at the top (separated by a line with '---').
        Also extract the references section (after '## References')
        and store in metadata as 'references', but do not remove it from the main text.
        If no references, do not add the key.
        Returns a dictionary: { 'metadata': {...}, 'text': ... }
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
        """Chunk a document after extracting metadata. Returns (name, metadata, chunks)."""
        name, text = doc_args
        parsed = self.pre_chunk(text)
        metadata = parsed['metadata']
        main_text = parsed['text']
        return (name, metadata, self.chunker.chunk(main_text))

    def embed(self, chunks: List[str], max_retries: int = 5) -> List[list]:
        """
        Embed a list of text chunks using the configured
        embedding model (supports batching for OpenAI).
        Retries with exponential backoff on rate limit errors.
        Returns a list of embedding vectors.
        """
        if self.embedding_model == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = "text-embedding-3-small"
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        input=chunks,
                        model=model
                    )
                    return [d.embedding for d in response.data]
                except openai.RateLimitError:
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            raise RuntimeError("Failed to embed after multiple retries due to rate limits.")
        if self.embedding_model == "fastembed":
            from fastembed import TextEmbedding
            model = TextEmbedding()
            embeddings = [list(model.embed(chunk))[0].tolist() for chunk in chunks]
            return embeddings
        raise ValueError(f"Embedding model '{self.embedding_model}' not supported.")

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
        Embed the provided chunks and store results in the SQLite database, optionally in parallel.
        Args:
            chunked_results: List of dicts with 'chunk' and 'metadata'.
            parallel_embed: If True, use multiple processes for embedding.
            num_workers: Number of worker processes if parallel_embed is True.
        """
        if parallel_config is None:
            parallel_config = ParallelConfig()
        chunk_texts = [entry['chunk'] for entry in chunked_results]
        if not chunk_texts:
            print("No chunks to embed.")
            return
        if parallel_config.parallel_embed:
            # heuristic: 2 batches per worker
            batch_size = max(1, len(chunk_texts) // (parallel_config.num_workers * 2))
            batches = [chunk_texts[i:i+batch_size] for i in range(0, len(chunk_texts), batch_size)]
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(
                f"Embedding {len(chunk_texts)} chunks "
                f"using {parallel_config.num_workers} workers..."
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
                    future.result()  # raise exceptions if any
        else:
            embeddings = self.embed(chunk_texts)
            self._insert_embeddings_to_db(chunked_results, embeddings)
        print("Indexing complete.")

    def _optional_loading_and_kb_init(self, cli_args: argparse.Namespace) -> Path:
        """Helper method for handling optional loading and initializing kb path in main"""
        if getattr(cli_args, 'command', None) == 'load':
            subprocess.run([
                'python', 'RAGToolBox/loader.py', *cli_args.urls,
                '--output-dir', cli_args.output_dir,
                *(["--email", cli_args.email] if cli_args.email else []),
                *(["--use-readability"] if cli_args.use_readability else [])
                ], check=True)
            # After loading, continue to chunking and indexing
            return Path(cli_args.output_dir)
        return Path(getattr(cli_args, 'kb_dir', 'assets/kb'))

    def _concurrent_chunker(self, docs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """Helper method for implementing concurrent chunking"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Chunking {len(docs)} documents using {os.cpu_count()} processes...")
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
                print(f"Chunked {name}: {len(chunks)} chunks")
        return chunked_results

    def main(self, cli_args: argparse.Namespace) -> None:
        """Main method for indexing content."""
        kb_dir = self._optional_loading_and_kb_init(cli_args=cli_args)

        # 1. Gather all .txt files in the knowledge base directory
        txt_files = list(kb_dir.glob('*.txt'))
        if not txt_files:
            print(f"No .txt files found in {kb_dir}. Skipping chunking and indexing.")
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
