import argparse
import os
import subprocess
from typing import Optional, List, Tuple
from pathlib import Path
from src.chunk import Chunker


class Indexer:
    """Indexer class for loading (optional), chunking, and embedding content."""

    def __init__(self, chunker: Chunker, embedding_model: str, kb_dir: Path = Path('assets/kb')):
        self.chunker = chunker
        supported_embedding_models = ['openai']
        if embedding_model not in supported_embedding_models:
            raise ValueError(f"Unsupported embedding model: {embedding_model}. Embedding model must be one of: {supported_embedding_models}")
        self.embedding_model = embedding_model

    def chunk(self, args: argparse.Namespace) -> Tuple[str, List[str]]:
        """Method to chunk a documnet."""
        name, text = args
        return (name, self.chunker.chunk(text))

    def main(self, args: argparse.Namespace) -> None:
        """Main method for indexing content."""
        if getattr(args, 'command', None) == 'load':
            subprocess.run([
                'python', 'src/loader.py', *args.urls,
                '--output-dir', args.output_dir,
                *(["--email", args.email] if args.email else []),
                *(["--use-readability"] if args.use_readability else [])
                ])
            # After loading, continue to chunking and indexing
            kb_dir = Path(args.output_dir)
        else:
            kb_dir = Path(getattr(args, 'kb_dir', 'assets/kb'))

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
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"Chunking {len(docs)} documents using {os.cpu_count()} processes...")
        chunked_results = {}
        with ProcessPoolExecutor() as executor:
            future_to_name = {executor.submit(self.chunk, doc): doc[0] for doc in docs}
            for future in as_completed(future_to_name):
                name, chunks = future.result()
                chunked_results[name] = chunks
                print(f"Chunked {name}: {len(chunks)} chunks")

        # 4. Embedding and indexing (single-threaded, rate-limited step)
        # Placeholder: implement embedding/indexing logic here
        for name, chunks in chunked_results.items():
            print(f"Indexing {name} with {len(chunks)} chunks (embedding model: {self.embedding_model})")
            # TODO: Call embedding API and add to index here
        print("Indexing complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Indexing pipeline with optional loading"
    )
    parser.add_argument(
        '--kb-dir', '-k', default='assets/kb',
        help='Directory where knowledge base is stored'
    )
    parser.add_argument(
        '--embedding-model', '-e', default='openai',
        help='Embedding model to use'
    )
    # Load subcommand
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    load_parser = subparsers.add_parser('load', help='Load documents from URLs')
    load_parser.add_argument('urls', nargs='+', help='URLs to load')
    load_parser.add_argument('--output-dir', '-o', default='assets/kb', help='Output directory')
    load_parser.add_argument('--email', '-e', help='Email for NCBI E-utilities')
    load_parser.add_argument('--use-readability', action='store_true', help='Use readability fallback')
    args = parser.parse_args()

