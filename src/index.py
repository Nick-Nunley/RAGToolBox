import argparse
import os
import subprocess
from typing import Optional, List
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

    def main(self, args: argparse.Namespace) -> None:
        """Main method for indexing content."""
        if getattr(args, 'command', None) == 'load':
            subprocess.run([
                'python', 'src/loader.py', *args.urls,
                '--output-dir', args.output_dir,
                *(["--email", args.email] if args.email else []),
                *(["--use-readability"] if args.use_readability else [])
                ])


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

