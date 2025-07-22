import pytest
import argparse
from unittest.mock import patch
from src.index import Indexer
from src.chunk import Chunker

class DummyChunker:
    def chunk(self, text: str):
        return [text]


# =====================
# UNIT TESTS
# =====================

# Test the load subparser
def test_indexer_load_subparser(monkeypatch):
    # Recreate the parser logic from src/index.py
    parser = argparse.ArgumentParser(description="Indexing pipeline with optional loading")
    parser.add_argument('--kb-dir', '-k', default='assets/kb', help='Directory where knowledge base is stored')
    parser.add_argument('--embedding-model', '-e', default='openai', help='Embedding model to use')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    load_parser = subparsers.add_parser('load', help='Load documents from URLs')
    load_parser.add_argument('urls', nargs='+', help='URLs to load')
    load_parser.add_argument('--output-dir', '-o', default='assets/kb', help='Output directory')
    load_parser.add_argument('--email', '-e', help='Email for NCBI E-utilities')
    load_parser.add_argument('--use-readability', action='store_true', help='Use readability fallback')

    # Simulate CLI args for the load subcommand
    cli_args = [
        'load',
        'http://example.com/doc1',
        '--output-dir', 'mydir',
        '--email', 'test@example.com',
        '--use-readability'
        ]
    args = parser.parse_args(cli_args)
    assert args.command == 'load'
    assert args.urls == ['http://example.com/doc1']
    assert args.output_dir == 'mydir'
    assert args.email == 'test@example.com'
    assert args.use_readability is True

    # Patch subprocess.run to check invocation
    with patch('subprocess.run') as mock_run:
        indexer = Indexer(DummyChunker(), embedding_model='openai')
        indexer.main(args)
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert called_args[:3] == ['python', 'src/loader.py', 'http://example.com/doc1']
        assert '--output-dir' in called_args and 'mydir' in called_args
        assert '--email' in called_args and 'test@example.com' in called_args
        assert '--use-readability' in called_args


# Test the chunk method
def test_indexer_chunk():
    indexer = Indexer(DummyChunker(), embedding_model='openai')
    doc = ("test.txt", "This is a test document.")
    result = indexer.chunk(doc)
    assert isinstance(result, tuple)
    assert result[0] == "test.txt"
    assert isinstance(result[1], dict)  # metadata
    assert result[1] == {}  # no metadata in this doc
    assert isinstance(result[2], list)  # chunks
    assert result[2] == ["This is a test document."]

