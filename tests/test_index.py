"""Tests associated with Index module"""

import argparse
import sqlite3
from unittest.mock import patch, MagicMock
from RAGToolBox.index import Indexer, IndexerConfig, ParallelConfig

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
        assert called_args[:3] == ['python', 'RAGToolBox/loader.py', 'http://example.com/doc1']
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


# Test the embed method
def test_indexer_embed():
    indexer = Indexer(DummyChunker(), embedding_model='openai')
    chunks = ["chunk one", "chunk two"]
    fake_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    with patch('openai.OpenAI') as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Simulate response.data as a list of objects with .embedding attribute
        mock_response.data = [MagicMock(embedding=fake_embeddings[0]), MagicMock(embedding=fake_embeddings[1])]
        mock_client.embeddings.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = indexer.embed(chunks)
        assert result == fake_embeddings


def test_insert_embeddings_to_db(tmp_path):
    # Setup
    output_dir = tmp_path
    indexer = Indexer(DummyChunker(), embedding_model='openai', config = IndexerConfig(output_dir=output_dir))
    chunked_results = [
        {'chunk': 'This is chunk 1.', 'metadata': {'source': 'test1'}},
        {'chunk': 'This is chunk 2.', 'metadata': {'source': 'test2'}},
    ]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    # Act
    indexer._insert_embeddings_to_db(chunked_results, embeddings)
    # Assert
    db_path = output_dir / 'embeddings.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, chunk, embedding, metadata, source FROM embeddings')
    rows = cursor.fetchall()
    assert len(rows) == 2
    chunks = {row[1] for row in rows}
    sources = {row[4] for row in rows}
    assert chunks == {'This is chunk 1.', 'This is chunk 2.'}
    assert sources == {'test1', 'test2'}
    conn.close()

def test_embed_and_save_in_batch(tmp_path):
    # Setup
    output_dir = tmp_path
    class FakeIndexer(Indexer):
        def embed(self, chunks):
            # Return a fake embedding for each chunk
            return [[float(i)] for i in range(len(chunks))]
    indexer = FakeIndexer(DummyChunker(), embedding_model='openai', config = IndexerConfig(output_dir=output_dir))
    batch = ['chunkA', 'chunkB']
    batch_entries = [
        {'chunk': 'chunkA', 'metadata': {'source': 'A'}},
        {'chunk': 'chunkB', 'metadata': {'source': 'B'}},
    ]
    # Act
    indexer._embed_and_save_in_batch(batch, batch_entries)
    # Assert
    db_path = output_dir / 'embeddings.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT chunk, embedding, source FROM embeddings')
    rows = cursor.fetchall()
    assert len(rows) == 2
    chunks = {row[0] for row in rows}
    sources = {row[2] for row in rows}
    assert chunks == {'chunkA', 'chunkB'}
    assert sources == {'A', 'B'}
    conn.close()

def test_index_method_single_threaded(tmp_path):
    output_dir = tmp_path
    class FakeIndexer(Indexer):
        def embed(self, chunks):
            # Return a fake embedding for each chunk
            return [[float(i)] for i in range(len(chunks))]
    indexer = FakeIndexer(DummyChunker(), embedding_model='openai', config=IndexerConfig(output_dir=output_dir))
    chunked_results = [
        {'chunk': 'chunkA', 'metadata': {'source': 'A'}},
        {'chunk': 'chunkB', 'metadata': {'source': 'B'}},
    ]
    # Act
    indexer.index(chunked_results, parallel_config=ParallelConfig(parallel_embed=False))
    # Assert
    db_path = output_dir / 'embeddings.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT chunk, source FROM embeddings')
    rows = cursor.fetchall()
    chunks = {row[0] for row in rows}
    sources = {row[1] for row in rows}
    assert chunks == {'chunkA', 'chunkB'}
    assert sources == {'A', 'B'}
    conn.close()
