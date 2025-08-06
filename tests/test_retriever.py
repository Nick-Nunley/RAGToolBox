import pytest
import os
import tempfile
import sqlite3
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import json
from RAGToolBox.vector_store import SQLiteVectorStore

from RAGToolBox.retriever import Retriever

@pytest.fixture(autouse=True)
def no_sqlite_init(monkeypatch):
    # prevent it from creating any files on disk
    monkeypatch.setattr(SQLiteVectorStore, "initialize", lambda self: None)


# =====================
# UNIT TESTS
# =====================

def test_retriever_init_valid_models(tmp_path: Path) -> None:
    """Test Retriever initialization with valid embedding models"""
    # Test with openai
    retriever = Retriever(embedding_model='openai')
    assert retriever.embedding_model == 'openai'
    assert retriever.db_path == Path('assets/kb/embeddings/embeddings.db')

    # Test with fastembed
    retriever = Retriever(embedding_model='fastembed')
    assert retriever.embedding_model == 'fastembed'

    # Test with custom db path
    custom_path = Path('custom/path/embeddings.db')
    custom_path = tmp_path / 'custom' / 'embeddings.db'
    retriever = Retriever(embedding_model='openai', db_path=custom_path)
    assert retriever.db_path == custom_path


def test_retriever_init_invalid_model() -> None:
    """Test Retriever initialization with invalid embedding model"""
    with pytest.raises(ValueError) as exc_info:
        Retriever(embedding_model='invalid_model')

    assert 'Unsupported embedding model: invalid_model' in str(exc_info.value)
    assert 'openai' in str(exc_info.value)
    assert 'fastembed' in str(exc_info.value)


def test_load_db_success() -> None:
    """Test successful database loading"""
    # Create a temporary database with test data
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Create test database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE embeddings (
                id TEXT PRIMARY KEY,
                chunk TEXT,
                embedding TEXT,
                metadata TEXT,
                source TEXT
            )
        ''')

        # Insert test data
        test_data = [
            ('hash1', 'test text 1', '[0.1, 0.2, 0.3]', '{"title": "Test 1"}', 'test1.txt'),
            ('hash2', 'test text 2', '[0.4, 0.5, 0.6]', '{"title": "Test 2"}', 'test2.txt'),
            ('hash3', 'test text 3', '[0.7, 0.8, 0.9]', '{"title": "Test 3"}', 'test3.txt')
        ]
        cursor.executemany(
            'INSERT INTO embeddings (id, chunk, embedding, metadata, source) VALUES (?, ?, ?, ?, ?)',
            test_data
        )
        conn.commit()
        conn.close()

        # Test loading
        retriever = Retriever(embedding_model='openai', db_path=Path(db_path))
        embeddings_data = retriever.vector_store.get_all_embeddings()

        assert len(embeddings_data) == 3
        assert embeddings_data[0]['chunk'] == 'test text 1'
        assert embeddings_data[1]['chunk'] == 'test text 2'
        assert embeddings_data[2]['chunk'] == 'test text 3'

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_load_db_file_not_found() -> None:
    """Test database loading when file doesn't exist"""
    # Use a temporary path that definitely doesn't exist
    with tempfile.NamedTemporaryFile(suffix='.db', delete=True) as tmp_file:
        # Close and delete the temp file immediately
        tmp_file.close()
        db_path = tmp_file.name

    # Ensure the file doesn't exist
    assert not os.path.exists(db_path)

    retriever = Retriever(embedding_model='openai', db_path=Path(db_path))

    # Should not raise an exception, but should return empty results
    embeddings_data = retriever.vector_store.get_all_embeddings()
    assert len(embeddings_data) == 0

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_load_db_empty_database() -> None:
    """Test database loading when database exists but has no embeddings table"""
    # Create a temporary empty database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Create empty database (no tables)
        conn = sqlite3.connect(db_path)
        conn.close()

        retriever = Retriever(embedding_model='openai', db_path=Path(db_path))

        # The new implementation should handle empty databases gracefully
        # by creating the table when needed
        embeddings_data = retriever.vector_store.get_all_embeddings()
        assert len(embeddings_data) == 0

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_embed_query_fastembed_success() -> None:
    """Test successful FastEmbed embedding"""
    # Skip this test if fastembed is not available
    try:
        from fastembed import TextEmbedding
    except ImportError:
        pytest.skip("fastembed package not available")

    # Mock FastEmbed
    with patch('fastembed.TextEmbedding') as mock_text_embedding:
        mock_model = Mock()
        mock_model.embed.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_text_embedding.return_value = mock_model

        retriever = Retriever(embedding_model='fastembed')
        result = retriever._embed_query('test query')

        # Verify FastEmbed was called correctly
        mock_text_embedding.assert_called_once()
        mock_model.embed.assert_called_once_with('test query')

        # Verify result
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        assert result.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_embed_query_unsupported_model() -> None:
    """Test embedding with unsupported model"""
    retriever = Retriever(embedding_model='openai')
    # Manually set to unsupported model to test the error case
    retriever.embedding_model = 'unsupported'

    with pytest.raises(ValueError) as exc_info:
        retriever._embed_query('test query')

    assert "Embedding model 'unsupported' not supported" in str(exc_info.value)

def test_retrieve_method_full_workflow() -> None:
    """Test the complete retrieve method workflow with dummy data"""
    # Create a temporary database with test data
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Create test database with embeddings and chunks
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE embeddings (
                id TEXT PRIMARY KEY,
                chunk TEXT,
                embedding TEXT,
                metadata TEXT,
                source TEXT
            )
        ''')

        # Create test data with known embeddings for predictable similarity scores
        test_data = [
            ('hash1', 'biomedical research on ultrasound therapy', '[0.9, 0.1, 0.2, 0.3, 0.4]', '{"title": "Biomedical"}', 'bio.txt'),
            ('hash2', 'clinical trials for drug discovery', '[0.1, 0.8, 0.2, 0.3, 0.4]', '{"title": "Clinical"}', 'clinical.txt'),
            ('hash3', 'neuroscience and brain imaging studies', '[0.1, 0.2, 0.7, 0.3, 0.4]', '{"title": "Neuroscience"}', 'neuro.txt'),
            ('hash4', 'cancer treatment protocols', '[0.1, 0.2, 0.3, 0.6, 0.4]', '{"title": "Cancer"}', 'cancer.txt'),
            ('hash5', 'medical device regulations', '[0.1, 0.2, 0.3, 0.4, 0.5]', '{"title": "Medical"}', 'medical.txt')
        ]
        cursor.executemany(
            'INSERT INTO embeddings (id, chunk, embedding, metadata, source) VALUES (?, ?, ?, ?, ?)',
            test_data
        )
        conn.commit()
        conn.close()

        # Mock the embedding method to return a predictable query embedding
        with patch.object(Retriever, '_embed_query') as mock_embed:
            # Query embedding that will give highest dot product with first chunk
            mock_embed.return_value = np.array([0.9, 0.1, 0.2, 0.3, 0.4])

            retriever = Retriever(embedding_model='fastembed', db_path=Path(db_path))

            # Test retrieval with top_k=3
            results = retriever.retrieve('biomedical ultrasound research', top_k=3)

            # Verify the method was called correctly
            mock_embed.assert_called_once_with(query='biomedical ultrasound research', max_retries=5)

            # Verify we got exactly 3 results
            assert len(results) == 3

            expected_order = [
                {'data': 'biomedical research on ultrasound therapy', 'metadata': {'title': 'Biomedical'}},  # Highest similarity
                {'data': 'cancer treatment protocols', 'metadata': {'title': 'Cancer'}},                     # Second highest
                {'data': 'neuroscience and brain imaging studies', 'metadata': {'title': 'Neuroscience'}}    # Third highest
            ]
            assert results == expected_order

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_retrieve_method_top_k_parameter() -> None:
    """Test that the top_k parameter correctly limits the number of results"""
    # Create a temporary database with test data
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Create test database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE embeddings (
                id TEXT PRIMARY KEY,
                chunk TEXT,
                embedding TEXT,
                metadata TEXT,
                source TEXT
            )
        ''')

        # Insert test data
        test_data = [
            ('hash1', 'chunk1', '[0.9, 0.1, 0.2]', '{"title": "Chunk 1"}', 'chunk1.txt'),
            ('hash2', 'chunk2', '[0.8, 0.2, 0.3]', '{"title": "Chunk 2"}', 'chunk2.txt'),
            ('hash3', 'chunk3', '[0.7, 0.3, 0.4]', '{"title": "Chunk 3"}', 'chunk3.txt'),
            ('hash4', 'chunk4', '[0.6, 0.4, 0.5]', '{"title": "Chunk 4"}', 'chunk4.txt'),
            ('hash5', 'chunk5', '[0.5, 0.5, 0.6]', '{"title": "Chunk 5"}', 'chunk5.txt')
        ]
        cursor.executemany(
            'INSERT INTO embeddings (id, chunk, embedding, metadata, source) VALUES (?, ?, ?, ?, ?)',
            test_data
        )
        conn.commit()
        conn.close()

        # Mock the embedding method
        with patch.object(Retriever, '_embed_query') as mock_embed:
            mock_embed.return_value = np.array([0.9, 0.1, 0.2])

            retriever = Retriever(embedding_model='fastembed', db_path=Path(db_path))

            # Test with different top_k values
            results_k2 = retriever.retrieve('test query', top_k=2)
            results_k4 = retriever.retrieve('test query', top_k=4)

            assert len(results_k2) == 2
            assert len(results_k4) == 4

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_retrieve_method_empty_database() -> None:
    """Test retrieve method with empty database"""
    # Create a temporary empty database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Create empty database (no tables)
        conn = sqlite3.connect(db_path)
        conn.close()

        with patch.object(Retriever, '_embed_query') as mock_embed:
            mock_embed.return_value = np.array([0.1, 0.2, 0.3])

            retriever = Retriever(embedding_model='fastembed', db_path=Path(db_path))

            # Should return empty list instead of raising exception
            results = retriever.retrieve('test query')
            assert len(results) == 0

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


# =====================
# EDGE CASES AND ERROR HANDLING
# =====================

def test_retriever_empty_query() -> None:
    """Test embedding with empty query"""
    retriever = Retriever(embedding_model='fastembed')

    # This should work without error
    embedding = retriever._embed_query('')
    assert isinstance(embedding, np.ndarray)


def test_retriever_very_long_query() -> None:
    """Test embedding with very long query"""
    retriever = Retriever(embedding_model='fastembed')

    long_query = 'This is a very long query ' * 100  # 2500 characters
    embedding = retriever._embed_query(long_query)
    assert isinstance(embedding, np.ndarray)


def test_retriever_special_characters_query() -> None:
    """Test embedding with special characters"""
    retriever = Retriever(embedding_model='fastembed')

    special_query = "Query with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
    embedding = retriever._embed_query(special_query)
    assert isinstance(embedding, np.ndarray)


def test_retriever_unicode_query() -> None:
    """Test embedding with unicode characters"""
    retriever = Retriever(embedding_model='fastembed')

    unicode_query = "Query with unicode: αβγδε 中文 español français"
    embedding = retriever._embed_query(unicode_query)
    assert isinstance(embedding, np.ndarray)


# =====================
# INTEGRATION TESTS
# =====================

def test_retriever_full_integration() -> None:
    """Integration test using actual Retriever.retrieve method without mocking internal methods"""
    # Skip this test if fastembed is not available
    try:
        from fastembed import TextEmbedding
    except ImportError:
        pytest.skip("fastembed package not available")

    # Create a temporary database with test data
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Create test database with real embeddings
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE embeddings (
                id TEXT PRIMARY KEY,
                chunk TEXT,
                embedding TEXT,
                metadata TEXT,
                source TEXT
            )
        ''')

        # Create test data with biomedical content
        chunks = [
            {"data": 'Low-intensity focused ultrasound therapy for non-invasive brain stimulation', "metadata": {}},
            {"data": 'Clinical trials investigating drug efficacy in cancer treatment', "metadata": {}},
            {"data": 'Medical device regulations and safety standards', "metadata": {}},
            {"data": 'Neuroscience research on brain imaging techniques', "metadata": {}},
            {"data": 'Biomedical engineering applications in healthcare', "metadata": {}}
        ]

        # Generate real embeddings for the test chunks using FastEmbed
        model = TextEmbedding()

        # Insert test data
        for i, chunk in enumerate(chunks):
            embedding = list(model.embed(chunk['data']))[0]
            chunk_id = f'hash_{i}'
            cursor.execute('''
                INSERT INTO embeddings (id, chunk, embedding, metadata, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (chunk_id, chunk['data'], json.dumps(embedding.tolist()), '{"title": "Test"}', f'test_{i}.txt'))

        conn.commit()
        conn.close()

        # Create retriever and test actual retrieval
        retriever = Retriever(embedding_model='fastembed', db_path=Path(db_path))

        # Test retrieval with a query related to ultrasound
        query = "ultrasound therapy brain stimulation"
        results = retriever.retrieve(query, top_k=3)

        # Verify we got results
        assert len(results) == 3
        assert isinstance(results, list)
        assert all(isinstance(result, dict) for result in results)

        # Verify the first result is most relevant (should be about ultrasound)
        assert 'ultrasound' in results[0]['data'].lower()

        # Test with a different query
        query2 = "medical device safety"
        results2 = retriever.retrieve(query2, top_k=2)

        assert len(results2) == 2
        # Should include the medical device regulation chunk
        assert any('regulation' in result['data'].lower() for result in results2)

        # Test with top_k larger than available data
        results3 = retriever.retrieve("biomedical research", top_k=10)
        assert len(results3) == 5  # Should return all available chunks

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
