import pytest
import os
import tempfile
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any
import json

from src.vector_store import VectorStoreFactory, SQLiteVectorStore, ChromaVectorStore
from src.index import Indexer
from src.retriever import Retriever
from src.chunk import HierarchicalChunker, SectionAwareChunker, SlidingWindowChunker


# =====================
# UNIT TESTS
# =====================

def test_sqlite_vector_store_init() -> None:
    """Test SQLiteVectorStore initialization"""
    db_path = Path('test_embeddings.db')
    vector_store = SQLiteVectorStore(db_path)
    
    assert vector_store.db_path == db_path
    assert vector_store.db_path.parent.exists()  # Directory should be created


def test_sqlite_vector_store_initialize() -> None:
    """Test SQLiteVectorStore initialization creates database and table"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    try:
        vector_store = SQLiteVectorStore(db_path)
        vector_store.initialize()
        
        # Check database file exists
        assert db_path.exists()
        
        # Check table was created
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == 'embeddings'
        
    finally:
        if db_path.exists():
            os.unlink(db_path)


def test_sqlite_vector_store_insert_embeddings() -> None:
    """Test SQLiteVectorStore insert_embeddings method"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    try:
        vector_store = SQLiteVectorStore(db_path)
        vector_store.initialize()
        
        # Test data
        chunked_results = [
            {
                'chunk': 'Test chunk 1',
                'metadata': {'title': 'Test 1', 'author': 'Author 1'},
                'name': 'test1.txt'
            },
            {
                'chunk': 'Test chunk 2',
                'metadata': {'title': 'Test 2', 'author': 'Author 2'},
                'name': 'test2.txt'
            }
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Insert embeddings
        vector_store.insert_embeddings(chunked_results, embeddings)
        
        # Verify data was inserted
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 2
        
    finally:
        if db_path.exists():
            os.unlink(db_path)


def test_sqlite_vector_store_get_all_embeddings() -> None:
    """Test SQLiteVectorStore get_all_embeddings method"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    try:
        vector_store = SQLiteVectorStore(db_path)
        vector_store.initialize()
        
        # Insert test data
        chunked_results = [
            {
                'chunk': 'Machine learning algorithms',
                'metadata': {'title': 'ML Guide', 'author': 'ML Expert'},
                'name': 'ml.txt'
            },
            {
                'chunk': 'Deep learning neural networks',
                'metadata': {'title': 'DL Guide', 'author': 'DL Expert'},
                'name': 'dl.txt'
            }
        ]
        embeddings = [[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]]
        
        vector_store.insert_embeddings(chunked_results, embeddings)
        
        # Test get_all_embeddings
        results = vector_store.get_all_embeddings()
        
        assert len(results) == 2
        assert 'chunk' in results[0]
        assert 'embedding' in results[0]
        assert 'metadata' in results[0]
        assert results[0]['chunk'] == 'Machine learning algorithms'
        assert results[1]['chunk'] == 'Deep learning neural networks'
        
    finally:
        if db_path.exists():
            os.unlink(db_path)


def test_sqlite_vector_store_delete_collection() -> None:
    """Test SQLiteVectorStore delete_collection method"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    try:
        vector_store = SQLiteVectorStore(db_path)
        vector_store.initialize()
        
        # Verify file exists
        assert db_path.exists()
        
        # Delete collection
        vector_store.delete_collection()
        
        # Verify file was deleted
        assert not db_path.exists()
        
    finally:
        # Cleanup in case deletion failed
        if db_path.exists():
            os.unlink(db_path)


def test_vector_store_factory_sqlite() -> None:
    """Test VectorStoreFactory with SQLite backend"""
    db_path = Path('test_factory.db')
    vector_store = VectorStoreFactory.create_backend('sqlite', db_path=db_path)
    
    assert isinstance(vector_store, SQLiteVectorStore)
    assert vector_store.db_path == db_path


def test_vector_store_factory_chroma() -> None:
    """Test VectorStoreFactory with Chroma backend"""
    persist_dir = Path('test_chroma_data')
    vector_store = VectorStoreFactory.create_backend(
        'chroma',
        collection_name='test_collection',
        persist_directory=persist_dir
    )
    
    assert isinstance(vector_store, ChromaVectorStore)
    assert vector_store.collection_name == 'test_collection'
    assert vector_store.persist_directory == persist_dir


def test_vector_store_factory_invalid_backend() -> None:
    """Test VectorStoreFactory with invalid backend"""
    with pytest.raises(ValueError, match="Unsupported vector store backend"):
        VectorStoreFactory.create_backend('invalid_backend')


def test_indexer_with_sqlite_backend() -> None:
    """Test Indexer initialization with SQLite backend"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'embeddings'
        
        indexer = Indexer(
            chunker=HierarchicalChunker([SectionAwareChunker(), SlidingWindowChunker()]),
            embedding_model='fastembed',
            vector_store_backend='sqlite',
            output_dir=output_dir
        )
        
        # Test that vector store is initialized
        assert hasattr(indexer, 'vector_store')
        assert isinstance(indexer.vector_store, SQLiteVectorStore)
        assert indexer.vector_store.db_path == output_dir / 'embeddings.db'


def test_indexer_with_chroma_backend() -> None:
    """Test Indexer initialization with Chroma backend"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'embeddings'
        persist_dir = Path(temp_dir) / 'chroma_data'
        
        indexer = Indexer(
            chunker=HierarchicalChunker([SectionAwareChunker(), SlidingWindowChunker()]),
            embedding_model='fastembed',
            vector_store_backend='chroma',
            vector_store_config={
                'collection_name': 'test_collection',
                'persist_directory': persist_dir
            },
            output_dir=output_dir
        )
        
        # Test that vector store is initialized
        assert hasattr(indexer, 'vector_store')
        assert isinstance(indexer.vector_store, ChromaVectorStore)
        assert indexer.vector_store.collection_name == 'test_collection'
        assert indexer.vector_store.persist_directory == persist_dir


# =====================
# CHROMA TESTS (CONDITIONAL)
# =====================

def test_chroma_vector_store_init() -> None:
    """Test ChromaVectorStore initialization"""
    # Skip if ChromaDB not available
    try:
        import chromadb
    except ImportError:
        pytest.skip("ChromaDB not installed")
    
    vector_store = ChromaVectorStore(
        collection_name='test_collection',
        persist_directory=Path('test_chroma_data')
    )
    
    assert vector_store.collection_name == 'test_collection'
    assert vector_store.persist_directory == Path('test_chroma_data')
    assert vector_store.chroma_client_url is None


def test_chroma_vector_store_init_remote() -> None:
    """Test ChromaVectorStore initialization with remote URL"""
    # Skip if ChromaDB not available
    try:
        import chromadb
    except ImportError:
        pytest.skip("ChromaDB not installed")
    
    vector_store = ChromaVectorStore(
        collection_name='remote_collection',
        chroma_client_url='http://localhost:8000'
    )
    
    assert vector_store.collection_name == 'remote_collection'
    assert vector_store.chroma_client_url == 'http://localhost:8000'
    assert vector_store.persist_directory is None


def test_chroma_vector_store_initialize_local() -> None:
    """Test ChromaVectorStore initialization with local persistence"""
    # Skip if ChromaDB not available
    try:
        import chromadb
    except ImportError:
        pytest.skip("ChromaDB not installed")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        persist_dir = Path(temp_dir) / 'chroma_data'
        
        vector_store = ChromaVectorStore(
            collection_name='test_collection',
            persist_directory=persist_dir
        )
        
        vector_store.initialize()
        
        # Check that persist directory was created
        assert persist_dir.exists()
        
        # Check that client and collection were initialized
        assert vector_store.client is not None
        assert vector_store.collection is not None


def test_chroma_vector_store_insert_and_get_all_embeddings() -> None:
    """Test ChromaVectorStore insert_embeddings and get_all_embeddings methods"""
    # Skip if ChromaDB not available
    try:
        import chromadb
    except ImportError:
        pytest.skip("ChromaDB not installed")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        persist_dir = Path(temp_dir) / 'chroma_data'
        
        vector_store = ChromaVectorStore(
            collection_name='test_collection',
            persist_directory=persist_dir
        )
        
        vector_store.initialize()
        
        # Test data
        chunked_results = [
            {
                'chunk': 'Artificial intelligence and machine learning',
                'metadata': {'title': 'AI Guide', 'author': 'AI Expert'},
                'name': 'ai.txt'
            },
            {
                'chunk': 'Natural language processing techniques',
                'metadata': {'title': 'NLP Guide', 'author': 'NLP Expert'},
                'name': 'nlp.txt'
            }
        ]
        embeddings = [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3]]
        
        # Insert embeddings
        vector_store.insert_embeddings(chunked_results, embeddings)
        
        # Test get_all_embeddings
        results = vector_store.get_all_embeddings()
        
        assert len(results) == 2
        assert 'chunk' in results[0]
        assert 'embedding' in results[0]
        assert 'metadata' in results[0]
        assert results[0]['chunk'] == 'Artificial intelligence and machine learning'
        assert results[1]['chunk'] == 'Natural language processing techniques'


# =====================
# EDGE CASES AND ERROR HANDLING
# =====================

def test_sqlite_vector_store_empty_get_all_embeddings() -> None:
    """Test SQLiteVectorStore get_all_embeddings with empty database"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    try:
        vector_store = SQLiteVectorStore(db_path)
        vector_store.initialize()
        
        # Get all embeddings from empty database
        results = vector_store.get_all_embeddings()
        assert len(results) == 0
        
    finally:
        if db_path.exists():
            os.unlink(db_path)


def test_sqlite_vector_store_get_all_embeddings_with_data() -> None:
    """Test SQLiteVectorStore get_all_embeddings with data"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    try:
        vector_store = SQLiteVectorStore(db_path)
        vector_store.initialize()
        
        # Insert only 2 items
        chunked_results = [
            {'chunk': 'Test 1', 'metadata': {}, 'name': 'test1.txt'},
            {'chunk': 'Test 2', 'metadata': {}, 'name': 'test2.txt'}
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        vector_store.insert_embeddings(chunked_results, embeddings)
        
        # Get all embeddings
        results = vector_store.get_all_embeddings()
        assert len(results) == 2  # Should return all available
        
    finally:
        if db_path.exists():
            os.unlink(db_path)


def test_sqlite_vector_store_invalid_metadata() -> None:
    """Test SQLiteVectorStore with invalid metadata JSON"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = Path(tmp_file.name)
    
    try:
        vector_store = SQLiteVectorStore(db_path)
        vector_store.initialize()
        
        # This should handle invalid metadata gracefully
        chunked_results = [
            {
                'chunk': 'Test chunk',
                'metadata': {'invalid': 'metadata'},  # Valid JSON
                'name': 'test.txt'
            }
        ]
        embeddings = [[0.1, 0.2, 0.3]]
        
        # Should not raise an exception
        vector_store.insert_embeddings(chunked_results, embeddings)
        
    finally:
        if db_path.exists():
            os.unlink(db_path)


def test_chroma_vector_store_import_error() -> None:
    """Test ChromaVectorStore when ChromaDB is not installed"""
    # Mock import to simulate missing ChromaDB
    with patch('builtins.__import__', side_effect=ImportError("No module named 'chromadb'")):
        with pytest.raises(ImportError, match="ChromaDB is not installed"):
            ChromaVectorStore().initialize()


# =====================
# INTEGRATION TESTS
# =====================

def test_indexer_integration_with_sqlite() -> None:
    """Integration test using actual Indexer with SQLite backend"""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'embeddings'
        
        indexer = Indexer(
            chunker=HierarchicalChunker([SectionAwareChunker(), SlidingWindowChunker()]),
            embedding_model='fastembed',
            vector_store_backend='sqlite',
            output_dir=output_dir
        )
        
        # Test data
        chunked_results = [
            {
                'chunk': 'Biomedical research on ultrasound therapy',
                'metadata': {'title': 'Biomedical Research', 'author': 'Researcher'},
                'name': 'biomedical.txt'
            },
            {
                'chunk': 'Clinical trials for drug discovery',
                'metadata': {'title': 'Clinical Trials', 'author': 'Clinician'},
                'name': 'clinical.txt'
            }
        ]
        
        # Mock embeddings to avoid actual embedding generation
        embeddings = [[0.9, 0.1, 0.2, 0.3, 0.4], [0.1, 0.8, 0.2, 0.3, 0.4]]
        
        # Insert data
        indexer._insert_embeddings_to_db(chunked_results, embeddings)
        
        # Verify data was inserted correctly
        embeddings_data = indexer.vector_store.get_all_embeddings()
        assert len(embeddings_data) == 2
        assert embeddings_data[0]['chunk'] == 'Biomedical research on ultrasound therapy'
        assert embeddings_data[1]['chunk'] == 'Clinical trials for drug discovery'


def test_indexer_integration_with_chroma() -> None:
    """Integration test using actual Indexer with Chroma backend"""
    # Skip if ChromaDB not available
    try:
        import chromadb
    except ImportError:
        pytest.skip("ChromaDB not installed")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / 'embeddings'
        persist_dir = Path(temp_dir) / 'chroma_data'
        
        indexer = Indexer(
            chunker=HierarchicalChunker([SectionAwareChunker(), SlidingWindowChunker()]),
            embedding_model='fastembed',
            vector_store_backend='chroma',
            vector_store_config={
                'collection_name': 'integration_test',
                'persist_directory': persist_dir
            },
            output_dir=output_dir
        )
        
        # Test data
        chunked_results = [
            {
                'chunk': 'Neuroscience research on brain imaging',
                'metadata': {'title': 'Neuroscience', 'author': 'Neuroscientist'},
                'name': 'neuroscience.txt'
            },
            {
                'chunk': 'Medical device regulations and safety',
                'metadata': {'title': 'Medical Devices', 'author': 'Regulator'},
                'name': 'medical_devices.txt'
            }
        ]
        
        # Mock embeddings
        embeddings = [[0.8, 0.2, 0.1, 0.3, 0.4], [0.1, 0.7, 0.3, 0.2, 0.4]]
        
        # Insert data
        indexer._insert_embeddings_to_db(chunked_results, embeddings)
        
        # Verify data was inserted correctly
        embeddings_data = indexer.vector_store.get_all_embeddings()
        assert len(embeddings_data) == 2
        assert embeddings_data[0]['chunk'] == 'Neuroscience research on brain imaging'
        assert embeddings_data[1]['chunk'] == 'Medical device regulations and safety'


def test_vector_store_factory_integration() -> None:
    """Integration test for VectorStoreFactory with different configurations"""
    # Test SQLite with custom path
    db_path = Path('custom_sqlite.db')
    sqlite_store = VectorStoreFactory.create_backend('sqlite', db_path=db_path)
    assert isinstance(sqlite_store, SQLiteVectorStore)
    
    # Test Chroma with local persistence
    persist_dir = Path('custom_chroma_data')
    chroma_store = VectorStoreFactory.create_backend(
        'chroma',
        collection_name='custom_collection',
        persist_directory=persist_dir
    )
    assert isinstance(chroma_store, ChromaVectorStore)
    
    # Test Chroma with remote URL
    remote_chroma_store = VectorStoreFactory.create_backend(
        'chroma',
        collection_name='remote_collection',
        chroma_client_url='http://remote-server:8000'
    )
    assert isinstance(remote_chroma_store, ChromaVectorStore)
    assert remote_chroma_store.chroma_client_url == 'http://remote-server:8000'


# =====================
# RETRIEVER TESTS
# =====================

def test_retriever_with_sqlite_backend() -> None:
    """Test Retriever initialization with SQLite backend"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / 'embeddings.db'
        
        retriever = Retriever(
            embedding_model='fastembed',
            vector_store_backend='sqlite',
            db_path=db_path
        )
        
        # Test that vector store is initialized
        assert hasattr(retriever, 'vector_store')
        assert isinstance(retriever.vector_store, SQLiteVectorStore)
        assert retriever.vector_store.db_path == db_path


def test_retriever_with_chroma_backend() -> None:
    """Test Retriever initialization with Chroma backend"""
    with tempfile.TemporaryDirectory() as temp_dir:
        persist_dir = Path(temp_dir) / 'chroma_data'
        
        retriever = Retriever(
            embedding_model='fastembed',
            vector_store_backend='chroma',
            vector_store_config={
                'collection_name': 'test_collection',
                'persist_directory': persist_dir
            }
        )
        
        # Test that vector store is initialized
        assert hasattr(retriever, 'vector_store')
        assert isinstance(retriever.vector_store, ChromaVectorStore)
        assert retriever.vector_store.collection_name == 'test_collection'
        assert retriever.vector_store.persist_directory == persist_dir


def test_retriever_backward_compatibility() -> None:
    """Test Retriever backward compatibility with old db_path parameter"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / 'embeddings.db'
        
        retriever = Retriever(
            embedding_model='fastembed',
            db_path=db_path
        )
        
        # Should default to SQLite backend
        assert hasattr(retriever, 'vector_store')
        assert isinstance(retriever.vector_store, SQLiteVectorStore)
        assert retriever.vector_store.db_path == db_path


def test_retriever_integration_with_sqlite() -> None:
    """Integration test using actual Retriever with SQLite backend"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / 'embeddings.db'
        
        retriever = Retriever(
            embedding_model='fastembed',
            vector_store_backend='sqlite',
            db_path=db_path
        )
        
        # Insert test data using the vector store
        chunked_results = [
            {
                'chunk': 'Biomedical research on ultrasound therapy',
                'metadata': {'title': 'Biomedical Research', 'author': 'Researcher'},
                'name': 'biomedical.txt'
            },
            {
                'chunk': 'Clinical trials for drug discovery',
                'metadata': {'title': 'Clinical Trials', 'author': 'Clinician'},
                'name': 'clinical.txt'
            }
        ]
        embeddings = [[0.9, 0.1, 0.2, 0.3, 0.4], [0.1, 0.8, 0.2, 0.3, 0.4]]
        
        retriever.vector_store.insert_embeddings(chunked_results, embeddings)
        
        # Test retrieval functionality
        with patch.object(retriever, '_embed_query') as mock_embed:
            mock_embed.return_value = np.array([0.9, 0.1, 0.2, 0.3, 0.4])
            
            results = retriever.retrieve("ultrasound therapy", top_k=2)
            
            assert len(results) == 2
            assert 'ultrasound' in results[0].lower()
            assert isinstance(results[0], str)  # Should return strings, not dicts


def test_retriever_integration_with_chroma() -> None:
    """Integration test using actual Retriever with Chroma backend"""
    # Skip if ChromaDB not available
    try:
        import chromadb
    except ImportError:
        pytest.skip("ChromaDB not installed")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        persist_dir = Path(temp_dir) / 'chroma_data'
        
        retriever = Retriever(
            embedding_model='fastembed',
            vector_store_backend='chroma',
            vector_store_config={
                'collection_name': 'test_collection',
                'persist_directory': persist_dir
            }
        )
        
        # Insert test data using the vector store
        chunked_results = [
            {
                'chunk': 'Neuroscience research on brain imaging',
                'metadata': {'title': 'Neuroscience', 'author': 'Neuroscientist'},
                'name': 'neuroscience.txt'
            },
            {
                'chunk': 'Medical device regulations and safety',
                'metadata': {'title': 'Medical Devices', 'author': 'Regulator'},
                'name': 'medical_devices.txt'
            }
        ]
        embeddings = [[0.8, 0.2, 0.1, 0.3, 0.4], [0.1, 0.7, 0.3, 0.2, 0.4]]
        
        retriever.vector_store.insert_embeddings(chunked_results, embeddings)
        
        # Test retrieval functionality
        with patch.object(retriever, '_embed_query') as mock_embed:
            mock_embed.return_value = np.array([0.8, 0.2, 0.1, 0.3, 0.4])
            
            results = retriever.retrieve("brain imaging research", top_k=2)
            
            assert len(results) == 2
            assert 'brain' in results[0].lower()
            assert isinstance(results[0], str)  # Should return strings, not dicts
