"""
RAGToolBox Vector_store module.

Provides the VectorStore classes for handling and interfacing with various
vector storage formats (e.g. SQLite and Chromadb).
"""

import logging
import json
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStoreBackend(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store (create tables, collections, etc.)."""

    @abstractmethod
    def insert_embeddings(
        self, chunked_results: List[Dict[str, Any]], embeddings: List[List[float]]
        ) -> None:
        """Insert chunks and their embeddings into the vector store."""

    @abstractmethod
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings from the vector store for similarity calculation."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection/database."""


class SQLiteVectorStore(VectorStoreBackend):
    """SQLite-based vector store backend (current implementation)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Create the SQLite database and table if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                chunk TEXT,
                embedding TEXT,
                metadata TEXT,
                source TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def insert_embeddings(
        self, chunked_results: List[Dict[str, Any]], embeddings: List[List[float]]
        ) -> None:
        """Insert chunk, embedding, and metadata into SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for entry, embedding in zip(chunked_results, embeddings):
            chunk = entry['chunk']
            metadata = entry['metadata']
            hash_id = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
            source = metadata.get('source', None)

            cursor.execute('''
                INSERT OR REPLACE INTO embeddings (id, chunk, embedding, metadata, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (hash_id, chunk, json.dumps(embedding), json.dumps(metadata), source))

        conn.commit()
        conn.close()
        logger.info("Embddings inserted successfully")

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings from the SQLite database."""
        if not Path(self.db_path).exists():
            logger.warning(
                'Warning: no such database found at %s. Returning emtpy list...',
                self.db_path
                )
            return []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT chunk, embedding, metadata FROM embeddings')
        except sqlite3.OperationalError:
            logger.warning(
                "Warning: no 'embeddings' table in %s. Returning empty list...",
                self.db_path
                )
            conn.close()
            return []
        cursor.execute('SELECT chunk, embedding, metadata FROM embeddings')
        results = cursor.fetchall()
        conn.close()

        return [
            {
                'chunk': chunk,
                'embedding': json.loads(embedding_str),
                'metadata': json.loads(metadata_str) if metadata_str else {}
            }
            for chunk, embedding_str, metadata_str in results
        ]

    def delete_collection(self) -> None:
        """Delete the SQLite database file."""
        if self.db_path.exists():
            self.db_path.unlink()


class ChromaVectorStore(VectorStoreBackend):
    """Chroma-based vector store backend for remote/local vector database."""

    def __init__(
        self, collection_name: str = "rag_collection", persist_directory: Optional[Path] = None,
        chroma_client_url: Optional[str] = None
        ):
        """
        Initialize Chroma vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Local directory to persist data (for local Chroma)
            chroma_client_url: URL for remote Chroma server (e.g., "http://localhost:8000")
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chroma_client_url = chroma_client_url
        self.client = None
        self.collection = None

    def initialize(self) -> None:
        """Initialize Chroma client and create/get collection."""
        try:
            import chromadb
        except ImportError as e:
            err = "ChromaDB is not installed. Install it with: pip install chromadb"
            logger.error(err, exc_info=True)
            raise ImportError(err) from e

        if self.chroma_client_url:
            # Remote Chroma server
            logger.debug("Connecting to remote Chroma server at %s", self.chroma_client_url)
            self.client = chromadb.HttpClient(host=self.chroma_client_url)
        else:
            # Local Chroma with optional persistence
            if self.persist_directory:
                logger.debug(
                    "Connecting to persistence Chroma server at %s",
                    self.persist_directory
                    )
                self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            else:
                logger.debug("Connecting to local Chroma server with 'chromadb.Client()'")
                self.client = chromadb.Client()

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.debug("Chroma collection: %s found.", self.collection_name)
        except Exception: # pylint: disable=broad-exception-caught
            logger.debug(
                "Chroma collection: %s not found. Creating new collection.",
                self.collection_name
                )
            self.collection = self.client.create_collection(name=self.collection_name)

    def insert_embeddings(
        self, chunked_results: List[Dict[str, Any]], embeddings: List[List[float]]
        ) -> None:
        """Insert chunks and embeddings into Chroma collection."""
        if not self.collection:
            err = "Chroma collection not initialized. Call initialize() first."
            logger.error(err)
            raise RuntimeError(err)

        # Prepare data for Chroma
        ids = []
        documents = []
        metadatas = []

        for _, (entry, _) in enumerate(zip(chunked_results, embeddings)):
            chunk_id = hashlib.sha256(entry['chunk'].encode('utf-8')).hexdigest()
            ids.append(chunk_id)
            documents.append(entry['chunk'])

            # Prepare metadata
            metadata = entry['metadata'].copy()
            metadata['source'] = entry.get('name', 'unknown')
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info("Embddings inserted successfully")

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings from the Chroma collection."""
        if not self.collection:
            err = "Chroma collection not initialized. Call initialize() first."
            logger.error(err)
            raise RuntimeError(err)

        results = self.collection.get(
            include=['documents', 'embeddings', 'metadatas']
        )

        return [
            {
                'chunk': results['documents'][i],
                'embedding': results['embeddings'][i],
                'metadata': results['metadatas'][i] if results['metadatas'] else {}
            }
            for i in range(len(results['documents']))
        ]

    def delete_collection(self) -> None:
        """Delete the Chroma collection."""
        if self.client and self.collection:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.debug("Collection: %s deleted.", self.collection_name)
            except Exception: # pylint: disable=broad-exception-caught
                pass  # Collection might not exist


class VectorStoreFactory:
    """Factory class for creating vector store backends."""

    @staticmethod
    def create_backend(backend_type: str, **kwargs) -> VectorStoreBackend:
        """
        Create a vector store backend based on the specified type.

        Args:
            backend_type: Type of backend ('sqlite', 'chroma')
            **kwargs: Additional arguments for the specific backend

        Returns:
            VectorStoreBackend instance
        """
        if backend_type.lower() == 'sqlite':
            db_path = kwargs.get('db_path', Path('assets/kb/embeddings/embeddings.db'))
            logger.info("SQLite vector db detected at %s.", db_path)
            return SQLiteVectorStore(db_path)

        if backend_type.lower() == 'chroma':
            collection_name = kwargs.get('collection_name', 'rag_collection')
            logger.info("Chroma vector db collection: %s detected", collection_name)
            persist_directory = kwargs.get('persist_directory')
            chroma_client_url = kwargs.get('chroma_client_url')

            if persist_directory:
                persist_directory = Path(persist_directory)

            return ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                chroma_client_url=chroma_client_url
            )

        err = (
            f"Unsupported vector store backend: {backend_type}. "
            f"Supported backends: sqlite, chroma"
            )
        logger.error(err)
        raise ValueError(err)
