"""
RAGToolBox Vector_store module.

Provides an abstract :class:`VectorStoreBackend` and concrete implementations
for SQLite and ChromaDB, plus a small factory for instantiation from a string
identifier. These backends store text chunks, their vector embeddings, and
associated metadata for semantic retrieval.
"""

import logging
import json
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

__all__ = ['VectorStoreBackend', 'SQLiteVectorStore', 'ChromaVectorStore', 'VectorStoreFactory']
logger = logging.getLogger(__name__)

class VectorStoreBackend(ABC):
    """
    Abstract base class for vector storage backends.

    Backends must support initialization, inserting chunk/embedding pairs,
    enumerating all stored embeddings, and deleting the entire collection.
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the vector store.

        Should create any required tables/collections and establish connections
        so the instance is ready to accept inserts/queries.
        """

    @abstractmethod
    def insert_embeddings(
        self, chunked_results: List[Dict[str, Any]], embeddings: List[List[float]]
        ) -> None:
        """
        Insert chunks and their embeddings into the vector store.

        Args:
            chunked_results:
                List of items with at least keys ``'chunk'`` (str) and
                ``'metadata'`` (dict). Additional keys (e.g. ``'name'``) may be
                used by specific backends
            embeddings:
                A list of vectors aligned 1:1 with ``chunked_results``

        Raises:
            RuntimeError: If the write operation fails
            ValueError: If input lengths are mismatched
        """

    @abstractmethod
    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Get all embeddings from the vector store for similarity calculation.

        Returns:
            A list of dictionaries, each containing:
                - ``'chunk'``: str
                - ``'embedding'``: list[float]
                - ``'metadata'``: dict (may be empty)

        Notes:
            Implementations may load all rows into memory; callers should
            consider total collection size when using this method.
        """

    @abstractmethod
    def delete_collection(self) -> None:
        """
        Delete or drop the entire collection/database.

        Intended for test cleanup or rebuild operations.
        """


class SQLiteVectorStore(VectorStoreBackend):
    """
    SQLite-backed vector store.

    Stores each chunk, its embedding, and metadata in a single table
    named ``embeddings`` inside a file-backed SQLite database.

    Attributes:
        db_path: Filesystem path to the SQLite database file
    """

    def __init__(self, db_path: Path):
        """
        Create a SQLite backend bound to ``db_path``.

        Ensures the parent directory exists but does not create tables until
        :meth:`initialize` is called.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """
        Create the SQLite database and ``embeddings`` table if it does not exist.

        Table schema:
            - ``id`` (TEXT PRIMARY KEY): deterministic hash of the chunk text
            - ``chunk`` (TEXT)
            - ``embedding`` (TEXT, JSON-encoded list[float])
            - ``metadata`` (TEXT, JSON-encoded dict)
            - ``source`` (TEXT, optional)
        """
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
        """
        Insert chunk, embedding, and metadata rows into SQLite database.

        Args:
            chunked_results:
                Items with keys ``'chunk'`` (str) and ``'metadata'`` (dict),
                a deterministic SHA-256 of the chunk is used as ``id``
            embeddings:
                Embeddings aligned 1:1 with ``chunked_results``

        Raises:
            RuntimeError: If the database write fails
            ValueError: If input lengths are mismatched
        """
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
        """
        Fetch all rows from the ``embeddings`` table.

        Returns:
            A list of dictionaries with keys ``'chunk'``, ``'embedding'``,
            and ``'metadata'`` (if the database or table does not exist,
            an empty list is returned)
        """
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
        """Delete the SQLite database file if it exists."""
        if self.db_path.exists():
            self.db_path.unlink()


class ChromaVectorStore(VectorStoreBackend):
    """
    ChromaDB-backed vector store (local or remote).

    Supports local ephemeral, local persistent, and remote HTTP clients,
    and manages a single collection identified by ``collection_name``.

    Attributes:
        collection_name:
            Name of the Chroma collection
        persist_directory:
            Directory for local on-disk persistence (optional)
        chroma_client_url:
            Remote server URL for Chroma HTTP client (optional)
        client:
            The underlying Chroma client instance (set by :meth:`initialize`)
        collection:
            The active Chroma collection instance (set by :meth:`initialize`)
    """

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
        """
        Initialize Chroma client and create/get collection.

        Behavior:
            - If ``chroma_client_url`` is provided, uses ``chromadb.HttpClient``.
            - Else if ``persist_directory`` is provided, uses
              ``chromadb.PersistentClient`` with that path.
            - Else uses an in-memory ``chromadb.Client``.

        Raises:
            ImportError: If ``chromadb`` is not installed
            RuntimeError: If client or collection initialization fails
        """
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
        """
        Insert chunks, embeddings, and metadata into Chroma collection.

        Args:
            chunked_results:
                Items with keys ``'chunk'`` (str) and ``'metadata'`` (dict).
                A deterministic SHA-256 of the chunk text is used as the ``id``.
                The ``'name'`` key (if present) is copied to metadata as ``'source'``.
            embeddings:
                Embeddings aligned 1:1 with ``chunked_results``

        Raises:
            RuntimeError: If the collection is not initialized or the add fails
            ValueError: If input lengths are mismatched
        """
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
        """
        Get all chunks/embeddings/metadatas from the Chroma collection.

        Returns:
            A list of dictionaries of the form:
                ``{'chunk': str, 'embedding': list[float], 'metadata': dict}``

        Raises:
            RuntimeError: If the collection is not initialized
        """
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
        """Drop the Chroma collection if it exists."""
        if self.client and self.collection:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.debug("Collection: %s deleted.", self.collection_name)
            except Exception: # pylint: disable=broad-exception-caught
                pass  # Collection might not exist


class VectorStoreFactory:
    """
    Factory for constructing :class:`VectorStoreBackend` instances.

    Usage:
        >>> store = VectorStoreFactory.create_backend("sqlite", db_path=Path("embeddings.db"))
        >>> store.initialize()
    """

    @staticmethod
    def create_backend(backend_type: str, **kwargs) -> VectorStoreBackend:
        """
        Create a vector-store backend by name.

        Args:
            backend_type:
                Backend identifier (case-insensitive). Supported: ``"sqlite"``, ``"chroma"``
            **kwargs:
                Backend-specific options:
                    - ``sqlite``:
                        * ``db_path``: :class:`pathlib.Path` | str (default:
                          ``assets/kb/embeddings/embeddings.db``)
                    - ``chroma``:
                        * ``collection_name``: str (default: ``"rag_collection"``)
                        * ``persist_directory``: :class:`pathlib.Path` | str (optional)
                        * ``chroma_client_url``: str (optional)

        Returns:
            A concrete :class:`VectorStoreBackend` instance

        Raises:
            ValueError: If ``backend_type`` is not supported
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
