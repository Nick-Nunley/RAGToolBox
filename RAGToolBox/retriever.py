"""
RAGToolBox retriever module.

Provides :class:`Retriever` for semantic search over a vector store, returning
the most similar chunks for a user query. Designed to work with the embeddings
produced by the indexing pipeline, and with pluggable vector-store backends.

A small CLI is also provided when running the module as ``python -m RAGToolBox.retriever``.
"""

import argparse
import logging
from typing import List, Optional, TypedDict, Any
from pathlib import Path
import numpy as np
from RAGToolBox.embeddings import Embeddings
from RAGToolBox.vector_store import VectorStoreFactory

__all__ = ['RetrievedChunk', 'Retriever']
logger = logging.getLogger(__name__)

class RetrievedChunk(TypedDict):
    """
    A retrieved result item.

    Keys:
        data:
            The text content of the chunk
        metadata:
            Arbitrary metadata captured during indexing (e.g., source filename,
            section headers, custom tags)
    """
    data: str
    metadata: dict[str, Any]

class Retriever:
    """
    Retriever class for retrieving relevant chunks from the knowledge base via vector similarity.

    The retriever embeds the query with the configured backend and computes
    similarity against stored embeddings in the selected vector store, returning
    the top-k matches.

    Attributes:
        embedding_model:
            Name of the embedding backend (see :py:meth:`Embeddings.supported_models`)
        db_path:
            Path to the local SQLite embeddings DB (used when ``vector_store_backend='sqlite'``)
        vector_store:
            The instantiated vector store backend created via :class:`VectorStoreFactory`
        vector_store_config:
            Backend-specific configuration dictionary passed to the factory
    """

    def __init__(self, embedding_model: str,
                 vector_store_backend: str = 'sqlite',
                 vector_store_config: Optional[dict] = None,
                 db_path: Path = Path('assets/kb/embeddings/embeddings.db')):
        """
        Initializes an instance of :class:`Retriever`.

        Args:
            embedding_model:
                Embedding backend identifier
                (validated by :func:`Embeddings.validate_embedding_model`)
            vector_store_backend:
                Vector store backend name (e.g., ``"sqlite"``, ``"chroma"``)
            vector_store_config:
                Optional backend-specific configuration passed to
                :class:`VectorStoreFactory.create_backend`
            db_path:
                Path to the embeddings database used by the SQLite backend

        Behavior:
            If ``vector_store_backend == "sqlite"``, ``db_path`` is injected into
            ``vector_store_config`` automatically.

        Raises:
            ValueError:
                If ``embedding_model`` is not supported
            RuntimeError:
                If the vector store backend fails to initialize
        """

        logger.debug("Initializing Retriever with model=%s, backend=%s, db_path=%s",
                     embedding_model, vector_store_backend, db_path)
        Embeddings.validate_embedding_model(embedding_model)
        logger.info("Embedding model '%s' validated", embedding_model)
        self.embedding_model = embedding_model
        self.db_path = db_path

        # Initialize vector store backend
        self.vector_store_config = vector_store_config or {}
        if vector_store_backend == 'sqlite':
            # For SQLite, use the db_path to determine vector store path
            self.vector_store_config['db_path'] = self.db_path
            logger.debug("SQLite backend detected, db_path set to %s", self.db_path)

        self.vector_store = VectorStoreFactory.create_backend(
            vector_store_backend,
            **self.vector_store_config
        )
        logger.info("Vector store backend '%s' created", vector_store_backend)
        self.vector_store.initialize()
        logger.info("Vector store initialized successfully")

    def _embed_query(self, query: str, max_retries: int = 5) -> List[float]:
        """Method to embed the query using the embedding model"""
        logger.debug("Embedding query (len=%d) with model=%s, max_retries=%d",
                    len(query), self.embedding_model, max_retries)
        vec = Embeddings.embed_one(self.embedding_model, query, max_retries)
        logger.debug("Query embedding length=%d", len(vec))
        return vec

    def retrieve(self, query: str, top_k: int = 10, max_retries: int = 5) -> List[RetrievedChunk]:
        """
        Return the top-``k`` most similar chunks to ``query``.

        This method computes similarity between the query embedding and all stored
        embeddings, sorts by descending similarity, and returns the highest-scoring
        results along with their metadata.

        Note:
            Similarity is computed as a plain dot product. If your stored
            embeddings are not already normalized, scores will be scale-dependent.

        Args:
            query:
                The natural-language query
            top_k:
                Maximum number of results to return
            max_retries:
                Maximum retry attempts for rate-limited remote backends during
                query embedding

        Returns:
            A list of :class:`RetrievedChunk` dictionaries (length ``<= top_k``),
            when the vector store is empty, an empty list is returned

        Raises:
            ImportError:
                If the chosen embedding backend package is not installed
            RuntimeError:
                If embedding the query fails after retries
        """

        logger.info("Retrieve called: top_k=%d", top_k)
        query_embedding = self._embed_query(query=query, max_retries=max_retries)

        embeddings_data = self.vector_store.get_all_embeddings()
        n = len(embeddings_data)
        if not n:
            logger.warning("Vector store empty; returning no results")
            return []

        logger.debug("Computing similarities against %d embeddings", n)
        similarities = []
        for item in embeddings_data:
            embedding = np.array(item['embedding'])
            similarity = np.dot(embedding, query_embedding)
            similarities.append((similarity, item['chunk'], item['metadata']))

        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [{'data': c, 'metadata': m} for _, c, m in similarities[:top_k]]
        logger.info("Retrieved %d results (requested top_k=%d)", len(results), top_k)

        if logger.isEnabledFor(logging.DEBUG) and results:
            logger.debug("Top similarity=%.4f preview=%r",
                        similarities[0][0], results[0]['data'][:80])

        return results


if __name__ == "__main__":

    from RAGToolBox.logging import RAGTBLogger

    parser = argparse.ArgumentParser(description="Retriever for the knowledge base")

    parser.add_argument(
        '--query',
        '-q',
        required=True,
        type = str,
        help = 'User query to use for retrieval from knowledge base'
        )

    parser.add_argument(
        '--embedding-model',
        '-e',
        default = 'fastembed',
        type = str,
        help = 'Embedding model to use'
        )

    parser.add_argument(
        '--db-path',
        '-d',
        default = 'assets/kb/embeddings/embeddings.db',
        type = Path,
        help = 'Path to the database'
        )

    parser.add_argument(
        '--top-k',
        default = 10,
        type = int,
        help = 'Number of similar chunks to retrieve'
        )

    parser.add_argument(
        '--max-retries',
        default = 5,
        type = int,
        help = 'Number of times to tries to attempt reaching remote embedding model'
        )

    RAGTBLogger.add_logging_args(parser=parser)

    args = parser.parse_args()

    RAGTBLogger.configure_logging_from_args(args=args)
    logger.debug("CLI args: %s", vars(args))

    retriever = Retriever(
        embedding_model = args.embedding_model,
        db_path = args.db_path
        )

    context = retriever.retrieve(args.query, args.top_k, args.max_retries)
