"""
RAGToolBox Retriever module.

Provides the Retriever class for performing similarity search against
a user query for obtaining semantically similar context.

Additionally, this script provides a CLI entry point for execution as a standalone python module.
"""

import argparse
import logging
from typing import List, Optional
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from RAGToolBox.embeddings import Embeddings
from RAGToolBox.vector_store import VectorStoreFactory

logger = logging.getLogger(__name__)

class Retriever:
    """Retriever class for retrieving relevant chunks from the knowledge base"""

    def __init__(self, embedding_model: str,
                 vector_store_backend: str = 'sqlite',
                 vector_store_config: Optional[dict] = None,
                 db_path: Path = Path('assets/kb/embeddings/embeddings.db')):
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

    def retrieve(self, query: str, top_k: int = 10, max_retries: int = 5) -> List[str]:
        """Method to retrieve the top k results from the knowledge base"""
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

    reriever = Retriever(
        embedding_model = args.embedding_model,
        db_path = args.db_path
        )

    context = reriever.retrieve(args.query, args.top_k, args.max_retries)
