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

    def _load_db(self) -> pd.DataFrame:
        """Method to load the database into a pandas dataframe"""
        logger.debug("Opening SQLite DB at %s", self.db_path)
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM embeddings", conn)
            logger.info("Loaded %d rows from DB", len(df))
            return df
        except sqlite3.Error:
            logger.exception("Failed to load DB at %s", self.db_path)
            raise

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

    import os
    from RAGToolBox.logging import setup_logging, LoggingConfig

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

    parser.add_argument(
        '--log-level',
        default = os.getenv('RAGTB_LOG_LEVEL', 'INFO'),
        choices = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        help = 'Console logging level (default: INFO)'
        )

    parser.add_argument(
        '--log-file',
        default = os.getenv('RAGTB_LOG_FILE'),
        help = 'If set, write detailed logs to this file (rotating)'
        )

    parser.add_argument(
        '--log-file-level',
        default = os.getenv('RAGTB_LOG_FILE_LEVEL', 'DEBUG'),
        choices = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
        help = 'File log level if --log-file is provided (default: DEBUG)'
        )

    args = parser.parse_args()

    setup_logging(LoggingConfig(
        console_level = args.log_level,
        log_file = args.log_file,
        file_level = args.log_file_level
        ))
    logger.debug("CLI args: %s", vars(args))

    reriever = Retriever(
        embedding_model = args.embedding_model,
        db_path = args.db_path
        )

    context = reriever.retrieve(args.query, args.top_k, args.max_retries)
