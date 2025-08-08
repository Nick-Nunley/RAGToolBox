"""
RAGToolBox Retriever module.

Provides the Retriever class for performing similarity search against
a user query for obtaining semantically similar context.

Additionally, this script provides a CLI entry point for execution as a standalone python module.
"""

import argparse
from typing import List, Optional
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from RAGToolBox.embeddings import Embeddings
from RAGToolBox.vector_store import VectorStoreFactory



class Retriever:
    """Retriever class for retrieving relevant chunks from the knowledge base"""

    def __init__(self, embedding_model: str,
                 vector_store_backend: str = 'sqlite',
                 vector_store_config: Optional[dict] = None,
                 db_path: Path = Path('assets/kb/embeddings/embeddings.db')):
        Embeddings.validate_embedding_model(embedding_model)
        self.embedding_model = embedding_model
        self.db_path = db_path

        # Initialize vector store backend
        self.vector_store_config = vector_store_config or {}
        if vector_store_backend == 'sqlite':
            # For SQLite, use the db_path to determine vector store path
            self.vector_store_config['db_path'] = self.db_path

        self.vector_store = VectorStoreFactory.create_backend(
            vector_store_backend,
            **self.vector_store_config
        )
        self.vector_store.initialize()

    def _load_db(self) -> pd.DataFrame:
        """Method to load the database into a pandas dataframe"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM embeddings", conn)
        conn.close()
        return df

    def _embed_query(self, query: str, max_retries: int = 5) -> List[float]:
        """Method to embed the query using the embedding model"""
        return Embeddings.embed_one(self.embedding_model, query, max_retries)

    def retrieve(self, query: str, top_k: int = 10, max_retries: int = 5) -> List[str]:
        """Method to retrieve the top k results from the knowledge base"""
        query_embedding = self._embed_query(query=query, max_retries=max_retries)

        # Get all embeddings from vector store
        embeddings_data = self.vector_store.get_all_embeddings()

        if not embeddings_data:
            return []

        # Calculate similarities
        similarities = []
        for item in embeddings_data:
            embedding = np.array(item['embedding'])
            similarity = np.dot(embedding, query_embedding)
            similarities.append((similarity, item['chunk'], item['metadata']))

        # Sort by similarity and return top_k chunks
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [
            {'data': chunk, 'metadata': metadata}
            for _, chunk, metadata in similarities[:top_k]
            ]


if __name__ == "__main__":

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

    args = parser.parse_args()

    reriever = Retriever(
        embedding_model = args.embedding_model,
        db_path = args.db_path
        )

    context = reriever.retrieve(args.query, args.top_k, args.max_retries)
