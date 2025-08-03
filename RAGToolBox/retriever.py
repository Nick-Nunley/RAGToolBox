import argparse
import os
import time
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
from RAGToolBox.vector_store import VectorStoreFactory, VectorStoreBackend



class Retriever:
    """Retriever class for retrieving relevant chunks from the knowledge base"""

    def __init__(self, embedding_model: str, 
                 vector_store_backend: str = 'sqlite', 
                 vector_store_config: Optional[dict] = None,
                 db_path: Path = Path('assets/kb/embeddings/embeddings.db')):
        supported_embedding_models = ['openai', 'fastembed']
        if embedding_model not in supported_embedding_models:
            raise ValueError(f"Unsupported embedding model: {embedding_model}. Embedding model must be one of: {supported_embedding_models}")
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

    def _embed_query(self, query: str, max_retries: int = 5) -> np.ndarray:
        """Method to embed the query using the embedding model"""
        if self.embedding_model == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = "text-embedding-3-small"
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        input=[query],
                        model=model
                    )
                    return response.data[0].embedding
                except openai.RateLimitError as e:
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            raise RuntimeError("Failed to embed after multiple retries due to rate limits.")
        elif self.embedding_model == "fastembed":
            from fastembed import TextEmbedding
            model = TextEmbedding()
            embedding = list(model.embed(query))[0]
            return embedding
        raise ValueError(f"Embedding model '{self.embedding_model}' not supported.")

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
        return [{'data': chunk, 'metadata': metadata} for _, chunk, metadata in similarities[:top_k]]


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

    args = parser.parse_args()

    reriever = Retriever(
        embedding_model = args.embedding_model,
        db_path = args.db_path
        )

    query = args.query
    embedding = reriever._embed_query(query)
