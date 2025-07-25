import argparse
import os
import time
import json
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3



class Retriever:
    """Retriever class for retrieving relevant chunks from the knowledge base"""

    def __init__(self, embedding_model: str, db_path: Path = Path('assets/kb/embeddings.db')):
        supported_embedding_models = ['openai', 'fastembed']
        if embedding_model not in supported_embedding_models:
            raise ValueError(f"Unsupported embedding model: {embedding_model}. Embedding model must be one of: {supported_embedding_models}")
        self.embedding_model = embedding_model
        self.db_path = db_path

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
        query = self._embed_query(query = query, max_retries = max_retries)
        df = self._load_db()
        df['embedding'] = df['embedding'].apply(json.loads)
        df['similarity'] = df['embedding'].apply(lambda x: np.dot(x, query))
        df = df.sort_values(by='similarity', ascending=False)
        return df['chunk'].head(top_k).tolist()


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
        default = 'assets/kb/embeddings.db',
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
