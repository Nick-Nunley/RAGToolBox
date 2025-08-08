"""
RAGToolBox embeddings module

Provides methods for validating embedding-model name inputs and
performing text embedding.
"""

from __future__ import annotations
import os
import time
from typing import List

class Embeddings():
    """Embeddings class for handling embedding model validation and embedding"""

    SUPPORTED_EMBEDDING_MODELS = ("openai", "fastembed")
    OPENAI_EMBED_MODEL = "text-embedding-3-small"

    @staticmethod
    def validate_embedding_model(name: str) -> None:
        """Method to validate embedding_model inputs"""
        if name not in Embeddings.SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model: {name}. "
                f"Embedding model must be one of: {list(Embeddings.SUPPORTED_EMBEDDING_MODELS)}"
            )

    @staticmethod
    def _embed_openai(texts: List[str], max_retries: int) -> List[List[float]]:
        """Helper method to embed text using openai API model"""
        import openai  # local import so package users without openai aren’t penalized
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(input=texts, model=Embeddings.OPENAI_EMBED_MODEL)
                return [d.embedding for d in resp.data]
            except openai.RateLimitError:
                time.sleep(2 ** attempt)
        raise RuntimeError("Failed to embed after multiple retries due to rate limits.")

    @staticmethod
    def _embed_fastembed(texts: List[str]) -> List[List[float]]:
        """Helper method to embed text using fastembed"""
        # Normalize to (n, d) float32
        from fastembed import TextEmbedding
        model = TextEmbedding()
        out = [list(model.embed(t))[0].tolist() for t in texts]
        return out

    @staticmethod
    def embed_texts(model_name: str, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Method to embed text from a list of strings input"""
        if model_name == "openai":
            return Embeddings._embed_openai(texts, max_retries)
        if model_name == "fastembed":
            return Embeddings._embed_fastembed(texts)
        raise ValueError(f"Embedding model '{model_name}' not supported.")

    @staticmethod
    def embed_one(model_name: str, text: str, max_retries: int = 5) -> List[float]:
        """Method to embed text from a single string input"""
        return Embeddings.embed_texts(model_name, [text], max_retries)[0]
