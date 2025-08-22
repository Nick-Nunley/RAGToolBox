"""
RAGToolBox embeddings module.

Provides utilities for validating embedding model names and performing text
embeddings via either OpenAI or FastEmbed backends.

Environment:
    OPENAI_API_KEY: API key used when embedding with the OpenAI backend

See Also:
    - docs on configuring extras/optional dependencies for OpenAI and FastEmbed
"""

from __future__ import annotations
import os
import logging
import time
from typing import List

__all__ = ['Embeddings']
logger = logging.getLogger(__name__)

class Embeddings():
    """
    Embeddings class for handling embedding model validation and embedding.

    Attributes:
        SUPPORTED_EMBEDDING_MODELS: Tuple of allowed identifiers
        OPENAI_EMBED_MODEL: Default OpenAI embedding model name
    """

    SUPPORTED_EMBEDDING_MODELS = ("openai", "fastembed")
    OPENAI_EMBED_MODEL = "text-embedding-3-small"

    @classmethod
    def supported_models(cls) -> tuple[str, ...]:
        """Return the currently supported embedding backend names."""
        return cls.SUPPORTED_EMBEDDING_MODELS

    @staticmethod
    def validate_embedding_model(name: str) -> None:
        """
        Validate an embedding backend name.

        Args:
            name: The name of the embedding backend to validate

        Raises:
            ValueError: If ``name`` is not in
                :py:attr:`Embeddings.SUPPORTED_EMBEDDING_MODELS`

        Examples:
            >>> Embeddings.validate_embedding_model("openai")
            >>> Embeddings.validate_embedding_model("bogus")
            Traceback (most recent call last):
            ...
            ValueError: Unsupported embedding model: bogus. Choose one of: ('openai', 'fastembed')
        """
        if name not in Embeddings.SUPPORTED_EMBEDDING_MODELS:
            err = (
                f"Unsupported embedding model: {name}. "
                f"Embedding model must be one of: {list(Embeddings.SUPPORTED_EMBEDDING_MODELS)}"
                )
            logger.error(err)
            raise ValueError(err)

    @staticmethod
    def _embed_openai(texts: List[str], max_retries: int) -> List[List[float]]:
        """Helper method to embed text using openai API model."""
        try:
            import openai  # local import so package users without openai aren’t penalized
        except ImportError as e:
            err = (
                "openai package is required. "
                "Install with: pip install openai"
                )
            logger.error(err, exc_info=True)
            raise ImportError(err ) from e
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(
                    input=texts,
                    model=Embeddings.OPENAI_EMBED_MODEL
                    )
                return [d.embedding for d in resp.data]
            except openai.RateLimitError:
                time.sleep(2 ** attempt)
        err = "Failed to embed after multiple retries due to rate limits."
        logger.error(err)
        raise RuntimeError(err)

    @staticmethod
    def _embed_fastembed(texts: List[str]) -> List[List[float]]:
        """Helper method to embed text using fastembed."""
        # Normalize to (n, d) float32
        from fastembed import TextEmbedding
        model = TextEmbedding()
        out = [list(model.embed(t))[0].tolist() for t in texts]
        return out

    @staticmethod
    def embed_texts(model_name: str, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """
        Embed a list of texts using the selected backend.

        Args:
            model_name: Name of the embedding backend. See
                :py:meth:`Embeddings.supported_models` for the current list
            texts: List of strings to embed
            max_retries: Max retry attempts on rate limits (remote backend only)

        Returns:
            List of embeddings aligned to ``texts``

        Raises:
            ValueError: If ``model_name`` is not supported
            ImportError: If the chosen backend package is not installed
            RuntimeError: If the backend call fails
        """
        if model_name == "openai":
            return Embeddings._embed_openai(texts, max_retries)
        if model_name == "fastembed":
            return Embeddings._embed_fastembed(texts)
        err = f"Embedding model '{model_name}' not supported."
        logger.error(err)
        raise ValueError(err)

    @staticmethod
    def embed_one(model_name: str, text: str, max_retries: int = 5) -> List[float]:
        """
        Embed a single string.

        Args:
            model_name: Name of the embedding backend
            text: Single string to embed
            max_retries: Max retry attempts on rate limits (remote backend only)

        Returns:
            A single embedding vector for ``text``
        """
        return Embeddings.embed_texts(model_name, [text], max_retries)[0]
