"""
RAGToolBox types module.

Centralized TypedDict definitions used throughout RAGToolBox to ensure
consistent structure for chunks, embeddings, and retrieval results.

These lightweight type definitions help enforce clear boundaries between:
- `ChunkEntry`: inputs to vector stores before embedding
- `StoredEmbedding`: records returned by vector stores (includes embeddings)
- `RetrievedChunk`: results returned by retrievers (no embeddings, user-facing)
"""

from __future__ import annotations
from typing import TypedDict, Any, List

__all__ = ['ChunkEntry', 'StoredEmbedding', 'RetrievedChunk']

class ChunkEntry(TypedDict):
    """
    Input to vector stores before embedding.

    Represents the smallest unit of text content prepared for embedding.

    Keys:
        chunk:
            The raw text chunk to be embedded
        metadata:
            Arbitrary metadata associated with the chunk (e.g., source file,
            position, section header, or tags)
    """
    chunk: str
    metadata: dict[str, Any]

class StoredEmbedding(TypedDict):
    """
    Record returned by vector stores.

    Represents a chunk after it has been embedded and persisted in a
    vector store.

    Keys:
        chunk:
            The original text chunk content
        embedding:
            The numeric embedding vector corresponding to the chunk
        metadata:
            Arbitrary metadata associated with the chunk (e.g., source file,
            position, section header, or tags)
    """
    chunk: str
    embedding: List[float]
    metadata: dict[str, Any]

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
