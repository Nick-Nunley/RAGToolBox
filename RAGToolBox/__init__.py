"""
RAGToolBox package.

A modular toolkit for building Retrieval-Augmented Generation (RAG) pipelines.
See submodules for functionality:
- loader: fetch/convert sources (HTML, PDF, PubMed/PMC)
- chunk: text chunkers (paragraph/sentence/section/sliding-window/hierarchical)
- index: embedding + vector-store indexing
- retriever: semantic retrieval over stored embeddings
- augmenter: prompt formatting + LLM generation
- vector_store: SQLite/Chroma backends
"""

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("ragtoolbox")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
