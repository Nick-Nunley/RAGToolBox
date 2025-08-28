# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - <Template_date>

### Added

- **Core RAG pipeline components:**
  - Document loading (`BaseLoader` with support for PDFs, HTML, PubMed/PMC webpages, etc.)
  - Text chunking strategies:
    - `SectionAwareChunker`
    - `SlidingWindowChunker`
    - `HierarchicalChunker`
  - Embedding + storage backends:
    - `SQLiteVectorStore`
    - `ChromaVectorStore`
  - Indexing pipeline (`Indexer`, `IndexerConfig`) with parallel embedding support
  - Retriever (`Retriever`) with OpenAI or FastEmbed embedding models
  - Augmenter (`Augmenter`) for LLM integration with customizable prompts
- **CLI entrypoints** for each module:
  - `RAGToolBox.loader`
  - `RAGToolBox.index`
  - `RAGToolBox.retriever`
  - `RAGToolBox.augmenter`
- **Configuration support** with sensible defaults and environment variable handling.
- **Testing framework** with `pytest` and GitHub Actions CI workflow.
- README with installation, quickstart, and CLI usage
- Interactive chat feature with knowledgebase *via* `RAGToolBox.augmenter --chat`

---

[0.1.0]: https://github.com/Nick-Nunley/RAGToolBox/releases/tag/v0.1.0
