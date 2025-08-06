# RAGToolBox

[![CI](https://github.com/Nick-Nunley/RAGToolBox/actions/workflows/CI.yml/badge.svg)](https://github.com/Nick-Nunley/RAGToolBox/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/Nick-Nunley/RAGToolBox/graph/badge.svg?token=MUXZWZMZV0)](https://codecov.io/github/Nick-Nunley/RAGToolBox)

**RAGToolBox** is a modular, extensible Python package for building Retrieval-Augmented Generation (RAG) pipelines. It provides end-to-end components for:

* **Loading** content (local files, web pages, PubMed/PMC articles)
* **Chunking** text (paragraphs, sentences, sliding windows, section‑aware, hierarchical)
* **Storing** embeddings (SQLite and ChromaVectorStore backends)
* **Indexing** content with parallel embedding and metadata parsing
* **Retrieval** *via* OpenAI or FastEmbed embeddings and similarity search
* **Augmentation** using customizable prompts and LLMs (local or Hugging Face)

---

## Table of Contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
    * [Loading Documents](#loading-documents)
    * [Indexing Pipeline](#indexing-pipeline)
    * [Retrieval & Augmentation](#retrieval--augmentation)
3. [CLI Usage](#cli-usage)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Contributing](#contributing)
7. [License](#license)

---

## Installation

Install the latest release from PyPI:

```bash
pip install RAGToolBox
```

Or install from source:

```bash
git clone https://github.com/Nick-Nunley/RAGToolBox.git
cd RAGToolBox
pip install .
```

### Prerequisites

* Python 3.10+
* Core dependencies (see also `pyproject.toml`):
    * langchain>=0.3.26
    * openai>=1.95.1
    * faiss-cpu>=1.11.0
    * tiktoken>=0.9.0
    * streamlit>=1.46.1
    * requests>=2.32.4
    * pdfplumber>=0.11.7
    * html2text>=2025.4.15
    * pytest>=8.4.1
    * bio>=1.8.0
    * bs4>=0.0.2
    * readability-lxml>=0.8.4.1
    * nltk>=3.9.1
    * fastembed>=0.7.1
    * huggingface_hub>=0.33.5
    * chromadb>=1.0.15
* Optional dependencies (for local model inference):
    * torch>=2.7.1
    * transformers>=4.30.0

---

## Quickstart

### Loading Documents

Use the built‑in factory to detect and process various formats:

```python
from RAGToolBox.loader import BaseLoader

# Raw bytes fetched externally (e.g., via requests or open file)
raw_bytes = open("example.pdf", "rb").read()
LoaderClass = BaseLoader.detect_loader("example.pdf", raw_bytes)
loader = LoaderClass("example.pdf", "assets/kb")
loader.raw_content = raw_bytes
loader.process()
```

### Indexing Pipeline

Chunk, embed, and store your KB in one pipeline:

```python
from pathlib import Path
from RAGToolBox.chunk import SectionAwareChunker, SlidingWindowChunker, HierarchicalChunker
from RAGToolBox.index import Indexer

chunker = HierarchicalChunker([
    SectionAwareChunker(max_chunk_size=1000, overlap=200),
    SlidingWindowChunker(window_size=1000, overlap=200)
])
indexer = Indexer(
    chunker=chunker,
    embedding_model="fastembed",          # or 'openai'
    vector_store_backend="sqlite",       # or 'chroma'
    vector_store_config={"db_path": Path("assets/kb/embeddings/embeddings.db")}
)
indexer.index(chunked_results)
```

### Retrieval & Augmentation

Retrieve relevant chunks and generate an LLM response:

```python
from pathlib import Path
from RAGToolBox.retriever import Retriever
from RAGToolBox.augmenter import Augmenter

# Example query
user_query = "What is RAG?"

# Initialize retriever
retriever = Retriever(
    embedding_model="fastembed",
    db_path=Path("assets/kb/embeddings/embeddings.db")
)

# Perform retrieval
contexts = retriever.retrieve(user_query, top_k=5)

# Initialize augmenter
augmenter = Augmenter(
    model_name="google/gemma-2-2b-it",
    prompt_type="default",
    api_key="${HUGGINGFACE_API_KEY}",
    use_local=False
)

# Generate a response with sources
result = augmenter.generate_response_with_sources(
    query=user_query,
    retrieved_chunks=contexts,
    temperature=0.25,
    max_new_tokens=200
)
print(result["response"])
```

---

## CLI Usage

Each module exposes a CLI entrypoint. Use `-h` for details:

```bash
# Load documents from URLs or files
python -m RAGToolBox.loader https://example.com/article.pdf --output-dir assets/kb

# Build index (chunk + embed)
python -m RAGToolBox.index --kb-dir assets/kb --embedding-model fastembed --vector-store sqlite

# Retrieve top-10 chunks\ npython -m RAGToolBox.retriever --query "Explain RAG" --embedding-model openai

# Augment with LLM
python -m RAGToolBox.augmenter "What is RAG?" --sources

# For a concise one-liner answer
python -m RAGToolBox.augmenter "What is RAG?" --prompt-type concise
```

---

## Testing

Run the full test suite:

```bash
bash tests/Run_tests.sh
```

Continuous integration is configured via GitHub Actions (see `.github/workflows/ci.yml`).

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to your fork and open a PR

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing Guidelines](CONTRIBUTING.md).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
