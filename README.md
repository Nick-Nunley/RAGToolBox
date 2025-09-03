# RAGToolBox

[![CI](https://github.com/Nick-Nunley/RAGToolBox/actions/workflows/CI.yml/badge.svg)](https://github.com/Nick-Nunley/RAGToolBox/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/Nick-Nunley/RAGToolBox/graph/badge.svg?token=MUXZWZMZV0)](https://codecov.io/github/Nick-Nunley/RAGToolBox)

**RAGToolBox** is a modular, extensible Python package for building Retrieval-Augmented Generation (RAG) pipelines. It provides end-to-end components for:

* **Loading** content (local files, web pages, PubMed/PMC articles)
* **Chunking** text (paragraphs, sentences, sliding windows, section‑aware, hierarchical)
* **Storing** embeddings (SQLite and optional ChromaVectorStore backends)
* **Indexing** content with parallel embedding and metadata parsing
* **Retrieval** *via* FastEmbed or (optional) openai embeddings and similarity search
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
pip install ragtoolbox

# with optional extras
pip install "ragtoolbox[transformers,chromadb,openai,ncbi]"
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
    * numpy
    * requests
    * pyyaml
    * pdfplumber
    * html2text
    * pytest
    * beautifulsoup4
    * readability-lxml
    * nltk
    * fastembed
    * huggingface_hub
* Optional dependencies (for extended capabilities):
    * openai
    * chromadb
    * torch
    * transformers
    * biopython

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
from RAGToolBox.index import Indexer, IndexerConfig

# Directory containing .txt files
kb_dir = Path("assets/kb")

chunker = HierarchicalChunker([
    SectionAwareChunker(max_chunk_size=1000, overlap=200),
    SlidingWindowChunker(window_size=1000, overlap=200)
])
indexer = Indexer(
    chunker=chunker,
    embedding_model="fastembed",
    config = IndexerConfig(
        vector_store_backend="sqlite",
        vector_store_config={"db_path": Path("assets/kb/embeddings/embeddings.db")}
    )
)

# Read documents from files
documents = []
txt_files = list(kb_dir.glob('*.txt'))
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        documents.append((file_path.name, f.read()))

# Process documents and create chunked_results
chunked_results = []
for name, text in documents:
    _, metadata, chunks = indexer.chunk((name, text))
    for chunk in chunks:
        chunked_results.append({
            'name': name,
            'metadata': metadata,
            'chunk': chunk
        })

# Index the chunked results into a vector db
indexer.index(chunked_results)
```

### Retrieval & Augmentation

Retrieve relevant chunks and generate an LLM response:

```python
from pathlib import Path
from RAGToolBox.retriever import Retriever, RetrievalConfig
from RAGToolBox.augmenter import Augmenter, GenerationConfig

# Example query
user_query = "What is RAG?"

# Initialize retriever
retriever = Retriever(
    embedding_model="fastembed",
    db_path=Path("assets/kb/embeddings/embeddings.db")
)

# Perform retrieval
contexts = retriever.retrieve(user_query, RetrievalConfig(top_k=5))

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
    gen_config=GenerationConfig(temperature=0.25, max_new_tokens=200)
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

# Retrieve top-10 chunks
python -m RAGToolBox.retriever --query "Explain RAG" --embedding-model openai

# Augment with LLM
python -m RAGToolBox.augmenter "What is RAG?" --sources

# For a concise one-liner answer
python -m RAGToolBox.augmenter "What is RAG?" --prompt-type concise

# Interactively chat with your knowledgebase
python -m RAGToolBox.augmenter --chat
```

---

## Testing

Run the full test suite:

```bash
bash tests/Run_tests.sh
```

Continuous integration is configured *via* GitHub Actions (see `.github/workflows/CI.yml`).

---

## Note

**RAGToolBox** is in active development. The current release (v0.1.0) is an unstable preview. APIs and behavior may change, and some features may not yet be fully tested.

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to your fork and open a PR

Please follow the [Contributing Guidelines](CONTRIBUTING.md).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
