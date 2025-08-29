# Contributing to RAGToolBox

Thanks for considering a contribution! 🎉 This document explains how to set up your dev environment, run tests, and open a quality PR.

## Ways to Contribute
- Bug reports and reproductions
- Docs fixes and improvements
- Tests and CI improvements
- Features (discuss in an issue first when substantial)

See open issues labeled **good first issue** or **help wanted**.

## Development Setup

### Prereqs

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### Clone and install

```bash
git clone https://github.com/Nick-Nunley/RAGToolBox.git
cd RAGToolBox

# Base deps + dev tools
poetry install --with dev

# Optional extras:
#   - transformers (local LLMs):          -E transformers
#   - chromadb backend:                    -E chromadb
#   - openai embeddings:                   -E openai
#   - NCBI/Entrez (NCBILoader):            -E ncbi
poetry install --with dev -E "transformers chromadb openai ncbi"

poetry shell  # activate virtualenv

```

### Environment variables

Some features will need credentials:

- `HUGGINGFACE_API_KEY`: A HuggingFace access token for interfacing with models through the HuggingFace API.
- `OPENAI_API_KEY`: An OpenAI access token for using OpenAI's embedding system and other features.
- `NCBI_EMAIL`: An institutional email for accessing research articles through the `Entrez` utility.

### Running tests & checking coverage

```bash
# full suite (preferred)
bash tests/Run_tests.sh

# or directly
pytest

# coverage HTML report in htmlcov/
pytest --cov=RAGToolBox --cov-report=term-missing --cov-report=html

```

### Lint checking

```bash
# Lint modules together
pylint RAGToolBox

# Lint tests together but separate from package modules
pylint tests

```

### Project Layout & Building

- Package: `RAGToolBox/`
- Testing: `tests/`
- Dev-helpers: `utils/`
- Package building: `poetry build`

### Commit & PR Guidelines

- Be polite, courteous, and professional in PR conversations.
- Follow clear commit messages (Conventional Commits encouraged, e.g., fix:, feat:, docs:).
- Keep PRs focused and small when possible.
- Ensure CI passes (tests + lint).
- Update docs/README/CHANGELOG when behavior or options change.
- By contributing, you agree your code is licensed under the MIT License.
