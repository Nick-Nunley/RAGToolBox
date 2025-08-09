"""Tests associated with Embeddings module"""
# pylint: disable=protected-access

from __future__ import annotations
import sys
import types
from typing import List
import numpy as np
import pytest
from unittest.mock import Mock, patch

from RAGToolBox.embeddings import Embeddings

# -----------------------
# Validation tests
# -----------------------

def test_validate_embedding_model_valid() -> None:
    Embeddings.validate_embedding_model("openai")
    Embeddings.validate_embedding_model("fastembed")


def test_validate_embedding_model_invalid() -> None:
    with pytest.raises(ValueError) as exc:
        Embeddings.validate_embedding_model("bogus")
    assert "Unsupported embedding model" in str(exc.value)
    assert "openai" in str(exc.value) and "fastembed" in str(exc.value)

# -----------------------
# FastEmbed (mocked)
# -----------------------

def test_embed_fastembed_success(monkeypatch) -> None:
    # Mock fastembed.TextEmbedding so we don't require the package here
    fake_model = Mock()
    # Your implementation uses: list(model.embed(t))[0].tolist()
    # Return an array-of-arrays so list(...)[0] works:
    fake_model.embed.side_effect = lambda t: np.array([[0.1, 0.2, 0.3]])
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=lambda: fake_model))

    texts = ["hello", "world"]
    out = Embeddings._embed_fastembed(texts)  # private helper by design; OK to test directly
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(v, list) for v in out)
    assert out[0] == [0.1, 0.2, 0.3]
    assert out[1] == [0.1, 0.2, 0.3]
    # Ensure we called embed once per text
    assert fake_model.embed.call_count == 2

# -----------------------
# OpenAI (mocked)
# -----------------------

class _FakeRespItem:
    def __init__(self, vec: List[float]):
        self.embedding = vec

class _FakeClient:
    """Client that mimics openai.OpenAI with a retry-able create()."""
    def __init__(self, *, fail_first_n: int = 0):
        self.fail_first_n = fail_first_n
        self.calls = 0
        # expose .embeddings.create(...)
        self.embeddings = self

    def create(self, input, model):  # pylint: disable=redefined-builtin
        self.calls += 1
        if self.calls <= self.fail_first_n:
            # Will be replaced by openai.RateLimitError in setup
            raise self.RateLimitError  # type: ignore[attr-defined]
        # Return one vector per input string
        data = [_FakeRespItem([0.1, 0.2, 0.3, 0.4, 0.5]) for _ in input]
        return types.SimpleNamespace(data=data)

def _install_fake_openai(monkeypatch, *, fail_first_n: int = 0):
    fake_openai = types.SimpleNamespace()
    # Exception type to raise on rate limiting
    fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
    # Bind exception type onto client so it can raise it
    client = _FakeClient(fail_first_n=fail_first_n)
    setattr(client, "RateLimitError", fake_openai.RateLimitError)
    # Factory that returns our client instance
    fake_openai.OpenAI = lambda api_key=None: client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return client, fake_openai

def test_embed_openai_success(monkeypatch) -> None:
    client, _ = _install_fake_openai(monkeypatch, fail_first_n=0)
    # Avoid sleeping in tests
    monkeypatch.setattr("time.sleep", lambda _s: None)

    out = Embeddings._embed_openai(["a", "b"], max_retries=3)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(v, list) for v in out)
    assert out[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert client.calls == 1  # no retries needed

def test_embed_openai_retries_then_success(monkeypatch) -> None:
    client, fake_openai = _install_fake_openai(monkeypatch, fail_first_n=2)
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    out = Embeddings._embed_openai(["x"], max_retries=5)
    assert out == [[0.1, 0.2, 0.3, 0.4, 0.5]]
    # Should have retried twice, then succeeded: total 3 calls
    assert client.calls == 3
    # Backoff uses 2 ** attempt for attempt=0,1 on failures
    assert sleep_calls == [1, 2]  # 2**0, 2**1

def test_embed_openai_exhausts_retries(monkeypatch) -> None:
    _client, _ = _install_fake_openai(monkeypatch, fail_first_n=10)  # more than retries
    monkeypatch.setattr("time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError) as exc:
        Embeddings._embed_openai(["only"], max_retries=3)
    assert "Failed to embed after multiple retries" in str(exc.value)

# -----------------------
# Public dispatchers
# -----------------------

def test_embed_texts_dispatch_fastembed(monkeypatch) -> None:
    fake_model = Mock()
    fake_model.embed.side_effect = lambda t: np.array([[0.9, 0.8]])
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=lambda: fake_model))

    out = Embeddings.embed_texts("fastembed", ["q1", "q2"])
    assert out == [[0.9, 0.8], [0.9, 0.8]]

def test_embed_texts_dispatch_openai(monkeypatch) -> None:
    client, _ = _install_fake_openai(monkeypatch, fail_first_n=0)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    out = Embeddings.embed_texts("openai", ["q"])
    assert out == [[0.1, 0.2, 0.3, 0.4, 0.5]]
    assert client.calls == 1

def test_embed_texts_invalid_model() -> None:
    with pytest.raises(ValueError):
        Embeddings.embed_texts("nope", ["q"])

def test_embed_one_fastembed(monkeypatch) -> None:
    fake_model = Mock()
    fake_model.embed.side_effect = lambda t: np.array([[0.42, 0.24, 0.06]])
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=lambda: fake_model))

    vec = Embeddings.embed_one("fastembed", "single")
    assert vec == [0.42, 0.24, 0.06]

# -----------------------
# Optional integration (real FastEmbed)
# -----------------------

def test_fastembed_integration_real_model() -> None:
    """Runs only if fastembed is installed. Uses real CPU embedding (offline)."""
    try:
        from fastembed import TextEmbedding  # type: ignore
    except ImportError:
        pytest.skip("fastembed not installed")

    out = Embeddings.embed_texts("fastembed", ["hello", "world"])
    # Basic shape/content checks
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(v, list) and len(v) > 0 for v in out)
    # Elements are floats
    assert all(isinstance(x, float) for x in out[0])
