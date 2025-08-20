"""Tests associated with Embeddings module"""
# pylint: disable=protected-access
# pylint: disable=unused-import

from __future__ import annotations
import sys
import logging
import types
from typing import List
from unittest.mock import Mock
import numpy as np
import pytest
from RAGToolBox.embeddings import Embeddings
from RAGToolBox.logging import RAGTBLogger, LoggingConfig

# Check for optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# -----------------------
# Validation tests
# -----------------------

def test_validate_embedding_model_valid() -> None:
    """Test that validate_embedding_model method works"""
    Embeddings.validate_embedding_model("openai")
    Embeddings.validate_embedding_model("fastembed")


def test_validate_embedding_model_invalid() -> None:
    """Test validate_embedding_model throws error as expected"""
    with pytest.raises(ValueError) as exc:
        Embeddings.validate_embedding_model("bogus")
    assert "Unsupported embedding model" in str(exc.value)
    assert "openai" in str(exc.value) and "fastembed" in str(exc.value)

# -----------------------
# FastEmbed (mocked)
# -----------------------

def test_embed_fastembed_success(monkeypatch) -> None:
    """Test embed with fastembed mock works"""
    # Mock fastembed.TextEmbedding so we don't require the package here
    fake_model = Mock()
    # Your implementation uses: list(model.embed(t))[0].tolist()
    # Return an array-of-arrays so list(...)[0] works:
    fake_model.embed.side_effect = lambda t: np.array([[0.1, 0.2, 0.3]])
    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        types.SimpleNamespace(TextEmbedding=lambda: fake_model)
        )

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
    """Mock response"""
    def __init__(self, vec: List[float]):
        self.embedding = vec

class _FakeClient:
    """Client that mimics openai.OpenAI with a retry-able create()."""
    def __init__(self, *, fail_first_n: int = 0, rate_limit_exc: type[Exception] = Exception):
        self.fail_first_n = fail_first_n
        self.calls = 0
        self.embeddings = self
        self.rate_limit_exc = rate_limit_exc

    def create(self, input, model):  # pylint: disable=unused-argument,redefined-builtin
        """Mocked create method to create embeddings"""
        self.calls += 1
        if self.calls <= self.fail_first_n:
            raise self.rate_limit_exc()
        data = [_FakeRespItem([0.1, 0.2, 0.3, 0.4, 0.5]) for _ in input]
        return types.SimpleNamespace(data=data)

def _install_fake_openai(monkeypatch, *, fail_first_n: int = 0):
    # Build a fake openai module with the exact exception class the code catches
    RateLimitError = type("RateLimitError", (Exception,), {})
    client = _FakeClient(fail_first_n=fail_first_n, rate_limit_exc=RateLimitError)

    fake_openai = types.SimpleNamespace(
        RateLimitError=RateLimitError,
        OpenAI=lambda api_key=None: client,
    )

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return client, fake_openai

def test_embed_openai_success(monkeypatch) -> None:
    """Test embed with openai mock works"""
    client, _ = _install_fake_openai(monkeypatch, fail_first_n=0)
    # Avoid sleeping in tests
    monkeypatch.setattr("time.sleep", lambda _s: None)

    out = Embeddings._embed_openai(["a", "b"], max_retries=3)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(v, list) for v in out)
    assert out[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert client.calls == 1  # no retries needed

def test_embed_openai_retries_then_success(monkeypatch) -> None:
    """Test embed_openai retries sucessfully"""
    client, _ = _install_fake_openai(monkeypatch, fail_first_n=2)
    sleep_calls = []
    def _record_sleep(seconds):
        """Record the sleep duration for assertions."""
        sleep_calls.append(seconds)
    monkeypatch.setattr("time.sleep", _record_sleep)

    out = Embeddings._embed_openai(["x"], max_retries=5)
    assert out == [[0.1, 0.2, 0.3, 0.4, 0.5]]
    # Should have retried twice, then succeeded: total 3 calls
    assert client.calls == 3
    # Backoff uses 2 ** attempt for attempt=0,1 on failures
    assert sleep_calls == [1, 2]  # 2**0, 2**1

@pytest.mark.skipif(
    not OPENAI_AVAILABLE, reason="openai package not installed"
    )
def test_embed_openai_exhausts_retries(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test embed_openai erros with exhuasted retries"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    _client, _ = _install_fake_openai(monkeypatch, fail_first_n=10)  # more than retries
    monkeypatch.setattr("time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError) as exc:
        Embeddings._embed_openai(["only"], max_retries=3)
    err_msg = "Failed to embed after multiple retries"
    assert err_msg in str(exc.value)
    assert err_msg in caplog.text

# -----------------------
# Public dispatchers
# -----------------------

def test_embed_texts_dispatch_fastembed(monkeypatch) -> None:
    """Test embed_texts with fastembed dispatch works"""
    fake_model = Mock()
    fake_model.embed.side_effect = lambda t: np.array([[0.9, 0.8]])
    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        types.SimpleNamespace(TextEmbedding=lambda: fake_model)
        )

    out = Embeddings.embed_texts("fastembed", ["q1", "q2"])
    assert out == [[0.9, 0.8], [0.9, 0.8]]

def test_embed_texts_dispatch_openai(monkeypatch) -> None:
    """Test embed_texts with openai dispath works"""
    client, _ = _install_fake_openai(monkeypatch, fail_first_n=0)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    out = Embeddings.embed_texts("openai", ["q"])
    assert out == [[0.1, 0.2, 0.3, 0.4, 0.5]]
    assert client.calls == 1

def test_embed_texts_invalid_model(caplog: pytest.LogCaptureFixture) -> None:
    """Test embed_texts throws error with invalid model"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    with pytest.raises(ValueError) as exc:
        Embeddings.embed_texts("nope", ["q"])
    err_msg = "Embedding model 'nope' not supported."
    assert err_msg in str(exc.value)
    assert err_msg in caplog.text

def test_embed_one_fastembed(monkeypatch) -> None:
    """Test embed_one with fastembed dispath works"""
    fake_model = Mock()
    fake_model.embed.side_effect = lambda t: np.array([[0.42, 0.24, 0.06]])
    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        types.SimpleNamespace(TextEmbedding=lambda: fake_model)
        )

    vec = Embeddings.embed_one("fastembed", "single")
    assert vec == [0.42, 0.24, 0.06]

# -----------------------
# Optional integration (real FastEmbed)
# -----------------------

def test_fastembed_integration_real_model() -> None:
    """Full test using real fastembed"""
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
