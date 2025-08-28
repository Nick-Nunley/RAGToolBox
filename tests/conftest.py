"""Helpers for testing"""

from __future__ import annotations

from typing import Callable, Tuple
from unittest.mock import MagicMock, Mock
import pytest


@pytest.fixture
def hf_completion(
    monkeypatch: pytest.MonkeyPatch
    ) -> Callable[[str], MagicMock]:
    """
    Factory fixture that wires a mocked HF InferenceClient to return a completion
    with the provided text. Usage:

        mock_client = hf_completion("Generated response")
        aug = Augmenter(api_key="test_key")
        aug.client = mock_client
    """
    def _factory(content: str = "Generated response") -> MagicMock:
        # Build the completion-shaped stub
        mock_message = MagicMock()
        mock_message.content = content
        mock_choice = MagicMock(message=mock_message)
        mock_completion = MagicMock(choices=[mock_choice])

        # Build the client and patch constructor
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        monkeypatch.setattr(
            "huggingface_hub.InferenceClient",
            MagicMock(return_value=mock_client),
            raising=True,
        )
        return mock_client

    return _factory

@pytest.fixture
def local_transformers(
    monkeypatch: pytest.MonkeyPatch
    ) -> Tuple[MagicMock, MagicMock]:
    """
    Factory that patches transformers + torch.no_grad() and returns (tokenizer, model).
    It makes .encode(), .decode(), .generate() behave like your tests expect.
    """
    # tokenizer
    tokenizer = MagicMock()
    tokenizer.encode.return_value = MagicMock(shape=[1, 10])
    tokenizer.decode.return_value = "Input prompt Generated response"
    tokenizer.eos_token_id = 2
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        MagicMock(return_value=tokenizer),
        raising=True,
    )

    # model
    model = MagicMock()
    outputs = MagicMock()
    outputs[0] = MagicMock()
    model.generate.return_value = outputs
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        MagicMock(return_value=model),
        raising=True,
    )

    # no_grad context manager
    no_grad_cm = MagicMock()
    no_grad_cm.__enter__ = Mock()
    no_grad_cm.__exit__ = Mock(return_value=None)
    monkeypatch.setattr("torch.no_grad", MagicMock(return_value=no_grad_cm), raising=True)

    return tokenizer, model

@pytest.fixture
def fallback_message() -> str:
    """Helper for fallback message when context is not present in generation."""
    return (
        "I don't have enough information to answer your question. "
        "Please try rephrasing or expanding your query."
    )
