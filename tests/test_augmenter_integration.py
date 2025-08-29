"""Integration tests associated with Augmenter module"""
# pylint: disable=protected-access
# pylint: disable=unused-import

import os
import logging
from typing import List, Dict, Union, Callable, Tuple
from unittest.mock import patch, MagicMock
import pytest

# Check for optional dependencies
try:
    import transformers
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from RAGToolBox.retriever import RetrievalConfig
from RAGToolBox.augmenter import Augmenter, GenerationConfig, initiate_chat
from RAGToolBox.logging import LoggingConfig, RAGTBLogger


# Helpers and mocks

def _stub_call_hf(
    self, prompt: str, temperature: float, max_new_tokens: int # pylint: disable=unused-argument
    ) -> str:
    # We can assert prompt has our chunk text if desired:
    assert "Context 1: chunk A about ultrasound therapy" in prompt
    return "stubbed-answer"

def _fake_chunks() -> List[Dict[str, Union[str, Dict[str, str]]]]:
    return [
        {"data": "chunk A about ultrasound therapy", "metadata": {"id": "A"}},
        {"data": "chunk B about brain stimulation", "metadata": {"id": "B"}},
    ]


# =====================
# INTEGRATION TESTS
# =====================

def test_full_augmenter_workflow_api(
    hf_completion: Callable[[str], MagicMock]
    ) -> None:
    """Test complete augmenter workflow using API."""
    mock_client = hf_completion(
        "LIFU (Low-Intensity Focused Ultrasound) is a therapeutic "
        "technique that uses focused ultrasound waves."
    )

    augmenter = Augmenter(api_key="test_key")
    augmenter.client = mock_client

    query = "What is LIFU?"
    chunks = [
        {"data": "LIFU stands for Low-Intensity Focused Ultrasound.", "metadata": {}},
        {"data": "It is a therapeutic technique used in medical applications.", "metadata": {}},
    ]

    result = augmenter.generate_response(query, chunks)

    assert "LIFU" in result
    assert "therapeutic" in result.lower()
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_full_augmenter_workflow_local(
    local_transformers: Tuple[MagicMock, MagicMock]
    ) -> None:
    """Test complete augmenter workflow using local model."""
    tokenizer, model = local_transformers
    tokenizer.decode.return_value = "Formatted prompt LIFU is a therapeutic technique."

    augmenter = Augmenter(use_local=True)
    augmenter.tokenizer = tokenizer
    augmenter.model = model

    query = "What is LIFU?"
    chunks = [{"data": "LIFU is a therapeutic technique.", "metadata": {}}]

    result = augmenter.generate_response(query, chunks)

    assert "LIFU is a therapeutic technique" in result
    tokenizer.encode.assert_called_once()
    model.generate.assert_called_once()


def test_augmenter_with_retriever_integration(
    hf_completion: Callable[[str], MagicMock]
    ) -> None:
    """Test augmenter integration with retriever (mocked)."""
    mock_client = hf_completion(
        "Based on the context, LIFU and LIPUS are related but different techniques."
        )
    augmenter = Augmenter(api_key="test_key")
    augmenter.client = mock_client

    query = "Is LIPUS the same thing as LIFU?"
    retrieved_chunks = [
        {
            "data": "LIFU (Low-Intensity Focused Ultrasound) uses focused ultrasound waves.",
            "metadata": {}
            },
        {
            "data": "LIPUS (Low-Intensity Pulsed Ultrasound) uses pulsed ultrasound waves.",
            "metadata": {}
            },
        {
            "data": "Both techniques are used for therapeutic purposes " + \
            "but have different mechanisms.",
            "metadata": {}
            },
    ]

    result = augmenter.generate_response_with_sources(query, retrieved_chunks)

    assert result["response"] == (
        "Based on the context, LIFU and LIPUS are related but different techniques."
    )
    assert result["num_sources"] == 3
    assert result["query"] == query
    assert len(result["sources"]) == 3
    assert result["temperature"] == 0.25
    assert result["max_new_tokens"] == 200


def test_augmenter_error_handling(
    hf_completion: Callable[[str], MagicMock]
    ) -> None:
    """Test augmenter error handling in full workflow."""
    mock_client = hf_completion()
    augmenter = Augmenter(api_key="test_key")
    augmenter.client = mock_client

    mock_client.chat.completions.create.side_effect = Exception("API rate limit exceeded")

    query = "What is LIFU?"
    chunks = [{"data": "LIFU is a therapeutic technique.", "metadata": {}}]

    with pytest.raises(RuntimeError, match="Error calling Hugging Face API"):
        augmenter.generate_response(query, chunks)

def test_augmenter_environment_variable_handling():
    """Test augmenter initialization with environment variables."""
    test_env = {
        'HUGGINGFACE_API_KEY': 'env_test_key',
        'MODEL_NAME': 'test/model'
    }

    with patch.dict(os.environ, test_env, clear=True):
        augmenter = Augmenter()
        assert augmenter.api_key == 'env_test_key'
        assert augmenter.model_name == "google/gemma-2-2b-it"  # Should use default, not env var


def test_augmenter_custom_parameters(
    hf_completion: Callable[[str], MagicMock]
    ) -> None:
    """Test augmenter with custom model and parameters."""
    mock_client = hf_completion("Custom model response")

    augmenter = Augmenter(model_name="custom/model", api_key="custom_key", use_local=False)
    augmenter.client = mock_client

    result = augmenter.generate_response("Test query", [{"data": "Test chunk", "metadata": {}}])
    assert result == "Custom model response"

    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["model"] == "custom/model"
    assert call_args[1]["temperature"] == 0.25
    assert call_args[1]["max_tokens"] == 200

def test_augmenter_integration_verbose_logging_with_context(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
    """
    Test full flow for Augmenter.generate_response with DEBUG console logging.
    We stub out HF client init and the actual API call so there are no external deps.
    """
    caplog.set_level(logging.DEBUG)
    # Keep existing handlers; just set console to DEBUG for this test
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    # Avoid importing huggingface_hub in the test
    monkeypatch.setattr(Augmenter, "_initialize_api_client", lambda self: None)

    # Stub the actual API call to return a canned answer and ensure the log line is emitted
    monkeypatch.setattr(Augmenter, "_call_huggingface_api", _stub_call_hf)

    # Instantiate the augmenter (non-local path)
    aug = Augmenter(
        model_name="fake/model",
        api_key="dummy-key",
        use_local=False,
        prompt_type="default",
    )

    # Run end-to-end with context
    out = aug.generate_response(
        "ultrasound therapy", _fake_chunks(), GenerationConfig(temperature=0.2, max_new_tokens=32)
        )

    assert out == "stubbed-answer"
    log_text = caplog.text
    assert "Valid response from LLM generated" in log_text


def test_augmenter_integration_verbose_logging_no_context(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
    """
    Same test as above but with no retrieved chunks -> should WARN and return the fallback message.
    """
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    # Avoid HF client import/creation
    monkeypatch.setattr(Augmenter, "_initialize_api_client", lambda self: None)

    aug = Augmenter(
        model_name="fake/model",
        api_key="dummy-key",
        use_local=False,
        prompt_type="default",
    )

    out = aug.generate_response("anything", retrieved_chunks=[])

    # The fallback message your code returns/logs
    expected_msg = (
        "I don't have enough information to answer your question. "
        "Please try rephrasing or expanding your query."
    )
    assert expected_msg in out
    assert expected_msg in caplog.text


def test_initiate_chat(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ) -> None:
    """
    Full test of augmenter module with --chat option and minimal mocks.
    """
    from types import SimpleNamespace

    # Capture the actual prompt sent to the LLM
    seen = {"prompts": []}

    # Prevent HF client creation during Augmenter.__init__
    monkeypatch.setattr(Augmenter, "_initialize_api_client", lambda self: None, raising=True)

    # Stub the HF API call and capture the prompt
    def _stub_hf(
        self, prompt: str, temperature: float, max_new_tokens: int # pylint: disable=unused-argument
        ) -> str:
        seen["prompts"].append(prompt)
        return "ok-from-llm"
    monkeypatch.setattr(Augmenter, "_call_huggingface_api", _stub_hf, raising=True)

    aug = Augmenter(api_key="dummy", use_local=False, prompt_type="default")

    # Minimal fake retriever that returns two chunks
    class _FakeRetriever:
        """Mock retriever"""
        def __init__(self):
            self.calls = []
        def retrieve(self, query: str, ret_config: RetrievalConfig):
            """Mock retrieve method"""
            self.calls.append((query, ret_config.top_k, ret_config.max_retries))
            return [
                {"data": "KB chunk A", "metadata": {"id": "A"}},
                {"data": "KB chunk B", "metadata": {"id": "B"}},
            ]
    retriever = _FakeRetriever()

    args = SimpleNamespace(
        top_k=7,
        max_retries=3,
        temperature=0.33,
        max_tokens=128,
        sources=True,
        history_turns=2
        )

    # Simulate two turns and then exit
    inputs = iter(["hello world", "second turn", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("sys.exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as excinfo:
        initiate_chat(augmenter_obj=aug, retriever_obj=retriever, command_args=args)

    assert excinfo.value.code == 0

    # Printed output checks
    out = capsys.readouterr().out
    assert "Chat mode: type your message." in out
    assert "Assistant: ok-from-llm" in out
    assert "[Sources used: 2]" in out

    assert retriever.calls == [
        ("hello world", args.top_k, args.max_retries),
        ("second turn", args.top_k, args.max_retries)
        ]

    assert seen["prompts"], "Expected at least one prompt captured"
    prompt = seen["prompts"][-1]

    assert "KB chunk A" in prompt
    assert "KB chunk B" in prompt
    assert "hello world" in prompt
