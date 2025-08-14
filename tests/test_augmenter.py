"""Tests associated with Augmenter module"""
# pylint: disable=protected-access
# pylint: disable=unused-import

import os
import logging
from typing import List, Dict, Union
from unittest.mock import patch, MagicMock, Mock
import pytest

# Check for optional dependencies
try:
    import transformers
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from RAGToolBox.augmenter import Augmenter
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
# UNIT TESTS
# =====================

def test_augmenter_initialization_default():
    """Test Augmenter initialization with default parameters."""
    with patch.dict(os.environ, {'HUGGINGFACE_API_KEY': 'test_key'}):
        augmenter = Augmenter()
        assert augmenter.model_name == "google/gemma-2-2b-it"
        assert augmenter.api_key == "test_key"
        assert augmenter.use_local is False


def test_augmenter_initialization_custom():
    """Test Augmenter initialization with custom parameters."""
    augmenter = Augmenter(
        model_name="test-model",
        api_key="custom_key",
        use_local=False
    )
    assert augmenter.model_name == "test-model"
    assert augmenter.api_key == "custom_key"
    assert augmenter.use_local is False


def test_augmenter_initialization_no_api_key():
    """Test Augmenter initialization without API key."""
    with patch.dict(os.environ, {}, clear=True):
        augmenter = Augmenter()
        assert augmenter.api_key is None


def test_initialize_api_client_success():
    """Test successful API client initialization."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create augmenter without calling _initialize_api_client in __init__
        augmenter = Augmenter.__new__(Augmenter)
        augmenter.model_name = "google/gemma-2-2b-it"
        augmenter.api_key = "test_key"
        augmenter.use_local = False

        # Now call the method we want to test
        augmenter._initialize_api_client()

        mock_client_class.assert_called_once_with(token="test_key")
        assert augmenter.client == mock_client


def test_initialize_api_client_import_error():
    """Test API client initialization with missing dependency."""
    with patch('huggingface_hub.InferenceClient', side_effect=ImportError("No module")):
        # Create augmenter without calling _initialize_api_client in __init__
        augmenter = Augmenter.__new__(Augmenter)
        augmenter.model_name = "google/gemma-2-2b-it"
        augmenter.api_key = "test_key"
        augmenter.use_local = False

        with pytest.raises(ImportError, match="huggingface_hub package is required"):
            augmenter._initialize_api_client()


def test_initialize_api_client_runtime_error():
    """Test API client initialization with runtime error."""
    with patch('huggingface_hub.InferenceClient', side_effect=Exception("API Error")):
        # Create augmenter without calling _initialize_api_client in __init__
        augmenter = Augmenter.__new__(Augmenter)
        augmenter.model_name = "google/gemma-2-2b-it"
        augmenter.api_key = "test_key"
        augmenter.use_local = False

        with pytest.raises(RuntimeError, match="Error initializing Hugging Face client"):
            augmenter._initialize_api_client()


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_initialize_local_model_success():
    """Test successful local model initialization."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer_class, \
         patch('transformers.AutoModelForCausalLM') as mock_model_class:

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Create augmenter without calling _initialize_local_model in __init__
        augmenter = Augmenter.__new__(Augmenter)
        augmenter.model_name = "google/gemma-2-2b-it"
        augmenter.api_key = None
        augmenter.use_local = True

        augmenter._initialize_local_model()

        assert augmenter.tokenizer == mock_tokenizer
        assert augmenter.model == mock_model
        assert mock_tokenizer.pad_token == "<eos>"


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_initialize_local_model_with_pad_token():
    """Test local model initialization when pad_token already exists."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer_class, \
         patch('transformers.AutoModelForCausalLM') as mock_model_class:

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"  # Already set
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Create augmenter without calling _initialize_local_model in __init__
        augmenter = Augmenter.__new__(Augmenter)
        augmenter.model_name = "google/gemma-2-2b-it"
        augmenter.api_key = None
        augmenter.use_local = True

        augmenter._initialize_local_model()

        assert augmenter.tokenizer.pad_token == "<pad>"  # Should remain unchanged


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_initialize_local_model_import_error():
    """Test local model initialization with missing dependency."""
    with patch('transformers.AutoTokenizer', side_effect=ImportError("No module")):
        # Create augmenter without calling _initialize_local_model in __init__
        augmenter = Augmenter.__new__(Augmenter)
        augmenter.model_name = "google/gemma-2-2b-it"
        augmenter.api_key = None
        augmenter.use_local = True

        with pytest.raises(ImportError, match="Transformers package is required"):
            augmenter._initialize_local_model()


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_initialize_local_model_runtime_error():
    """Test local model initialization with runtime error."""
    with patch('transformers.AutoTokenizer', side_effect=Exception("Model Error")):
        # Create augmenter without calling _initialize_local_model in __init__
        augmenter = Augmenter.__new__(Augmenter)
        augmenter.model_name = "google/gemma-2-2b-it"
        augmenter.api_key = None
        augmenter.use_local = True

        with pytest.raises(RuntimeError, match="Error loading model"):
            augmenter._initialize_local_model()


def test_format_prompt():
    """Test prompt formatting with retrieved chunks."""
    augmenter = Augmenter()
    query = "What is LIFU?"
    chunks = [
        {"data": "LIFU is a therapeutic technique.", "metadata": {}},
        {"data": "It uses focused ultrasound.", "metadata": {}}
        ]

    prompt = augmenter._format_prompt(query, chunks)

    assert "Context 1: LIFU is a therapeutic technique." in prompt
    assert "Context 2: It uses focused ultrasound." in prompt
    assert "Question: What is LIFU?" in prompt
    assert "Answer:" in prompt


def test_format_prompt_empty_chunks():
    """Test prompt formatting with empty chunks."""
    augmenter = Augmenter()
    query = "What is LIFU?"
    chunks = []

    prompt = augmenter._format_prompt(query, chunks)

    assert "Context:" in prompt
    assert "Question: What is LIFU?" in prompt
    assert "Answer:" in prompt


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_call_local_model():
    """Test local model inference."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer_class, \
         patch('transformers.AutoModelForCausalLM') as mock_model_class, \
         patch('torch.no_grad') as mock_no_grad:

        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = MagicMock(shape=[1, 10])
        mock_tokenizer.decode.return_value = "Input prompt Generated response"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs[0] = MagicMock()
        mock_model.generate.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model

        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        augmenter = Augmenter(use_local=True)
        augmenter.tokenizer = mock_tokenizer
        augmenter.model = mock_model

        prompt = "Test prompt"
        result = augmenter._call_local_model(prompt)

        assert result == "Generated response"
        mock_tokenizer.encode.assert_called_once()
        mock_model.generate.assert_called_once()


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_call_local_model_empty_response():
    """Test local model inference with empty response."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer_class, \
         patch('transformers.AutoModelForCausalLM') as mock_model_class, \
         patch('torch.no_grad') as mock_no_grad:

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = MagicMock(shape=[1, 10])
        mock_tokenizer.decode.return_value = "Input prompt"  # No generated content
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs[0] = MagicMock()
        mock_model.generate.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model

        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        augmenter = Augmenter(use_local=True)
        augmenter.tokenizer = mock_tokenizer
        augmenter.model = mock_model

        prompt = "Test prompt"
        result = augmenter._call_local_model(prompt)

        assert result == "I don't have enough information to provide a detailed answer."


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_call_local_model_exception():
    """Test local model inference with exception."""
    augmenter = Augmenter(use_local=True)
    augmenter.tokenizer = MagicMock()
    augmenter.model = MagicMock()

    # Mock tokenizer to raise exception
    augmenter.tokenizer.encode.side_effect = Exception("Tokenization error")

    with pytest.raises(RuntimeError, match="Error calling local model"):
        augmenter._call_local_model("test prompt")


def test_call_huggingface_api_success():
    """Test successful Hugging Face API call."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Generated response"
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_client_class.return_value = mock_client

        augmenter = Augmenter(api_key="test_key")
        augmenter.client = mock_client

        prompt = "Test prompt"
        result = augmenter._call_huggingface_api(prompt)

        assert result == "Generated response"
        mock_client.chat.completions.create.assert_called_once()


def test_call_huggingface_api_model_not_found():
    """Test Hugging Face API call with model not found error."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("404 Not Found")
        mock_client_class.return_value = mock_client

        augmenter = Augmenter(api_key="test_key")
        augmenter.client = mock_client

        with pytest.raises(RuntimeError, match="Model 'google/gemma-2-2b-it' is not available"):
            augmenter._call_huggingface_api("test prompt")


def test_call_huggingface_api_authentication_error():
    """Test Hugging Face API call with authentication error."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Authentication failed")
        mock_client_class.return_value = mock_client

        augmenter = Augmenter(api_key="test_key")
        augmenter.client = mock_client

        with pytest.raises(RuntimeError, match="Authentication error"):
            augmenter._call_huggingface_api("test prompt")


def test_call_huggingface_api_generic_error():
    """Test Hugging Face API call with generic error."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Generic API error")
        mock_client_class.return_value = mock_client

        augmenter = Augmenter(api_key="test_key")
        augmenter.client = mock_client

        with pytest.raises(RuntimeError, match="Error calling Hugging Face API"):
            augmenter._call_huggingface_api("test prompt")


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_call_llm_local():
    """Test LLM call routing to local model."""
    with patch.object(Augmenter, '_call_local_model') as mock_local:
        mock_local.return_value = "Local response"

        augmenter = Augmenter(use_local=True)
        result = augmenter._call_llm("test prompt")

        assert result == "Local response"
        mock_local.assert_called_once_with("test prompt", 0.25, 200)


def test_call_llm_api():
    """Test LLM call routing to API."""
    with patch.object(Augmenter, '_call_huggingface_api') as mock_api:
        mock_api.return_value = "API response"

        augmenter = Augmenter(use_local=False)
        result = augmenter._call_llm("test prompt")

        assert result == "API response"
        mock_api.assert_called_once_with("test prompt", 0.25, 200)


def test_generate_response_with_chunks():
    """Test response generation with retrieved chunks."""
    with patch.object(Augmenter, '_format_prompt') as mock_format, \
         patch.object(Augmenter, '_call_llm') as mock_call:

        mock_format.return_value = "Formatted prompt"
        mock_call.return_value = "Generated answer"

        augmenter = Augmenter()
        query = "What is LIFU?"
        chunks = [{"data": "LIFU is a therapeutic technique.", "metadata": {}}]

        result = augmenter.generate_response(query, chunks)

        assert result == "Generated answer"
        mock_format.assert_called_once_with(query, chunks)
        mock_call.assert_called_once_with("Formatted prompt", 0.25, 200)


def test_generate_response_empty_chunks():
    """Test response generation with empty chunks."""
    augmenter = Augmenter()
    query = "What is LIFU?"
    chunks = []

    result = augmenter.generate_response(query, chunks)

    assert result == "I don't have enough information to answer your question. " + \
    "Please try rephrasing or expanding your query."


def test_generate_response_with_sources():
    """Test response generation with source information."""
    with patch.object(Augmenter, 'generate_response') as mock_generate:
        mock_generate.return_value = "Generated answer"

        augmenter = Augmenter()
        query = "What is LIFU?"
        chunks = [{"data": "Chunk 1", "metadata": {}}, {"data": "Chunk 2", "metadata": {}}]

        result = augmenter.generate_response_with_sources(query, chunks)

        expected = {
            "response": "Generated answer",
            "sources": [{"data": "Chunk 1", "metadata": {}}, {"data": "Chunk 2", "metadata": {}}],
            "num_sources": 2,
            "query": "What is LIFU?",
            "temperature": 0.25,
            "max_new_tokens": 200
        }
        assert result == expected
        mock_generate.assert_called_once_with(query, chunks, 0.25, 200)


# =====================
# INTEGRATION TESTS
# =====================

def test_full_augmenter_workflow_api():
    """Test complete augmenter workflow using API."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "LIFU (Low-Intensity Focused Ultrasound) " + \
        "is a therapeutic technique that uses focused ultrasound waves."
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_client_class.return_value = mock_client

        augmenter = Augmenter(api_key="test_key")
        augmenter.client = mock_client

        query = "What is LIFU?"
        chunks = [
            {"data": "LIFU stands for Low-Intensity Focused Ultrasound.", "metadata": {}},
            {"data": "It is a therapeutic technique used in medical applications.", "metadata": {}}
        ]

        result = augmenter.generate_response(query, chunks)

        assert "LIFU" in result
        assert "therapeutic" in result.lower()
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_full_augmenter_workflow_local():
    """Test complete augmenter workflow using local model."""
    with patch('transformers.AutoTokenizer') as mock_tokenizer_class, \
         patch('transformers.AutoModelForCausalLM') as mock_model_class, \
         patch('torch.no_grad') as mock_no_grad:

        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = MagicMock(shape=[1, 10])
        mock_tokenizer.decode.return_value = "Formatted prompt LIFU is a therapeutic technique."
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs[0] = MagicMock()
        mock_model.generate.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model

        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)

        augmenter = Augmenter(use_local=True)
        augmenter.tokenizer = mock_tokenizer
        augmenter.model = mock_model

        query = "What is LIFU?"
        chunks = [{"data": "LIFU is a therapeutic technique.", "metadata": {}}]

        result = augmenter.generate_response(query, chunks)

        assert "LIFU is a therapeutic technique" in result
        mock_tokenizer.encode.assert_called_once()
        mock_model.generate.assert_called_once()


def test_augmenter_with_retriever_integration():
    """Test augmenter integration with retriever (mocked)."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Based on the context, LIFU and " + \
        "LIPUS are related but different techniques."
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_client_class.return_value = mock_client

        augmenter = Augmenter(api_key="test_key")
        augmenter.client = mock_client

        # Simulate retrieved chunks from retriever
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
                }
        ]

        result = augmenter.generate_response_with_sources(query, retrieved_chunks)

        assert result["response"] == "Based on the context, LIFU and LIPUS are " + \
        "related but different techniques."
        assert result["num_sources"] == 3
        assert result["query"] == query
        assert len(result["sources"]) == 3
        assert result["temperature"] == 0.25
        assert result["max_new_tokens"] == 200


def test_augmenter_error_handling():
    """Test augmenter error handling in full workflow."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API rate limit exceeded")
        mock_client_class.return_value = mock_client

        augmenter = Augmenter(api_key="test_key")
        augmenter.client = mock_client

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


def test_augmenter_custom_parameters():
    """Test augmenter with custom model and parameters."""
    with patch('huggingface_hub.InferenceClient') as mock_client_class:
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Custom model response"
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_client_class.return_value = mock_client

        augmenter = Augmenter(
            model_name="custom/model",
            api_key="custom_key",
            use_local=False
        )
        augmenter.client = mock_client

        query = "Test query"
        chunks = [{"data": "Test chunk", "metadata": {}}]

        result = augmenter.generate_response(query, chunks)

        assert result == "Custom model response"
        # Verify the custom model name was used in the API call
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "custom/model"
        # Verify default parameters were used
        assert call_args[1]['temperature'] == 0.25
        assert call_args[1]['max_tokens'] == 200

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
        "ultrasound therapy", _fake_chunks(), temperature=0.2, max_new_tokens=32
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
