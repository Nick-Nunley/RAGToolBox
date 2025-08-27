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

from RAGToolBox.retriever import RetrievalConfig
from RAGToolBox.augmenter import Augmenter, GenerationConfig, ChatConfig, initiate_chat
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


def test_augmenter_initialization_prompt_type_error(
    caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test Augmenter initialization with invalid prompt type"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    with pytest.raises(ValueError) as exc:
        Augmenter(prompt_type='Invalid prompt')
    err_msg = 'Invalid prompt_type'
    assert err_msg in str(exc.value)
    assert err_msg in caplog.text


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
        mock_generate.assert_called_once_with(query, chunks, GenerationConfig(0.25, 200))


def test_make_history_chunk() -> None:
    """Test that _make_history_chunk method returns a chat history chunk"""
    from collections import deque

    aug = Augmenter.__new__(Augmenter)

    # Non-empty history
    history = deque(
        [
            ("Hi", "Hello!"),
            ("What is RAG?", "RAG is Retrieval-Augmented Generation."),
        ],
        maxlen=50,
    )
    chunk = aug._make_history_chunk(history)

    assert isinstance(chunk, dict)
    assert chunk["metadata"]["type"] == "history"
    assert "Conversation so far:" in chunk["data"]
    assert "User: Hi" in chunk["data"]
    assert "Assistant: Hello!" in chunk["data"]
    assert "User: What is RAG?" in chunk["data"]
    assert "Assistant: RAG is Retrieval-Augmented Generation." in chunk["data"]

    # Empty history falls back to empty data but keeps history metadata
    empty = deque([], maxlen=50)
    empty_chunk = aug._make_history_chunk(empty)
    assert empty_chunk["data"] == ""
    assert empty_chunk["metadata"]["type"] == "history"


def test_process_query_once_without_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that _process_query_once uses retriever,
    injects history chunk, and calls generate_response.
    """
    from collections import deque

    aug = Augmenter.__new__(Augmenter)

    class _FakeRetriever:
        def retrieve(self, query: str, ret_config: RetrievalConfig):
            assert query == "Q"
            assert ret_config.top_k == 5
            assert ret_config.max_retries == 7
            return [
                {"data": "KB chunk 1", "metadata": {"id": "k1"}},
                {"data": "KB chunk 2", "metadata": {"id": "k2"}},
            ]

    retriever = _FakeRetriever()

    history = deque(
        [
            ("old user msg", "old assistant msg"),
            ("recent user msg", "recent assistant msg"),
        ],
        maxlen=50,
    )

    captured = {}
    def _fake_generate(self, query, retrieved_chunks, gen_config):
        # Capture what _process_query_once sends to generate_response
        captured["query"] = query
        captured["retrieved_chunks"] = retrieved_chunks
        captured["temperature"] = gen_config.temperature
        captured["max_new_tokens"] = gen_config.max_new_tokens
        return "final answer"

    monkeypatch.setattr(Augmenter, "generate_response", _fake_generate, raising=True)

    chat_config = ChatConfig(
        ret_config = RetrievalConfig(
            top_k=5,
            max_retries=7
            ),
        gen_config = GenerationConfig(
            temperature=0.33,
            max_new_tokens=128
            ),
        history=history,
        include_sources=False,
        history_turns=1
        )

    out = aug._process_query_once(
        query="Q",
        retriever=retriever,
        chat_config = chat_config
        )

    assert out["response"] == "final answer"
    assert out["num_sources"] == 2
    assert out["sources"] == [
        {"data": "KB chunk 1", "metadata": {"id": "k1"}},
        {"data": "KB chunk 2", "metadata": {"id": "k2"}},
        ]

    assert captured["query"] == "Q"
    assert captured["temperature"] == 0.33
    assert captured["max_new_tokens"] == 128

    rc = captured["retrieved_chunks"]
    assert isinstance(rc, list) and len(rc) == 3
    assert rc[0]["metadata"]["type"] == "history"

    hist_text = rc[0]["data"]
    assert "User: recent user msg" in hist_text
    assert "Assistant: recent assistant msg" in hist_text
    assert "User: old user msg" not in hist_text
    assert "Assistant: old assistant msg" not in hist_text
    assert rc[1]["data"] == "KB chunk 1"
    assert rc[2]["data"] == "KB chunk 2"


def test_process_query_once_with_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that _process_query_once routes to generate_response_with_sources
    when include_sources=True.
    """
    from collections import deque
    aug = Augmenter.__new__(Augmenter)

    class _FakeRetriever:
        def retrieve(self, query: str, ret_config: RetrievalConfig):
            return [{"data": "KB", "metadata": {}}]

    retriever = _FakeRetriever()
    history = deque([("u1", "a1")], maxlen=50)

    captured = {}
    def _fake_generate_with_sources(self, query, retrieved_chunks, gen_config):
        captured["query"] = query
        captured["retrieved_chunks"] = retrieved_chunks
        return {
            "response": "with sources",
            "sources": retrieved_chunks,
            "num_sources": len(retrieved_chunks),
            "query": query,
            "temperature": gen_config.temperature,
            "max_new_tokens": gen_config.max_new_tokens,
        }

    monkeypatch.setattr(
        Augmenter, "generate_response_with_sources", _fake_generate_with_sources, raising=True
        )

    chat_config = ChatConfig(
        ret_config = RetrievalConfig(
            top_k=5,
            max_retries=7
            ),
        gen_config = GenerationConfig(
            temperature=0.33,
            max_new_tokens=128
            ),
        history=history,
        include_sources=True,
        history_turns=5
        )

    out = aug._process_query_once(
        query="hello",
        retriever=retriever,
        chat_config = chat_config
        )

    assert out["response"] == "with sources"
    assert out["query"] == "hello"
    assert out["num_sources"] == len(out["sources"]) == len(captured["retrieved_chunks"])
    assert out["sources"][0]["metadata"].get("type") == "history"
    assert "Conversation so far:" in out["sources"][0]["data"]


def test_initiate_chat_basic_flow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
    """
    Test initiate_chat reads input, calls _process_query_once,
    prints assistant reply, and exits cleanly.
    """
    from types import SimpleNamespace
    from collections import deque

    # Arrange: fake args and collaborators
    args = SimpleNamespace(
        top_k=10,
        max_retries=5,
        temperature=0.25,
        max_tokens=200,
        sources=False,
        history_turns=3
        )

    augmenter = Augmenter.__new__(Augmenter)
    retriever = object()

    # Simulate user typing one message then exiting
    inputs = iter(["Hello there", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    # Prevent sys.exit from ending the test
    called_exit = {"code": None}
    def _fake_exit(code=0):
        called_exit["code"] = code
        raise SystemExit(code)
    monkeypatch.setattr("sys.exit", _fake_exit)

    # Capture what initiate_chat passes into _process_query_once
    seen_calls = {"history_obj": None, "kwargs": None}
    def _fake_process(self, *, query, retriever, chat_config):
        # record the deque and params
        seen_calls["history_obj"] = chat_config.history
        seen_calls["kwargs"] = dict(
            query=query, top_k=chat_config.ret_config.top_k,
            max_retries=chat_config.ret_config.max_retries,
            temperature=chat_config.gen_config.temperature,
            max_new_tokens=chat_config.gen_config.max_new_tokens,
            include_sources=chat_config.include_sources, history_turns=chat_config.history_turns,
            retriever_is_passed=retriever is not None
            )
        return {"response": "Hi!", "sources": [], "num_sources": 0}

    monkeypatch.setattr(Augmenter, "_process_query_once", _fake_process, raising=True)
    with pytest.raises(SystemExit) as excinfo:
        initiate_chat(augmenter=augmenter, retriever=retriever, args=args)

    assert excinfo.value.code == 0
    assert called_exit["code"] == 0

    captured = capsys.readouterr()
    assert "Chat mode: type your message." in captured.out
    assert "Assistant: Hi!" in captured.out

    assert seen_calls["kwargs"]["query"] == "Hello there"
    assert seen_calls["kwargs"]["top_k"] == args.top_k
    assert seen_calls["kwargs"]["max_retries"] == args.max_retries
    assert seen_calls["kwargs"]["temperature"] == args.temperature
    assert seen_calls["kwargs"]["max_new_tokens"] == args.max_tokens
    assert seen_calls["kwargs"]["include_sources"] is False
    assert seen_calls["kwargs"]["history_turns"] == args.history_turns
    assert seen_calls["kwargs"]["retriever_is_passed"] is True

    assert isinstance(seen_calls["history_obj"], deque)

    user, assistant = seen_calls["history_obj"][-1]
    assert user == "Hello there"
    assert assistant == "Hi!"


def test_initiate_chat_prints_sources_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
    """
    Unit test: when args.sources=True, the function prints a [Sources used: N] line.
    """
    from types import SimpleNamespace

    args = SimpleNamespace(
        top_k=3,
        max_retries=2,
        temperature=0.4,
        max_tokens=128,
        sources=True,
        history_turns=2
        )

    augmenter = Augmenter.__new__(Augmenter)
    retriever = object()

    # Simulate one message and exit
    inputs = iter(["what are sources?", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("sys.exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    # Return a response with a specific num_sources
    def _fake_process_with_sources(self, **kwargs):
        return {
            "response": "Here you go.",
            "sources": [{"data": "A", "metadata": {}}, {"data": "B", "metadata": {}}],
            "num_sources": 2,
            }

    monkeypatch.setattr(Augmenter, "_process_query_once", _fake_process_with_sources, raising=True)

    with pytest.raises(SystemExit):
        initiate_chat(augmenter=augmenter, retriever=retriever, args=args)

    out = capsys.readouterr().out
    assert "Assistant: Here you go." in out
    assert "[Sources used: 2]" in out


def test_initiate_chat_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
    """
    Test when a KeyboardInterrupt occurs (e.g., Ctrl+C during input),
    the function prints the 'Interrupted' message and exits cleanly (code 0).
    """
    from types import SimpleNamespace

    args = SimpleNamespace(
        top_k=5,
        max_retries=3,
        temperature=0.25,
        max_tokens=200,
        sources=False,
        history_turns=3
        )

    augmenter = Augmenter.__new__(Augmenter)
    retriever = object()

    # First input call raises KeyboardInterrupt
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr("sys.exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as excinfo:
        initiate_chat(augmenter=augmenter, retriever=retriever, args=args)

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Interrupted. Exiting chat." in out


def test_initiate_chat_handles_generic_exception_and_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str]
    ) -> None:
    """
    Test if a generic Exception is raised during a turn, the function logs and prints
    an error, then continues to prompt for the next input until user exits.
    """
    from types import SimpleNamespace

    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    args = SimpleNamespace(
        top_k=5, max_retries=3, temperature=0.25, max_tokens=200, sources=True, history_turns=3
        )

    augmenter = Augmenter.__new__(Augmenter)
    retriever = object()

    inputs = iter(["hello", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("sys.exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    call_count = {"n": 0}
    def _fake_process(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("boom")
        return {"response": "OK", "sources": [], "num_sources": 0}

    monkeypatch.setattr(Augmenter, "_process_query_once", staticmethod(_fake_process), raising=True)

    with pytest.raises(SystemExit) as excinfo:
        initiate_chat(augmenter=augmenter, retriever=retriever, args=args)

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Error: boom" in out
    assert any("Chat turn failed" in rec.message and rec.levelname == "ERROR" for rec in caplog.records)



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
    def _stub_hf(self, prompt: str, temperature: float, max_new_tokens: int) -> str:  # noqa: ARG002
        seen["prompts"].append(prompt)
        return "ok-from-llm"
    monkeypatch.setattr(Augmenter, "_call_huggingface_api", _stub_hf, raising=True)

    aug = Augmenter(api_key="dummy", use_local=False, prompt_type="default")

    # Minimal fake retriever that returns two chunks
    class _FakeRetriever:
        def __init__(self):
            self.calls = []
        def retrieve(self, query: str, ret_config: RetrievalConfig):
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
        initiate_chat(augmenter=aug, retriever=retriever, args=args)

    assert excinfo.value.code == 0

    # Printed output checks
    out = capsys.readouterr().out
    assert "Chat mode: type your message." in out
    assert "Assistant: ok-from-llm" in out
    assert "[Sources used: 2]" in out
    # Second turn: 1 history + 2 KB chunks -> 3 sources
    assert "[Sources used: 3]" in out

    assert retriever.calls == [
        ("hello world", args.top_k, args.max_retries),
        ("second turn", args.top_k, args.max_retries)
        ]

    assert seen["prompts"], "Expected at least one prompt captured"
    prompt = seen["prompts"][-1]

    assert "KB chunk A" in prompt
    assert "KB chunk B" in prompt
    assert "hello world" in prompt
