"""Tests associated with Augmenter module"""
# pylint: disable=protected-access
# pylint: disable=unused-import

import os
import logging
from typing import Dict, Any, Callable, Tuple
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
from RAGToolBox.augmenter import Augmenter, GenerationConfig, ChatConfig, initiate_chat
from RAGToolBox.logging import LoggingConfig, RAGTBLogger


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

    assert "Context (ground-truth evidence):" in prompt
    assert "Question: What is LIFU?" in prompt
    assert "Answer:" in prompt


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_call_local_model(
    local_transformers: Tuple[MagicMock, MagicMock]
    ) -> None:
    """Test local model inference."""
    tokenizer, model = local_transformers

    augmenter = Augmenter(use_local=True)
    augmenter.tokenizer = tokenizer
    augmenter.model = model

    result = augmenter._call_local_model("Test prompt")

    assert result == "Generated response"
    tokenizer.encode.assert_called_once()
    model.generate.assert_called_once()


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE, reason="transformers and torch packages not available"
    )
def test_call_local_model_empty_response(
    local_transformers: Tuple[MagicMock, MagicMock]
    ) -> None:
    """Test local model inference with empty response."""
    tokenizer, model = local_transformers
    tokenizer.decode.return_value = "Input prompt"  # No generated content

    augmenter = Augmenter(use_local=True)
    augmenter.tokenizer = tokenizer
    augmenter.model = model

    result = augmenter._call_local_model("Test prompt")
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


def test_call_huggingface_api_success(
    hf_completion: Callable[[str], MagicMock]
    ) -> None:
    """Test successful Hugging Face API call."""
    mock_client = hf_completion("Generated response")
    augmenter = Augmenter(api_key="test_key")
    augmenter.client = mock_client

    result = augmenter._call_huggingface_api("Test prompt")

    assert result == "Generated response"
    mock_client.chat.completions.create.assert_called_once()


def test_call_huggingface_api_model_not_found(
    hf_completion: Callable[[str], MagicMock]
    ) -> None:
    """Test Hugging Face API call with model not found error."""
    mock_client = hf_completion()
    augmenter = Augmenter(api_key="test_key")
    augmenter.client = mock_client

    mock_client.chat.completions.create.side_effect = Exception("404 Not Found")

    with pytest.raises(RuntimeError, match="Model 'google/gemma-2-2b-it' is not available"):
        augmenter._call_huggingface_api("test prompt")


def test_call_huggingface_api_authentication_error(
    hf_completion: Callable[[str], MagicMock]
    ) -> None:
    """Test Hugging Face API call with authentication error."""
    mock_client = hf_completion()
    augmenter = Augmenter(api_key="test_key")
    augmenter.client = mock_client

    mock_client.chat.completions.create.side_effect = Exception("Authentication failed")

    with pytest.raises(RuntimeError, match="Authentication error"):
        augmenter._call_huggingface_api("test prompt")


def test_call_huggingface_api_generic_error(
    hf_completion: Callable[[str], MagicMock]
    ) -> None:
    """Test Hugging Face API call with generic error."""
    mock_client = hf_completion()
    augmenter = Augmenter(api_key="test_key")
    augmenter.client = mock_client

    mock_client.chat.completions.create.side_effect = Exception("Generic API error")

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
        mock_format.assert_called_once_with(query, chunks, None) # No history
        mock_call.assert_called_once_with("Formatted prompt", 0.25, 200)


def test_generate_response_empty_chunks(fallback_message: str) -> None:
    """Test response generation with empty chunks."""
    augmenter = Augmenter()
    result = augmenter.generate_response("What is LIFU?", [])
    assert result == fallback_message


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
        mock_generate.assert_called_once_with(query, chunks, GenerationConfig(0.25, 200), None)


def test_update_history() -> None:
    """Test that _update_history method returns a chat history chunk"""
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
    history = aug._update_history(history)

    assert isinstance(history, str)
    assert "Conversation so far (for disambiguation only):" in history
    assert "User: Hi" in history
    assert "Assistant: Hello!" in history
    assert "User: What is RAG?" in history
    assert "Assistant: RAG is Retrieval-Augmented Generation." in history

    # Empty history falls back to dtype=None
    empty = None
    empty = aug._update_history(empty)
    assert empty is None


def test_process_query_once_without_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that _process_query_once uses retriever,
    injects history chunk, and calls generate_response.
    """
    from collections import deque

    aug = Augmenter.__new__(Augmenter)

    class _FakeRetriever:
        """Mock retriever"""
        def retrieve(self, query: str, ret_config: RetrievalConfig):
            """Mock retrieve method"""
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
    def _fake_generate(self, query, retrieved_chunks, gen_config, chat_history): # pylint: disable=unused-argument
        # Capture what _process_query_once sends to generate_response
        captured["query"] = query
        captured["retrieved_chunks"] = retrieved_chunks
        captured["temperature"] = gen_config.temperature
        captured["max_new_tokens"] = gen_config.max_new_tokens
        captured["hist"] = chat_history
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
        retriever_obj=retriever,
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
    assert isinstance(rc, list) and len(rc) == 2

    hist_text = captured["hist"]
    assert "User: recent user msg" in hist_text
    assert "Assistant: recent assistant msg" in hist_text
    assert "User: old user msg" not in hist_text
    assert "Assistant: old assistant msg" not in hist_text
    assert rc[0]["data"] == "KB chunk 1"
    assert rc[1]["data"] == "KB chunk 2"


def test_process_query_once_with_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that _process_query_once routes to generate_response_with_sources
    when include_sources=True.
    """
    from collections import deque
    aug = Augmenter.__new__(Augmenter)

    class _FakeRetriever:
        """Mock retriever"""
        def retrieve(self, query: str, ret_config: RetrievalConfig): # pylint: disable=unused-argument
            """Mock retrieve method"""
            return [{"data": "KB", "metadata": {}}]

    retriever = _FakeRetriever()
    history = deque([("u1", "a1")], maxlen=50)

    captured = {}
    def _fake_generate_with_sources(
        self, query, retrieved_chunks, gen_config, chat_history # pylint: disable=unused-argument
        ):
        captured["query"] = query
        captured["retrieved_chunks"] = retrieved_chunks
        captured["hist"] = chat_history
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
        retriever_obj=retriever,
        chat_config = chat_config
        )

    assert out["response"] == "with sources"
    assert out["query"] == "hello"
    assert out["num_sources"] == len(out["sources"]) == len(captured["retrieved_chunks"])
    assert "Conversation so far (for disambiguation only):" in captured["hist"]


def test_initiate_chat_basic_flow( # pylint: disable=too-many-locals
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
    seen_calls: Dict[str, Any] = {"history_obj": None, "kwargs": None}
    def _fake_process(self, *, query, retriever_obj, chat_config): # pylint: disable=unused-argument
        # record the deque and params
        seen_calls["history_obj"] = chat_config.history
        seen_calls["kwargs"] = {
            'query': query, 'top_k': chat_config.ret_config.top_k,
            'max_retries': chat_config.ret_config.max_retries,
            'temperature': chat_config.gen_config.temperature,
            'max_new_tokens': chat_config.gen_config.max_new_tokens,
            'include_sources': chat_config.include_sources,
            'history_turns': chat_config.history_turns,
            'retriever_is_passed': retriever is not None
            }
        return {"response": "Hi!", "sources": [], "num_sources": 0}

    monkeypatch.setattr(Augmenter, "_process_query_once", _fake_process, raising=True)
    with pytest.raises(SystemExit) as excinfo:
        initiate_chat(augmenter_obj=augmenter, retriever_obj=retriever, command_args=args)

    assert excinfo.value.code == 0
    assert called_exit["code"] == 0

    captured = capsys.readouterr()
    assert "Chat mode: type your message." in captured.out
    assert "Assistant: Hi!" in captured.out

    assert seen_calls.get("kwargs")["query"] == "Hello there"
    assert seen_calls.get("kwargs")["top_k"] == args.top_k
    assert seen_calls.get("kwargs")["max_retries"] == args.max_retries
    assert seen_calls.get("kwargs")["temperature"] == args.temperature
    assert seen_calls.get("kwargs")["max_new_tokens"] == args.max_tokens
    assert seen_calls.get("kwargs")["include_sources"] is False
    assert seen_calls.get("kwargs")["history_turns"] == args.history_turns
    assert seen_calls.get("kwargs")["retriever_is_passed"] is True

    assert isinstance(seen_calls["history_obj"], deque)

    user, assistant = seen_calls.get("history_obj")[-1]
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
    def _fake_process_with_sources(self, **kwargs): # pylint: disable=unused-argument
        return {
            "response": "Here you go.",
            "sources": [{"data": "A", "metadata": {}}, {"data": "B", "metadata": {}}],
            "num_sources": 2,
            }

    monkeypatch.setattr(Augmenter, "_process_query_once", _fake_process_with_sources, raising=True)

    with pytest.raises(SystemExit):
        initiate_chat(augmenter_obj=augmenter, retriever_obj=retriever, command_args=args)

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
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": (_ for _ in ()).throw(KeyboardInterrupt())
        )
    monkeypatch.setattr("sys.exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as excinfo:
        initiate_chat(augmenter_obj=augmenter, retriever_obj=retriever, command_args=args)

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
    def _fake_process(**kwargs): # pylint: disable=unused-argument
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return {"response": "OK", "sources": [], "num_sources": 0}

    monkeypatch.setattr(Augmenter, "_process_query_once", staticmethod(_fake_process), raising=True)

    with pytest.raises(SystemExit) as excinfo:
        initiate_chat(augmenter_obj=augmenter, retriever_obj=retriever, command_args=args)

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Error: boom" in out
    assert any(
        "Chat turn failed" in rec.message and rec.levelname == "ERROR" for rec in caplog.records
        )
