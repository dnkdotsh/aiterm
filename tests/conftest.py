# tests/conftest.py
"""
This module contains shared fixtures for the pytest suite.
Fixtures defined here are automatically available to all test functions.
"""

import logging
from unittest.mock import MagicMock

import pytest
import requests
from aiterm import config  # Import config to access LOG_DIRECTORY
from aiterm.engine import GeminiEngine, OpenAIEngine
from aiterm.managers.session_manager import SessionManager
from aiterm.session_state import SessionState
from pyfakefs.fake_filesystem_unittest import Patcher


# Configure caplog to capture the 'aiterm' logger
# This hook ensures the logger is configured before tests run
def pytest_configure(config):
    logging.getLogger("aiterm").propagate = True


@pytest.fixture
def mock_openai_engine():
    """Provides a mock OpenAIEngine instance."""
    return OpenAIEngine(api_key="fake_openai_key")


@pytest.fixture
def mock_gemini_engine():
    """Provides a mock GeminiEngine instance."""
    return GeminiEngine(api_key="fake_gemini_key")


@pytest.fixture
def mock_session_state(mock_gemini_engine):
    """Provides a basic SessionState instance for testing, defaulting to Gemini."""
    return SessionState(
        engine=mock_gemini_engine,
        model="gemini-1.5-flash",
        system_prompt="You are a helpful assistant.",
        initial_system_prompt="You are a helpful assistant.",
        current_persona=None,
        max_tokens=1024,
        memory_enabled=True,
        debug_active=False,
        stream_active=True,
    )


@pytest.fixture
def mock_openai_session_state(mock_openai_engine):
    """Provides a SessionState instance using OpenAIEngine."""
    return SessionState(
        engine=mock_openai_engine,
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        initial_system_prompt="You are a helpful assistant.",
        current_persona=None,
        max_tokens=1024,
        memory_enabled=True,
        debug_active=False,
        stream_active=True,
    )


@pytest.fixture
def mock_session_manager(mocker, mock_session_state):
    """Provides a mock SessionManager instance (defaulting to Gemini) with a mock context manager."""
    mock_context_manager = mocker.MagicMock()
    # Create the session manager with the mock state and mock context manager
    manager = SessionManager(
        state=mock_session_state, context_manager=mock_context_manager
    )
    # Also mock the image workflow which is initialized inside the manager
    manager.image_workflow = mocker.MagicMock()
    return manager


@pytest.fixture
def mock_openai_session_manager(mocker, mock_openai_session_state):
    """Provides a SessionManager instance with an OpenAI engine."""
    mock_context_manager = mocker.MagicMock()
    manager = SessionManager(
        state=mock_openai_session_state, context_manager=mock_context_manager
    )
    manager.image_workflow = mocker.MagicMock()
    return manager


@pytest.fixture
def fake_fs():
    """
    Initializes a fake filesystem using pyfakefs for tests that
    require filesystem interactions (e.g., reading/writing settings,
    sessions, logs).
    Ensures core application directories exist in the fake filesystem.
    """
    patcher = Patcher()
    patcher.setUp()
    try:
        # Create all necessary application directories within the fake filesystem.
        config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        config.IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        config.SESSIONS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        config.PERSONAS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        yield patcher.fs
    finally:
        patcher.tearDown()


@pytest.fixture
def mock_requests_post(mocker):
    """
    A fixture that mocks `requests.post` to prevent actual network calls.
    Returns the mock object for customization within tests.
    """
    return mocker.patch("requests.post")


@pytest.fixture
def mock_openai_chat_response():
    """A fixture providing a standard, non-streaming OpenAI API chat response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_gemini_chat_response():
    """A fixture providing a standard, non-streaming Gemini API chat response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a test response."}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 25,
            "totalTokenCount": 40,
            "cachedContentTokenCount": 5,
        },
    }


@pytest.fixture
def mock_prompt_toolkit(mocker):
    """
    Mocks prompt_toolkit.prompt and underlying sys.stdin interactions
    to avoid EOFError in interactive tests.
    It provides an input_queue for tests to push expected inputs into.
    """
    input_queue = []

    # Mock sys.stdin.isatty to make prompt_toolkit think it's an interactive terminal.
    mocker.patch("sys.stdin.isatty", return_value=True)

    def _mocked_prompt_side_effect(message="", **kwargs):
        if not input_queue:
            # When the queue is empty, raise EOFError to simulate the user
            # pressing Ctrl+D, allowing loops to terminate gracefully.
            raise EOFError
        return input_queue.pop(0)

    # Patch prompt where it is USED. The tests that need this are in
    # `test_commands.py`, and the `prompt` function is imported and called
    # directly within the `aiterm.commands` module.
    mocker.patch("aiterm.commands.prompt", side_effect=_mocked_prompt_side_effect)

    return {
        "input_queue": input_queue  # Tests will populate this list
    }


@pytest.fixture
def mock_config_params():
    """Provides a standard mock dictionary for resolved configuration parameters."""
    return {
        "engine_name": "gemini",
        "model": "gemini-test-model",
        "max_tokens": 1024,
        "stream": True,
        "memory_enabled": True,
        "debug_enabled": False,
        "persona": None,
        "session_name": "test_session",
        "system_prompt_arg": None,
        "files_arg": [],
        "exclude_arg": [],
    }


@pytest.fixture
def mock_streaming_response_factory():
    """Factory fixture to create a mock streaming requests.Response object."""

    def _create_mock_response(chunks):
        response = MagicMock(spec=requests.Response)
        byte_chunks = [c.encode("utf-8") for c in chunks]
        response.iter_lines.return_value = iter(byte_chunks)
        response.status_code = 200
        return response

    return _create_mock_response
