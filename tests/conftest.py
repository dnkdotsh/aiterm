# tests/conftest.py
"""
This module contains shared fixtures for the pytest suite.
Fixtures defined here are automatically available to all test functions.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests
from aiterm import config  # Import config to access LOG_DIRECTORY
from aiterm import settings as app_settings
from aiterm.chat_ui import MultiChatUI, SingleChatUI
from aiterm.engine import GeminiEngine, OpenAIEngine
from aiterm.managers.multichat_manager import MultiChatSession
from aiterm.managers.session_manager import SessionManager
from aiterm.session_state import MultiChatSessionState, SessionState
from pyfakefs.fake_filesystem_unittest import Patcher

# Mock settings that will be used as the baseline for multiple test files
MOCK_SETTINGS = {
    "default_engine": "gemini",
    "default_gemini_model": "gemini-default",
    "default_openai_chat_model": "openai-default",
    "default_openai_image_model": "dall-e-default",
    "default_max_tokens": 1000,
    "stream": True,
    "memory_enabled": True,
}


@pytest.fixture
def mock_settings_patcher():
    """
    Fixture to patch the global settings dictionary for the duration of a test.
    It replaces the content of the settings dict with MOCK_SETTINGS.
    """
    with patch.dict(app_settings.settings, MOCK_SETTINGS, clear=True):
        yield


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
def mock_multichat_session_state(mock_openai_engine, mock_gemini_engine):
    """Provides a MultiChatSessionState instance for testing."""
    return MultiChatSessionState(
        openai_engine=mock_openai_engine,
        gemini_engine=mock_gemini_engine,
        openai_model="gpt-4o-mini",
        gemini_model="gemini-1.5-flash",
        max_tokens=2048,
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
    # Set the default state for the workflow mock to be inactive
    manager.image_workflow.img_prompt_crafting = False
    return manager


@pytest.fixture
def mock_openai_session_manager(mocker, mock_openai_session_state):
    """Provides a SessionManager instance with an OpenAI engine."""
    mock_context_manager = mocker.MagicMock()
    manager = SessionManager(
        state=mock_openai_session_state, context_manager=mock_context_manager
    )
    manager.image_workflow = mocker.MagicMock()
    manager.image_workflow.img_prompt_crafting = False
    return manager


@pytest.fixture
def mock_multichat_session(mocker, mock_multichat_session_state):
    """Provides a mock MultiChatSession instance."""
    # We can pass the real state object to a MagicMock to have it available
    # as `mock_session.state` while still mocking all methods.
    mock_session = MagicMock(spec=MultiChatSession)
    mock_session.state = mock_multichat_session_state
    return mock_session


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
        config.CHATLOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
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

    # Patch prompt where it is USED.
    # It's used in commands.py and ui_helpers.py
    mocker.patch("aiterm.commands.prompt", side_effect=_mocked_prompt_side_effect)
    mocker.patch(
        "aiterm.utils.ui_helpers.prompt", side_effect=_mocked_prompt_side_effect
    )

    return {
        "input_queue": input_queue  # Tests will populate this list
    }


@pytest.fixture
def mock_prompt_toolkit_app(mocker):
    """Mocks prompt_toolkit's get_app() and its invalidate() method."""
    mock_app = MagicMock()
    mock_buffer = MagicMock()
    mock_buffer.text = ""  # Default empty text buffer
    mock_app.current_buffer = mock_buffer
    mock_get_app = mocker.patch("aiterm.chat_ui.get_app", return_value=mock_app)
    # Also patch for commands module if needed there
    mocker.patch("aiterm.commands.get_app", return_value=mock_app)
    return {"get_app": mock_get_app, "app": mock_app}


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


@pytest.fixture
def mock_chat_ui(mock_session_manager):
    """Provides a SingleChatUI instance with a mocked session manager."""
    return SingleChatUI(
        session_manager=mock_session_manager, session_name="test_ui_session"
    )


@pytest.fixture
def mock_multichat_ui(mock_multichat_session):
    """Provides a MultiChatUI instance with a mocked session."""
    return MultiChatUI(
        session=mock_multichat_session,
        session_name="test_multi_ui",
        initial_prompt=None,
    )
