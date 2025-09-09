# src/aiterm/test_managers/test_multichat_manager.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.aiterm.engine import AIEngine
from src.aiterm.managers.context_manager import ContextManager
from src.aiterm.managers.multichat_manager import MultiChatSession
from src.aiterm.session_state import MultiChatSessionState


@pytest.fixture
def mock_multichat_session_state():
    """Provides a MultiChatSessionState instance for testing."""
    mock_openai_engine = MagicMock(spec=AIEngine)
    mock_openai_engine.name = "openai"
    mock_gemini_engine = MagicMock(spec=AIEngine)
    mock_gemini_engine.name = "gemini"
    return MultiChatSessionState(
        openai_engine=mock_openai_engine,
        gemini_engine=mock_gemini_engine,
        openai_model="gpt-test",
        gemini_model="gemini-test",
        max_tokens=100,
    )


@pytest.fixture
def mock_context_manager():
    """Provides a mock ContextManager instance."""
    return MagicMock(spec=ContextManager)


@pytest.fixture
def mock_multichat_session(mock_multichat_session_state, mock_context_manager):
    """Provides a MultiChatSession instance for testing with required state."""
    return MultiChatSession(
        state=mock_multichat_session_state, context_manager=mock_context_manager
    )


@pytest.mark.usefixtures("mock_settings_patcher")
class TestMultiChatManager:
    @patch("src.aiterm.managers.multichat_manager.api_client.perform_chat_request")
    @patch("src.aiterm.managers.multichat_manager.threading.Thread")
    def test_process_turn_broadcast(
        self,
        mock_thread_class,
        mock_perform_chat,
        mock_multichat_session,
        mocker,
    ):
        """Tests the broadcast functionality where both AIs respond."""
        # Set primary engine
        mocker.patch(
            "src.aiterm.managers.multichat_manager.app_settings.settings",
            {"default_engine": "openai"},
        )
        mock_multichat_session.primary_engine_name = "openai"

        # Mock API responses
        mock_perform_chat.return_value = (
            "OpenAI response.",
            {"prompt": 10, "completion": 5},
        )
        # Mock the secondary worker's behavior
        mock_thread_instance = mock_thread_class.return_value

        def mock_secondary_worker_target(*args, **kwargs):
            q = kwargs["args"][-1]
            q.put(
                {
                    "engine_name": "gemini",
                    "text": "Gemini response.",
                    "tokens": {"prompt": 12, "completion": 6},
                }
            )

        mock_thread_instance.start = lambda: mock_secondary_worker_target(
            args=mock_thread_class.call_args.kwargs["args"]
        )
        mock_thread_instance.join = lambda: None

        mock_multichat_session.process_turn("Hello world", Path("/fake/log.jsonl"))

        # Verify history
        assert len(mock_multichat_session.state.shared_history) == 3
        assert "Director to All: Hello world" in str(
            mock_multichat_session.state.shared_history[0]
        )
        assert "OpenAI response." in str(mock_multichat_session.state.shared_history[1])
        assert "Gemini response." in str(mock_multichat_session.state.shared_history[2])

        # Verify token counts
        assert mock_multichat_session.state.total_prompt_tokens == 22
        assert mock_multichat_session.state.total_completion_tokens == 11

    @patch("src.aiterm.managers.multichat_manager.api_client.perform_chat_request")
    def test_process_turn_targeted(self, mock_perform_chat, mock_multichat_session):
        """Tests targeting a single AI with a prompt."""
        mock_perform_chat.return_value = (
            "Targeted Gemini response.",
            {"prompt": 8, "completion": 4},
        )

        mock_multichat_session.process_turn(
            "/ai gem What is the capital of France?", Path("/fake/log.jsonl")
        )

        assert len(mock_multichat_session.state.shared_history) == 2
        user_msg = mock_multichat_session.state.shared_history[0]
        asst_msg = mock_multichat_session.state.shared_history[1]

        assert "Director to Gemini: What is the capital of France?" in str(user_msg)
        assert asst_msg["source_engine"] == "gemini"
        assert "Targeted Gemini response." in str(asst_msg)

        assert mock_multichat_session.state.total_prompt_tokens == 8
        assert mock_multichat_session.state.total_completion_tokens == 4

    @patch("src.aiterm.managers.multichat_manager.api_client.perform_chat_request")
    def test_process_turn_targeted_continuation(
        self, mock_perform_chat, mock_multichat_session
    ):
        """Tests asking a single AI to continue the conversation."""
        mock_perform_chat.return_value = (
            "Continuation.",
            {"prompt": 2, "completion": 1},
        )

        mock_multichat_session.process_turn("/ai gpt", Path("/fake/log.jsonl"))

        assert len(mock_multichat_session.state.shared_history) == 2
        user_msg = mock_multichat_session.state.shared_history[0]

        assert "Director to Openai: Please continue" in str(user_msg)
        assert mock_multichat_session.state.total_prompt_tokens == 2
        assert mock_multichat_session.state.total_completion_tokens == 1

    @patch("src.aiterm.managers.multichat_manager.api_client.perform_chat_request")
    @patch("src.aiterm.managers.multichat_manager.threading.Thread")
    def test_secondary_worker_api_error(
        self, mock_thread_class, mock_perform_chat, mock_multichat_session, mocker
    ):
        """Tests that an API error in the secondary worker is handled."""
        mocker.patch(
            "src.aiterm.managers.multichat_manager.app_settings.settings",
            {"default_engine": "openai"},
        )
        mock_multichat_session.primary_engine_name = "openai"

        mock_perform_chat.return_value = ("OK", {"prompt": 1, "completion": 1})

        def mock_secondary_worker_target(*args, **kwargs):
            q = kwargs["args"][-1]
            q.put(
                {
                    "engine_name": "gemini",
                    "text": "Error: API key invalid",
                    "tokens": {},
                }
            )

        mock_thread_instance = mock_thread_class.return_value
        mock_thread_instance.start = lambda: mock_secondary_worker_target(
            args=mock_thread_class.call_args.kwargs["args"]
        )
        mock_thread_instance.join = lambda: None

        mock_multichat_session.process_turn("test", Path("/fake/log.jsonl"))

        assert "Error: API key invalid" in str(
            mock_multichat_session.state.shared_history[2]
        )

    def test_log_multichat_turn(self, mocker, mock_multichat_session):
        """Tests that a turn is correctly logged to a file."""
        mock_open = mocker.patch("builtins.open", mocker.mock_open())
        mock_multichat_session._log_multichat_turn(
            Path("/fake.jsonl"), [{"role": "user"}]
        )
        mock_open.assert_called_with(Path("/fake.jsonl"), "a", encoding="utf-8")
        mock_open().write.assert_called_once()
