# tests/managers/test_multichat_manager.py
"""
Tests for the MultiChatSession class in aiterm/managers/multichat_manager.py.
"""

import json
import queue
from unittest.mock import patch

import pytest
from aiterm import config
from aiterm.managers.multichat_manager import MultiChatSession
from aiterm.prompts import CONTINUATION_PROMPT
from aiterm.utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
    extract_text_from_message,
)


@pytest.fixture
def mock_multichat_session(mock_multichat_session_state):
    """Provides a MultiChatSession instance for testing with required state."""
    # Add the system prompts that the manager expects to find.
    mock_multichat_session_state.system_prompts = {
        "openai": "You are OpenAI.",
        "gemini": "You are Gemini.",
    }
    return MultiChatSession(initial_state=mock_multichat_session_state)


class TestMultiChatManager:
    """Test suite for the MultiChatSession manager."""

    @patch("aiterm.managers.multichat_manager.queue.Queue")
    @patch("aiterm.managers.multichat_manager.threading.Thread")
    @patch("aiterm.api_client.perform_chat_request")
    def test_process_turn_broadcast(
        self, mock_api_request, mock_thread, mock_queue, mock_multichat_session, fake_fs
    ):
        """
        Tests a broadcast turn where both AIs are queried simultaneously.
        """
        # Arrange
        log_path = config.CHATLOG_DIRECTORY / "fake.log"
        fake_fs.create_file(log_path)
        mock_api_request.return_value = ("Primary response", {})
        mock_queue.return_value.get.return_value = {
            "engine_name": "openai",
            "text": "Secondary response",
        }
        initial_history_len = len(mock_multichat_session.state.shared_history)

        # Act
        mock_multichat_session.process_turn("Hello everyone", log_path)

        # Assert
        mock_api_request.assert_called_once()
        assert mock_api_request.call_args[0][0].name == "gemini"

        mock_thread.assert_called_once()
        thread_kwargs = mock_thread.call_args.kwargs
        assert thread_kwargs["target"] == mock_multichat_session._secondary_worker
        assert thread_kwargs["args"][0].name == "openai"

        final_history = mock_multichat_session.state.shared_history
        assert len(final_history) == initial_history_len + 3
        assert "Director to All: Hello everyone" in extract_text_from_message(
            final_history[-3]
        )
        assert "[Openai]: Secondary response" in extract_text_from_message(
            final_history[-2]
        )
        assert "[Gemini]: Primary response" in extract_text_from_message(
            final_history[-1]
        )

    @patch("aiterm.api_client.perform_chat_request")
    def test_process_turn_targeted(
        self, mock_api_request, mock_multichat_session, fake_fs
    ):
        """
        Tests a targeted turn where only one AI is queried via the /ai command.
        """
        # Arrange
        log_path = config.CHATLOG_DIRECTORY / "fake.log"
        fake_fs.create_file(log_path)
        mock_api_request.return_value = ("Targeted OpenAI response", {})
        initial_history_len = len(mock_multichat_session.state.shared_history)

        # Act
        mock_multichat_session.process_turn("/ai gpt What is your name?", log_path)

        # Assert
        mock_api_request.assert_called_once()
        call_args, _ = mock_api_request.call_args
        assert call_args[0].name == "openai"

        final_history = mock_multichat_session.state.shared_history
        assert len(final_history) == initial_history_len + 2
        assert "Director to Openai: What is your name?" in extract_text_from_message(
            final_history[-2]
        )
        assert "[Openai]: Targeted OpenAI response" in extract_text_from_message(
            final_history[-1]
        )

    @patch("aiterm.api_client.perform_chat_request")
    def test_process_turn_targeted_continuation(
        self, mock_api_request, mock_multichat_session, fake_fs
    ):
        """
        Tests that a targeted /ai command with no prompt uses the continuation prompt.
        """
        # Arrange
        log_path = config.CHATLOG_DIRECTORY / "fake.log"
        fake_fs.create_file(log_path)
        mock_api_request.return_value = ("Continuing...", {})

        # Act
        mock_multichat_session.process_turn("/ai gem", log_path)

        # Assert
        mock_api_request.assert_called_once()
        call_args, _ = mock_api_request.call_args
        history_sent = call_args[2]  # The 'messages_or_contents' argument
        last_message_text = extract_text_from_message(history_sent[-1])
        assert CONTINUATION_PROMPT in last_message_text

    def test_secondary_worker_api_error(self, mock_multichat_session, mocker):
        """
        Tests that the secondary worker correctly handles an API error
        and puts an error message in the result queue.
        """
        # Arrange
        mocker.patch(
            "aiterm.api_client.perform_chat_request",
            return_value=("API Error: Something went wrong", {}),
        )
        result_queue = queue.Queue()
        worker_engine = mock_multichat_session.engines["openai"]

        # Act
        mock_multichat_session._secondary_worker(
            worker_engine, "gpt-4o-mini", [], "system prompt", result_queue
        )

        # Assert
        result = result_queue.get_nowait()
        assert result["engine_name"] == "openai"
        assert "API Error: Something went wrong" in result["text"]

    def test_log_multichat_turn(self, mock_multichat_session, fake_fs):
        """
        Tests that a multi-chat turn is correctly logged to a file.
        """
        # Arrange
        log_path = config.CHATLOG_DIRECTORY / "multichat_test.jsonl"
        history_slice = [
            construct_user_message("openai", "User message", []),
            construct_assistant_message("openai", "AI response"),
        ]

        # Act
        mock_multichat_session._log_multichat_turn(log_path, history_slice)

        # Assert
        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
        assert "history_slice" in data
        assert len(data["history_slice"]) == 2
        assert data["history_slice"][0]["content"][0]["text"] == "User message"
