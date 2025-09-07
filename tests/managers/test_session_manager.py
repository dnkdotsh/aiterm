# tests/managers/test_session_manager.py
"""
Tests for the SessionManager class in aiterm/managers/session_manager.py.
"""

from aiterm import config
from aiterm.utils.message_builder import (
    construct_user_message,
)


class TestSessionManager:
    """Test suite for the SessionManager."""

    def test_take_turn_simple(self, mocker, mock_session_manager):
        """Tests a standard, non-streaming conversational turn."""
        # Arrange
        mock_api_request = mocker.patch(
            "aiterm.api_client.perform_chat_request",
            return_value=(
                "AI response",
                {"prompt": 10, "completion": 20, "total": 30},
            ),
        )
        initial_history_len = len(mock_session_manager.state.history)

        # Act
        result = mock_session_manager.take_turn("User input", first_turn=False)

        # Assert
        assert result is True  # Turn should be logged
        mock_api_request.assert_called_once()
        # History should have user and assistant message appended
        assert len(mock_session_manager.state.history) == initial_history_len + 2
        assert mock_session_manager.state.history[-2]["role"] == "user"
        assert mock_session_manager.state.history[-1]["role"] == "model"
        # Token counts should be updated
        assert mock_session_manager.state.total_prompt_tokens == 10
        assert mock_session_manager.state.total_completion_tokens == 20
        assert mock_session_manager.state.last_turn_tokens["total"] == 30

    def test_take_turn_delegates_to_image_workflow(self, mocker, mock_session_manager):
        """
        Tests that user input is passed to the image workflow if it's active,
        and no API chat request is made.
        """
        # Arrange
        mock_api_request = mocker.patch("aiterm.api_client.perform_chat_request")
        mock_session_manager.image_workflow.img_prompt_crafting = True
        mock_session_manager.image_workflow.process_prompt_input.return_value = (
            False,
            None,
        )

        # Act
        result = mock_session_manager.take_turn("refine prompt", first_turn=False)

        # Assert
        assert result is False  # Turn should not be logged
        mock_session_manager.image_workflow.process_prompt_input.assert_called_once_with(
            "refine prompt"
        )
        mock_api_request.assert_not_called()

    def test_history_condensation_is_triggered(self, mocker, mock_session_manager):
        """
        Tests that _condense_chat_history is called when the history
        length threshold is reached.
        """
        # Arrange
        mocker.patch(
            "aiterm.api_client.perform_chat_request", return_value=("AI response", {})
        )
        # Mock the method we expect to be called
        mock_condense = mocker.patch.object(
            mock_session_manager, "_condense_chat_history"
        )
        # Create a history that is exactly one turn short of the threshold
        turn_threshold = config.HISTORY_SUMMARY_THRESHOLD_TURNS
        mock_session_manager.state.history = [
            {"role": "user", "content": f"msg {i}"}
            for i in range((turn_threshold - 1) * 2)
        ]

        # Act: This turn should push the history length to the threshold
        mock_session_manager.take_turn("Final prompt", first_turn=False)

        # Assert
        mock_condense.assert_called_once()

    def test_condense_chat_history_logic(self, mocker, mock_session_manager):
        """Tests that the history is correctly summarized and restructured."""
        # Arrange
        # Mock the helper request that the condensation method uses
        mock_helper_request = mocker.patch.object(
            mock_session_manager,
            "_perform_helper_request",
            return_value=("This is the summary.", {}),
        )
        # Create a long history to be condensed
        history = [
            construct_user_message("gemini", f"msg {i}", [])
            for i in range(config.HISTORY_SUMMARY_THRESHOLD_TURNS * 2)
        ]
        mock_session_manager.state.history = history
        trim_point = config.HISTORY_SUMMARY_TRIM_TURNS * 2
        original_last_turn = history[trim_point:]

        # Act
        mock_session_manager._condense_chat_history()

        # Assert
        mock_helper_request.assert_called_once()
        # The new history should start with a summary message
        assert (
            "[PREVIOUSLY DISCUSSED]"
            in mock_session_manager.state.history[0]["parts"][0]["text"]
        )
        assert (
            "This is the summary."
            in mock_session_manager.state.history[0]["parts"][0]["text"]
        )
        # The end of the history should be the preserved recent turns
        assert mock_session_manager.state.history[1:] == original_last_turn

    def test_cleanup_full_run(self, mocker, fake_fs, mock_session_manager):
        """Tests the default cleanup path where everything is run."""
        mock_consolidate = mocker.patch("aiterm.workflows.consolidate_memory")
        mock_rename_ai = mocker.patch("aiterm.workflows.rename_log_with_ai")
        mock_rename_file = mocker.patch.object(mock_session_manager, "_rename_log_file")
        log_path = config.CHATLOG_DIRECTORY / "chat_test.jsonl"
        fake_fs.create_file(log_path)
        mock_session_manager.state.history.append({"role": "user", "content": "test"})

        mock_session_manager.cleanup(session_name=None, log_filepath=log_path)

        mock_consolidate.assert_called_once()
        mock_rename_ai.assert_called_once()
        mock_rename_file.assert_not_called()  # AI rename is called instead

    def test_cleanup_force_quit_skips_all(self, mocker, fake_fs, mock_session_manager):
        """Tests that force_quit skips all cleanup operations."""
        mock_consolidate = mocker.patch("aiterm.workflows.consolidate_memory")
        mock_rename_ai = mocker.patch("aiterm.workflows.rename_log_with_ai")
        log_path = config.CHATLOG_DIRECTORY / "chat_test.jsonl"
        fake_fs.create_file(log_path)
        mock_session_manager.state.force_quit = True

        mock_session_manager.cleanup(session_name=None, log_filepath=log_path)

        mock_consolidate.assert_not_called()
        mock_rename_ai.assert_not_called()

    def test_cleanup_no_memory(self, mocker, fake_fs, mock_session_manager):
        """Tests that memory consolidation is skipped when memory is disabled."""
        mock_consolidate = mocker.patch("aiterm.workflows.consolidate_memory")
        mock_rename_ai = mocker.patch("aiterm.workflows.rename_log_with_ai")
        log_path = config.CHATLOG_DIRECTORY / "chat_test.jsonl"
        fake_fs.create_file(log_path)
        mock_session_manager.state.memory_enabled = False
        mock_session_manager.state.history.append({"role": "user", "content": "test"})

        mock_session_manager.cleanup(session_name=None, log_filepath=log_path)

        mock_consolidate.assert_not_called()
        mock_rename_ai.assert_called_once()

    def test_cleanup_with_custom_name(self, mocker, fake_fs, mock_session_manager):
        """Tests that a custom name is used for renaming, skipping the AI call."""
        mock_consolidate = mocker.patch("aiterm.workflows.consolidate_memory")
        mock_rename_ai = mocker.patch("aiterm.workflows.rename_log_with_ai")
        mock_rename_file = mocker.patch.object(mock_session_manager, "_rename_log_file")
        log_path = config.CHATLOG_DIRECTORY / "chat_test.jsonl"
        fake_fs.create_file(log_path)
        mock_session_manager.state.history.append({"role": "user", "content": "test"})
        mock_session_manager.state.custom_log_rename = "my-custom-name"

        mock_session_manager.cleanup(session_name=None, log_filepath=log_path)

        mock_consolidate.assert_called_once()
        mock_rename_ai.assert_not_called()
        mock_rename_file.assert_called_once_with(log_path, "my-custom-name")

    def test_handle_single_shot(self, mocker, mock_session_manager, capsys):
        """Tests a single, non-interactive request."""
        # Arrange
        mock_api_request = mocker.patch(
            "aiterm.api_client.perform_chat_request",
            return_value=(
                "Single shot response.",
                {"prompt": 5, "completion": 10, "total": 15},
            ),
        )
        mock_session_manager.state.stream_active = False

        # Act
        mock_session_manager.handle_single_shot("Test prompt")

        # Assert
        mock_api_request.assert_called_once()
        captured = capsys.readouterr()
        # Response should be printed to stdout
        assert "Single shot response." in captured.out
        # Token usage should be printed to stderr
        assert "[P:5/C:10/R:0/T:15]" in captured.err
