# tests/test_commands.py
"""
Tests for the interactive command handlers in aiterm/commands.py.
"""

from aiterm import api_client, commands
from aiterm.engine import GeminiEngine, OpenAIEngine


class TestCommands:
    """Test suite for slash command handlers."""

    def test_handle_stream(self, mock_session_manager):
        """Tests the /stream command toggles the stream_active state."""
        initial_state = mock_session_manager.state.stream_active
        commands.handle_stream([], mock_session_manager)
        assert mock_session_manager.state.stream_active is not initial_state
        commands.handle_stream([], mock_session_manager)
        assert mock_session_manager.state.stream_active is initial_state

    def test_handle_debug(self, mock_session_manager):
        """Tests the /debug command toggles the debug_active state."""
        initial_state = mock_session_manager.state.debug_active
        commands.handle_debug([], mock_session_manager)
        assert mock_session_manager.state.debug_active is not initial_state
        commands.handle_debug([], mock_session_manager)
        assert mock_session_manager.state.debug_active is initial_state

    def test_handle_max_tokens(self, mock_session_manager):
        """Tests the /max-tokens command correctly sets the value."""
        assert mock_session_manager.state.max_tokens == 1024  # from fixture
        commands.handle_max_tokens(["2048"], mock_session_manager)
        assert mock_session_manager.state.max_tokens == 2048

    def test_handle_max_tokens_invalid(self, mock_session_manager, capsys):
        """Tests the /max-tokens command with an invalid argument."""
        commands.handle_max_tokens(["not_a_number"], mock_session_manager)
        assert mock_session_manager.state.max_tokens == 1024  # Unchanged
        captured = capsys.readouterr()
        assert "Usage: /max-tokens <number>" in captured.out

    def test_handle_model(self, mock_session_manager, capsys):
        """Tests the /model command correctly sets the model."""
        commands.handle_model(["gpt-4-turbo"], mock_session_manager)
        assert mock_session_manager.state.model == "gpt-4-turbo"
        captured = capsys.readouterr()
        assert "Model set to: gpt-4-turbo" in captured.out

    def test_handle_clear_confirm(self, mock_session_manager, mock_prompt_toolkit):
        """Tests that /clear proceeds when the user confirms."""
        mock_session_manager.state.history.append({"role": "user", "content": "test"})
        mock_session_manager.state.total_prompt_tokens = 10
        mock_session_manager.state.total_completion_tokens = 20
        mock_session_manager.state.last_turn_tokens = {"prompt": 1}
        mock_prompt_toolkit["input_queue"].append("proceed")

        commands.handle_clear([], mock_session_manager)

        assert not mock_session_manager.state.history
        assert mock_session_manager.state.total_prompt_tokens == 0
        assert mock_session_manager.state.total_completion_tokens == 0
        assert not mock_session_manager.state.last_turn_tokens

    def test_handle_clear_cancel(self, mock_session_manager, mock_prompt_toolkit):
        """Tests that /clear is cancelled when the user does not confirm."""
        mock_session_manager.state.history.append({"role": "user", "content": "test"})
        mock_session_manager.state.total_prompt_tokens = 10
        mock_prompt_toolkit["input_queue"].append("cancel")

        commands.handle_clear([], mock_session_manager)

        assert mock_session_manager.state.history  # Unchanged
        assert mock_session_manager.state.total_prompt_tokens == 10  # Unchanged

    def test_handle_engine_switch(self, mocker, mock_session_manager, capsys):
        """Tests a successful switch of the AI engine."""
        # Initial state is openai engine from fixture
        assert isinstance(mock_session_manager.state.engine, OpenAIEngine)

        # Mock dependencies for the switch
        mocker.patch("aiterm.api_client.check_api_keys", return_value="fake_gemini_key")
        mocker.patch(
            "aiterm.commands.get_engine", return_value=GeminiEngine("fake_gemini_key")
        )
        mocker.patch(
            "aiterm.commands.get_default_model_for_engine",
            return_value="gemini-1.5-flash",
        )
        mocker.patch("aiterm.commands.translate_history", return_value=[])

        commands.handle_engine(["gemini"], mock_session_manager)

        assert isinstance(mock_session_manager.state.engine, GeminiEngine)
        assert mock_session_manager.state.model == "gemini-1.5-flash"
        api_client.check_api_keys.assert_called_with("gemini")
        captured = capsys.readouterr()
        assert "Engine switched to Gemini" in captured.out

    def test_handle_engine_switch_no_key(self, mocker, mock_session_manager, capsys):
        """Tests that switching engines fails if the API key is missing."""
        mocker.patch(
            "aiterm.api_client.check_api_keys",
            side_effect=api_client.MissingApiKeyError("No key"),
        )

        commands.handle_engine(["gemini"], mock_session_manager)

        # State should not change
        assert isinstance(mock_session_manager.state.engine, OpenAIEngine)
        captured = capsys.readouterr()
        assert "Switch failed: No key" in captured.out
