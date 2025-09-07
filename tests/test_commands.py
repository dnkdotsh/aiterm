# tests/test_commands.py
"""
Tests for the interactive command handlers in aiterm/commands.py.
"""

from pathlib import Path
from unittest.mock import MagicMock

from aiterm import api_client, commands
from aiterm.engine import GeminiEngine, OpenAIEngine
from aiterm.managers.context_manager import Attachment
from aiterm.personas import Persona
from aiterm.utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
)


class TestCommands:
    """Test suite for slash command handlers."""

    def test_handle_exit(self, mock_session_manager):
        """Tests that /exit sets a custom log rename and returns True."""
        result = commands.handle_exit(["my", "custom", "name"], mock_session_manager)
        assert mock_session_manager.state.custom_log_rename == "my custom name"
        assert result is True

    def test_handle_quit(self, mock_session_manager):
        """Tests that /quit sets the force_quit flag and returns True."""
        result = commands.handle_quit([], mock_session_manager)
        assert mock_session_manager.state.force_quit is True
        assert result is True

    def test_handle_help(self, mocker, mock_session_manager, capsys):
        """Tests that /help calls the correct display function."""
        mock_display_help = mocker.patch("aiterm.commands.display_help")
        commands.handle_help([], mock_session_manager)
        mock_display_help.assert_called_once_with("chat")

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

    def test_handle_model(self, mock_session_manager, capsys):
        """Tests the /model command correctly sets the model."""
        commands.handle_model(["gpt-4-turbo"], mock_session_manager)
        assert mock_session_manager.state.model == "gpt-4-turbo"
        captured = capsys.readouterr()
        assert "Model set to: gpt-4-turbo" in captured.out

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

    def test_handle_history_empty(self, mock_session_manager, capsys):
        """Tests /history command when history is empty."""
        mock_session_manager.state.history = []
        commands.handle_history([], mock_session_manager)
        captured = capsys.readouterr()
        assert "History is empty." in captured.out

    def test_handle_history_populated(self, mock_session_manager, capsys):
        """Tests /history command with content."""
        mock_session_manager.state.history = [
            construct_user_message("openai", "Hello", []),
            construct_assistant_message("openai", "Hi there"),
        ]
        commands.handle_history([], mock_session_manager)
        captured = capsys.readouterr()
        assert "You:" in captured.out
        assert "Hello" in captured.out
        assert "Assistant:" in captured.out
        assert "Hi there" in captured.out

    def test_handle_state(self, mock_session_manager, capsys, mocker):
        """Tests that /state prints the current session state."""
        # Setup mock state for attachments
        mock_session_manager.state.attachments[Path("/fake/file.txt")] = Attachment(
            content="abc", mtime=123.0
        )
        # Mock the stat call for size calculation
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("pathlib.Path.stat", return_value=MagicMock(st_size=3))

        commands.handle_state([], mock_session_manager)
        captured = capsys.readouterr()
        assert "Engine: openai, Model: gpt-4o-mini" in captured.out
        assert "Max Tokens: 1024" in captured.out
        assert "Attached Text Files: 1 (3.00 B)" in captured.out
        assert "System Prompt: Active" in captured.out

    def test_handle_files(self, mock_session_manager):
        """Tests that /files calls the context manager's list_files method."""
        commands.handle_files([], mock_session_manager)
        mock_session_manager.context_manager.list_files.assert_called_once()

    def test_handle_print_file_found(self, mock_session_manager, capsys):
        """Tests /print for a file that exists in the attachments."""
        path = Path("/fake/file.txt")
        mock_session_manager.state.attachments[path] = Attachment(
            content="hello world", mtime=123.45
        )
        commands.handle_print(["file.txt"], mock_session_manager)
        captured = capsys.readouterr()
        assert "--- Content of file.txt ---" in captured.out
        assert "hello world" in captured.out

    def test_handle_print_file_not_found(self, mock_session_manager, capsys):
        """Tests /print for a file that is not attached."""
        commands.handle_print(["nonexistent.txt"], mock_session_manager)
        captured = capsys.readouterr()
        assert "No attached file named 'nonexistent.txt'" in captured.out

    def test_handle_personas_list(self, mocker, mock_session_manager, capsys):
        """Tests that /personas lists the available personas."""
        mock_personas = [
            Persona(
                name="Coder",
                filename="coder.json",
                description="A coding assistant.",
            ),
            Persona(
                name="Writer",
                filename="writer.json",
                description="A writing assistant.",
            ),
        ]
        mocker.patch(
            "aiterm.commands.persona_manager.list_personas", return_value=mock_personas
        )
        commands.handle_personas([], mock_session_manager)
        captured = capsys.readouterr()
        assert "--- Available Personas ---" in captured.out
        assert "- coder: A coding assistant." in captured.out
        assert "- writer: A writing assistant." in captured.out

    def test_handle_persona_clear(self, mock_session_manager):
        """Tests clearing an active persona."""
        mock_session_manager.state.current_persona = Persona(
            name="Test", filename="test.json"
        )
        mock_session_manager.state.system_prompt = "Persona prompt"
        mock_session_manager.state.initial_system_prompt = "Original prompt"
        mock_session_manager.state.persona_attachments = {
            Path("/fake/persona_file.txt")
        }
        mock_session_manager.state.attachments[Path("/fake/persona_file.txt")] = (
            Attachment(content="test", mtime=1)
        )
        assert len(mock_session_manager.state.history) == 0

        commands.handle_persona(["clear"], mock_session_manager)

        assert mock_session_manager.state.current_persona is None
        assert mock_session_manager.state.system_prompt == "Original prompt"
        assert not mock_session_manager.state.persona_attachments
        assert (
            Path("/fake/persona_file.txt") not in mock_session_manager.state.attachments
        )
        assert (
            "Persona cleared"
            in mock_session_manager.state.history[-1]["content"][0]["text"]
        )
        assert mock_session_manager.context_manager.attachments == {}

    def test_handle_persona_apply(self, mocker, mock_session_manager):
        """Tests applying a new persona with attachments and settings."""
        # Mock `handle_engine` as its test is separate
        mocker.patch("aiterm.commands.handle_engine")
        # Mock the context manager used inside the handler to simulate finding attachments
        mock_temp_context = MagicMock()
        mock_temp_context.attachments = {
            Path("/path/to/doc.md"): Attachment(content="doc", mtime=1)
        }
        mocker.patch("aiterm.commands.ContextManager", return_value=mock_temp_context)

        # Mock the persona that will be loaded
        new_persona = Persona(
            name="NewGuy",
            filename="newguy.json",
            engine="gemini",
            model="gemini-pro",
            attachments=["/path/to/doc.md"],
            system_prompt="You are NewGuy.",
        )
        mocker.patch(
            "aiterm.commands.persona_manager.load_persona", return_value=new_persona
        )

        # Set an initial state with a different persona's attachments
        mock_session_manager.state.persona_attachments = {Path("/old/file.txt")}
        mock_session_manager.state.attachments = {
            Path("/old/file.txt"): Attachment(content="old", mtime=1),
            Path("/user/file.txt"): Attachment(content="user", mtime=1),
        }

        commands.handle_persona(["newguy"], mock_session_manager)

        # 1. Assert old persona attachments are gone
        assert Path("/old/file.txt") not in mock_session_manager.state.attachments
        # 2. Assert new persona attachments are added
        assert Path("/path/to/doc.md") in mock_session_manager.state.attachments
        # 3. Assert user-added attachments remain
        assert Path("/user/file.txt") in mock_session_manager.state.attachments
        # 4. Assert persona attachment tracking is updated
        assert mock_session_manager.state.persona_attachments == {
            Path("/path/to/doc.md")
        }
        # 5. Assert persona settings are applied
        commands.handle_engine.assert_called_with(["gemini"], mock_session_manager)
        assert mock_session_manager.state.model == "gemini-pro"
        assert mock_session_manager.state.system_prompt == "You are NewGuy."
        assert mock_session_manager.state.current_persona == new_persona
        # 6. Assert the context manager is synced at the end
        assert (
            mock_session_manager.context_manager.attachments
            == mock_session_manager.state.attachments
        )

    def test_handle_detach(self, mock_session_manager):
        """Tests that /detach calls the context manager and adds a system message."""
        detached_path = Path("/fake/file.txt")
        mock_session_manager.context_manager.detach.return_value = [detached_path]
        initial_history_len = len(mock_session_manager.state.history)

        commands.handle_detach(["/fake/file.txt"], mock_session_manager)

        mock_session_manager.context_manager.detach.assert_called_once_with(
            "/fake/file.txt"
        )
        assert len(mock_session_manager.state.history) == initial_history_len + 1
        last_message = mock_session_manager.state.history[-1]
        assert "[SYSTEM]" in last_message["content"][0]["text"]
        assert "detached" in last_message["content"][0]["text"]
        assert "'file.txt'" in last_message["content"][0]["text"]
