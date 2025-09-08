# tests/test_commands.py
"""
Tests for the interactive command handlers in aiterm/commands.py.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from aiterm import api_client, commands, config
from aiterm.engine import GeminiEngine, OpenAIEngine
from aiterm.managers.context_manager import Attachment
from aiterm.personas import Persona
from aiterm.utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
    extract_text_from_message,
)
from prompt_toolkit.history import InMemoryHistory


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
        commands.handle_model(["gemini-pro"], mock_session_manager)
        assert mock_session_manager.state.model == "gemini-pro"
        captured = capsys.readouterr()
        assert "Model set to: gemini-pro" in captured.out

    def test_handle_engine_switch(self, mocker, mock_openai_session_manager, capsys):
        """Tests a successful switch of the AI engine."""
        # Initial state is openai engine from fixture
        assert isinstance(mock_openai_session_manager.state.engine, OpenAIEngine)

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

        commands.handle_engine(["gemini"], mock_openai_session_manager)

        assert isinstance(mock_openai_session_manager.state.engine, GeminiEngine)
        assert mock_openai_session_manager.state.model == "gemini-1.5-flash"
        api_client.check_api_keys.assert_called_with("gemini")
        captured = capsys.readouterr()
        assert "Engine switched to Gemini" in captured.out

    def test_handle_engine_switch_no_key(
        self, mocker, mock_openai_session_manager, capsys
    ):
        """Tests that switching engines fails if the API key is missing."""
        mocker.patch(
            "aiterm.api_client.check_api_keys",
            side_effect=api_client.MissingApiKeyError("No key"),
        )

        commands.handle_engine(["gemini"], mock_openai_session_manager)

        # State should not change
        assert isinstance(mock_openai_session_manager.state.engine, OpenAIEngine)
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
            construct_user_message("gemini", "Hello", []),
            construct_assistant_message("gemini", "Hi there"),
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
        assert "Engine: gemini, Model: gemini-1.5-flash" in captured.out
        assert "Max Tokens: 1024" in captured.out
        assert "Attached Text Files: 1 (3.00 B)" in captured.out
        assert "System Prompt: Active" in captured.out

    def test_handle_save_auto_name(self, mocker, mock_session_manager, fake_fs):
        """Tests that /save with no name calls the AI for a name."""
        # Use mocker.patch.object to mock the method on the real SessionManager instance
        mocker.patch.object(
            mock_session_manager,
            "_perform_helper_request",
            return_value=("test_ai_name", {}),
        )
        mock_history = InMemoryHistory()
        commands.handle_save([], mock_session_manager, mock_history)
        mock_session_manager._perform_helper_request.assert_called_once()
        saved_file = config.SESSIONS_DIRECTORY / "test_ai_name.json"
        assert saved_file.exists()

    def test_handle_save_auto_name_failure(
        self, mocker, mock_session_manager, fake_fs, capsys
    ):
        """Tests that /save cancels if the AI fails to generate a name."""
        mocker.patch.object(
            mock_session_manager, "_perform_helper_request", return_value=(None, {})
        )
        mock_history = InMemoryHistory()
        result = commands.handle_save([], mock_session_manager, mock_history)
        assert result is False
        captured = capsys.readouterr()
        assert "Could not auto-generate a name. Save cancelled." in captured.out

    def test_handle_save_write_error(
        self, mocker, mock_session_manager, fake_fs, capsys
    ):
        """Tests that an OSError during file save is handled."""
        mocker.patch("os.rename", side_effect=OSError("Permission denied"))
        mock_history = InMemoryHistory()
        result = commands.handle_save(
            ["my-session"], mock_session_manager, mock_history
        )
        assert result is False
        captured = capsys.readouterr()
        assert "Error saving session: Permission denied" in captured.out

    def test_handle_save_with_flags(self, mock_session_manager, fake_fs):
        """Tests the --stay and --remember flags for /save."""
        mock_history = InMemoryHistory()
        result = commands.handle_save(
            ["test-session", "--stay", "--remember"], mock_session_manager, mock_history
        )
        assert result is False  # --stay returns False to keep session running
        assert mock_session_manager.state.exit_without_memory is False  # --remember
        # The filename "test-session" is sanitized to "test_session"
        saved_file = config.SESSIONS_DIRECTORY / "test_session.json"
        assert saved_file.exists()

    def test_handle_load_invalid_file(self, mock_session_manager, fake_fs, capsys):
        """Tests that loading an invalid or malformed JSON fails gracefully."""
        invalid_file = config.SESSIONS_DIRECTORY / "invalid.json"
        invalid_file.write_text("this is not json")
        result = commands.handle_load(["invalid.json"], mock_session_manager)
        assert result is False
        captured = capsys.readouterr()
        assert "Error loading session" in captured.out

    def test_handle_load_missing_keys(self, mock_session_manager, fake_fs, capsys):
        """Tests loading a session file that is missing required keys."""
        incomplete_file = config.SESSIONS_DIRECTORY / "incomplete.json"
        incomplete_file.write_text('{"history": []}')  # Missing engine_name and model
        result = commands.handle_load(["incomplete.json"], mock_session_manager)
        assert result is False
        captured = capsys.readouterr()
        assert "Invalid session file" in captured.out

    def test_handle_load_multichat_in_single_chat(
        self, mock_session_manager, fake_fs, capsys
    ):
        """Tests that loading a multichat session into a single chat fails."""
        multichat_file = config.SESSIONS_DIRECTORY / "multi.json"
        multichat_data = {
            "session_type": "multichat",
            "engine_name": "openai",
            "model": "gpt-4",
        }
        multichat_file.write_text(json.dumps(multichat_data))
        result = commands.handle_load(["multi.json"], mock_session_manager)
        assert result is False
        captured = capsys.readouterr()
        assert "Cannot load a multi-chat session here" in captured.out

    def test_handle_image_delegates_to_workflow(self, mock_session_manager):
        """Tests that the /image command delegates to the image workflow."""
        args = ["a", "test", "prompt"]
        commands.handle_image(args, mock_session_manager)
        mock_session_manager.image_workflow.run.assert_called_once_with(args)

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
        assert "Persona cleared" in extract_text_from_message(
            mock_session_manager.state.history[-1]
        )
        assert mock_session_manager.context_manager.attachments == {}

    def test_handle_persona_apply(self, mocker, mock_openai_session_manager):
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
        mock_openai_session_manager.state.persona_attachments = {Path("/old/file.txt")}
        mock_openai_session_manager.state.attachments = {
            Path("/old/file.txt"): Attachment(content="old", mtime=1),
            Path("/user/file.txt"): Attachment(content="user", mtime=1),
        }

        commands.handle_persona(["newguy"], mock_openai_session_manager)

        # 1. Assert old persona attachments are gone
        assert (
            Path("/old/file.txt") not in mock_openai_session_manager.state.attachments
        )
        # 2. Assert new persona attachments are added
        assert Path("/path/to/doc.md") in mock_openai_session_manager.state.attachments
        # 3. Assert user-added attachments remain
        assert Path("/user/file.txt") in mock_openai_session_manager.state.attachments
        # 4. Assert persona attachment tracking is updated
        assert mock_openai_session_manager.state.persona_attachments == {
            Path("/path/to/doc.md")
        }
        # 5. Assert persona settings are applied
        commands.handle_engine.assert_called_with(
            ["gemini"], mock_openai_session_manager
        )
        assert mock_openai_session_manager.state.model == "gemini-pro"
        assert mock_openai_session_manager.state.system_prompt == "You are NewGuy."
        assert mock_openai_session_manager.state.current_persona == new_persona
        # 6. Assert the context manager is synced at the end
        assert (
            mock_openai_session_manager.context_manager.attachments
            == mock_openai_session_manager.state.attachments
        )

    def test_handle_persona_not_found(self, mock_session_manager, mocker, capsys):
        """Tests applying a persona that does not exist."""
        mocker.patch("aiterm.commands.persona_manager.load_persona", return_value=None)
        commands.handle_persona(["nonexistent"], mock_session_manager)
        captured = capsys.readouterr()
        assert "Persona 'nonexistent' not found." in captured.out

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
        last_message_text = extract_text_from_message(
            mock_session_manager.state.history[-1]
        )
        assert "[SYSTEM]" in last_message_text
        assert "detached" in last_message_text
        assert "'file.txt'" in last_message_text

    def test_handle_set_usage_error(self, mock_session_manager, capsys):
        """Tests /set with the wrong number of arguments."""
        commands.handle_set(["one_arg"], mock_session_manager)
        captured = capsys.readouterr()
        assert "Usage: /set <key> <value>" in captured.out

    def test_handle_set_unknown_key(self, mock_session_manager, capsys):
        """Tests /set with an unknown setting key."""
        commands.handle_set(["unknown_key", "some_value"], mock_session_manager)
        captured = capsys.readouterr()
        assert "Unknown setting: 'unknown_key'" in captured.out

    def test_handle_set_invalid_value(self, mock_session_manager, capsys):
        """Tests /set with an invalid value for the setting's type."""
        commands.handle_set(["stream", "not_a_bool"], mock_session_manager)
        captured = capsys.readouterr()
        assert "Error: Invalid boolean value" in captured.out

    def test_handle_toolbar_no_args(self, mock_session_manager, capsys, mocker):
        """Tests /toolbar with no args prints current settings."""
        mocker.patch(
            "aiterm.commands.settings",
            {
                "toolbar_enabled": True,
                "toolbar_priority_order": "a,b,c",
                "toolbar_separator": " - ",
                "toolbar_show_total_io": False,
                "toolbar_show_live_tokens": True,
                "toolbar_show_model": True,
                "toolbar_show_persona": False,
            },
        )
        commands.handle_toolbar([], mock_session_manager)
        captured = capsys.readouterr()
        assert "Toolbar Enabled: True" in captured.out
        assert "Component Order: a,b,c" in captured.out
        assert "Show Session I/O: False" in captured.out
        assert "Show Persona: False" in captured.out

    def test_handle_toolbar_on_off(
        self, mock_session_manager, mocker, mock_prompt_toolkit_app
    ):
        """Tests `/toolbar on` and `/toolbar off`."""
        mock_save = mocker.patch(
            "aiterm.commands.save_setting", return_value=(True, "OK")
        )
        commands.handle_toolbar(["on"], mock_session_manager)
        mock_save.assert_called_with("toolbar_enabled", "on")
        assert mock_prompt_toolkit_app["app"].invalidate.called

        commands.handle_toolbar(["off"], mock_session_manager)
        mock_save.assert_called_with("toolbar_enabled", "off")

    def test_handle_toolbar_toggle(
        self, mock_session_manager, mocker, mock_prompt_toolkit_app
    ):
        """Tests `/toolbar toggle <component>`."""
        mocker.patch("aiterm.commands.settings", {"toolbar_show_model": True})
        mock_save = mocker.patch(
            "aiterm.commands.save_setting", return_value=(True, "OK")
        )
        commands.handle_toolbar(["toggle", "model"], mock_session_manager)
        # Should toggle from True to False
        mock_save.assert_called_with("toolbar_show_model", "False")
        assert mock_prompt_toolkit_app["app"].invalidate.called

    def test_handle_theme_no_args(self, mock_session_manager, capsys, mocker):
        """Tests /theme with no args lists available themes."""
        mocker.patch(
            "aiterm.commands.settings",
            {"active_theme": "default"},
        )
        mocker.patch(
            "aiterm.commands.theme_manager.list_themes",
            return_value={"default": "The default theme.", "solarized": "A light one."},
        )
        commands.handle_theme([], mock_session_manager)
        captured = capsys.readouterr()
        assert "--- Available Themes ---" in captured.out
        assert "> default: The default theme." in captured.out
        assert "  solarized: A light one." in captured.out

    def test_handle_theme_apply_success(
        self, mock_session_manager, mocker, mock_prompt_toolkit_app
    ):
        """Tests successfully applying a new theme."""
        mocker.patch(
            "aiterm.commands.theme_manager.list_themes", return_value={"solarized": "d"}
        )
        mock_save = mocker.patch(
            "aiterm.commands.save_setting", return_value=(True, "Theme set!")
        )
        mock_reload = mocker.patch("aiterm.commands.theme_manager.reload_theme")

        commands.handle_theme(["solarized"], mock_session_manager)
        mock_save.assert_called_with("active_theme", "solarized")
        mock_reload.assert_called_once()
        assert mock_prompt_toolkit_app["app"].invalidate.called
        assert mock_session_manager.state.ui_refresh_needed is True

    def test_handle_theme_not_found(self, mock_session_manager, mocker, capsys):
        """Tests applying a theme that does not exist."""
        mocker.patch("aiterm.commands.theme_manager.list_themes", return_value={})
        commands.handle_theme(["nonexistent"], mock_session_manager)
        captured = capsys.readouterr()
        assert "Theme 'nonexistent' not found" in captured.out

    def test_handle_forget_turns(self, mock_session_manager):
        """Tests /forget N to remove the last N turns."""
        mock_session_manager.state.history = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "turn 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "turn 2"},
        ]
        commands.handle_forget(["1"], mock_session_manager)
        assert len(mock_session_manager.state.history) == 2
        assert mock_session_manager.state.history[-1]["content"] == "turn 1"

    def test_handle_forget_default_one_turn(self, mock_session_manager):
        """Tests that /forget with no args removes one turn."""
        mock_session_manager.state.history = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "turn 1"},
        ]
        commands.handle_forget([], mock_session_manager)
        assert len(mock_session_manager.state.history) == 0

    def test_handle_forget_too_many_turns(self, mock_session_manager, capsys):
        """Tests /forget N where N is larger than the history."""
        mock_session_manager.state.history = [{"role": "user", "content": "turn 1"}]
        commands.handle_forget(["2"], mock_session_manager)
        assert len(mock_session_manager.state.history) == 1
        captured = capsys.readouterr()
        assert "Cannot forget 2 turns" in captured.out

    def test_handle_forget_memory(self, mock_session_manager, mocker):
        """Tests /forget <topic> --memory delegates to the workflow."""
        mock_scrub = mocker.patch("aiterm.commands.workflows.scrub_memory")
        commands.handle_forget(["aiterm", "tool", "--memory"], mock_session_manager)
        mock_scrub.assert_called_once_with(mock_session_manager, "aiterm tool")


class TestMultiChatCommands:
    """Test suite for multi-chat specific slash command handlers."""

    def test_handle_multichat_exit(self, mock_multichat_session):
        """Tests /exit for multi-chat."""
        mock_history = InMemoryHistory()
        result = commands.handle_multichat_exit(
            ["my-log"], mock_multichat_session, mock_history
        )
        assert result is True
        assert mock_multichat_session.state.custom_log_rename == "my-log"

    def test_handle_multichat_quit(self, mock_multichat_session):
        """Tests /quit for multi-chat."""
        mock_history = InMemoryHistory()
        result = commands.handle_multichat_quit(
            [], mock_multichat_session, mock_history
        )
        assert result is True
        assert mock_multichat_session.state.force_quit is True

    def test_handle_multichat_help(self, mocker, mock_multichat_session):
        """Tests /help for multi-chat."""
        mock_history = InMemoryHistory()
        mock_display_help = mocker.patch("aiterm.commands.display_help")
        commands.handle_multichat_help([], mock_multichat_session, mock_history)
        mock_display_help.assert_called_once_with("multichat")

    def test_handle_multichat_model(self, mock_multichat_session, capsys):
        """Tests /model for multi-chat."""
        # Test setting gemini model
        commands.handle_multichat_model(
            ["gem", "gemini-pro"], mock_multichat_session, None
        )
        assert mock_multichat_session.state.gemini_model == "gemini-pro"
        captured = capsys.readouterr()
        assert "Gemini model set to: gemini-pro" in captured.out

        # Test setting openai model
        commands.handle_multichat_model(
            ["gpt", "gpt-4-turbo"], mock_multichat_session, None
        )
        assert mock_multichat_session.state.openai_model == "gpt-4-turbo"
        captured = capsys.readouterr()
        assert "OpenAI model set to: gpt-4-turbo" in captured.out

    def test_handle_multichat_clear(self, mock_multichat_session, mock_prompt_toolkit):
        """Tests /clear for multi-chat."""
        mock_multichat_session.state.shared_history = ["some history"]
        mock_prompt_toolkit["input_queue"].append("proceed")
        commands.handle_multichat_clear([], mock_multichat_session, None)
        assert not mock_multichat_session.state.shared_history

    def test_handle_multichat_save(self, mock_multichat_session, fake_fs):
        """Tests /save for multi-chat."""
        mock_history = InMemoryHistory()
        result = commands.handle_multichat_save(
            ["my-session"], mock_multichat_session, mock_history
        )
        assert result is True  # Should exit by default
        # The filename "my-session" is sanitized to "my_session"
        saved_file = config.SESSIONS_DIRECTORY / "my_session.json"
        assert saved_file.exists()
        with open(saved_file) as f:
            data = json.load(f)
            assert data["session_type"] == "multichat"
