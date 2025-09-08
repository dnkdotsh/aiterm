# tests/test_chat_ui.py
"""
Tests for the user interface components in aiterm/chat_ui.py.
"""

import json
import os
from pathlib import Path

from aiterm import commands, config
from aiterm.personas import Persona
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style


class TestSingleChatUI:
    """Test suite for the SingleChatUI class."""

    def test_create_style_from_theme_valid(self, mocker, mock_chat_ui):
        """Tests that a valid theme dictionary creates a Style object."""
        mocker.patch(
            "aiterm.chat_ui.theme_manager.ACTIVE_THEME",
            {"style_bottom_toolbar_background": "bg:#111 #fff"},
        )
        style = mock_chat_ui._create_style_from_theme()
        assert isinstance(style, Style)

    def test_create_style_from_theme_invalid(self, mocker, mock_chat_ui, caplog):
        """Tests that an invalid theme value is handled gracefully."""
        mocker.patch(
            "aiterm.chat_ui.theme_manager.ACTIVE_THEME",
            {"style_bottom_toolbar_background": "not-a-valid-style"},
        )
        style = mock_chat_ui._create_style_from_theme()
        # Should fall back to an empty style
        assert isinstance(style, Style)
        assert not style.style_rules
        assert "Invalid theme style format" in caplog.text

    def test_get_bottom_toolbar_content_full(
        self, mocker, mock_chat_ui, mock_prompt_toolkit_app
    ):
        """Tests toolbar generation with all components enabled."""
        mock_chat_ui.session.state.last_turn_tokens = {"prompt": 10, "total": 25}
        mock_chat_ui.session.state.current_persona = Persona(
            name="Test", filename="test.json"
        )
        mocker.patch(
            "aiterm.chat_ui.app_settings.settings",
            {
                "toolbar_show_total_io": True,
                "toolbar_show_model": True,
                "toolbar_show_persona": True,
                "toolbar_show_live_tokens": True,
                "toolbar_priority_order": "tokens,live,model,persona,io",
                "toolbar_separator": " | ",
            },
        )
        mocker.patch(
            "shutil.get_terminal_size", return_value=os.terminal_size((120, 24))
        )
        mock_prompt_toolkit_app["app"].current_buffer.text = "live prompt"

        content = mock_chat_ui._get_bottom_toolbar_content()
        content_text = "".join([part[1] for part in content])

        assert "[P:10/C:0/R:15/T:25]" in content_text
        assert "Session I/O" in content_text
        assert "Model: gemini-1.5-flash" in content_text
        assert "Persona: Test" in content_text
        assert "Live Context" in content_text

    def test_get_bottom_toolbar_content_respects_width(
        self, mocker, mock_chat_ui, mock_prompt_toolkit_app
    ):
        """Tests that the toolbar truncates content when terminal width is small."""
        # Use non-zero values to ensure the token string is generated
        mock_chat_ui.session.state.last_turn_tokens = {
            "prompt": 5,
            "completion": 10,
            "total": 15,
        }
        mocker.patch(
            "aiterm.chat_ui.app_settings.settings",
            {
                "toolbar_show_total_io": True,
                "toolbar_show_model": True,
                "toolbar_show_persona": False,
                "toolbar_show_live_tokens": True,
                "toolbar_priority_order": "tokens,live,model,io",
                "toolbar_separator": " | ",
            },
        )
        mocker.patch(
            "shutil.get_terminal_size", return_value=os.terminal_size((40, 24))
        )

        content = mock_chat_ui._get_bottom_toolbar_content()
        content_text = "".join([part[1] for part in content])

        # Check for the correct, non-zero token string
        assert "[P:5/C:10/R:0/T:15]" in content_text
        assert "Live Context" in content_text
        assert "Model: gemini-1.5-flash" not in content_text  # Should be truncated
        assert "Session I/O" not in content_text  # Should be truncated

    def test_handle_slash_command_known(self, mocker, mock_chat_ui):
        """Tests that a known command is dispatched correctly."""
        mock_handler = mocker.MagicMock(return_value=False)
        mocker.patch.dict(commands.COMMAND_MAP, {"/test": mock_handler})
        history = InMemoryHistory()

        result = mock_chat_ui._handle_slash_command("/test arg1 arg2", history)

        assert result is False
        mock_handler.assert_called_once_with(["arg1", "arg2"], mock_chat_ui.session)

    def test_handle_slash_command_save(self, mocker, mock_chat_ui):
        """Tests special handling for the /save command."""
        mock_save_handler = mocker.MagicMock(return_value=True)
        mocker.patch.dict(commands.COMMAND_MAP, {"/save": mock_save_handler})
        history = InMemoryHistory()

        result = mock_chat_ui._handle_slash_command("/save my-session", history)
        assert result is True
        mock_save_handler.assert_called_once_with(
            ["my-session"], mock_chat_ui.session, history
        )

    def test_handle_slash_command_unknown(self, mock_chat_ui, capsys):
        """Tests that an unknown command prints an error message."""
        history = InMemoryHistory()
        result = mock_chat_ui._handle_slash_command("/unknown", history)
        assert result is False
        captured = capsys.readouterr().out
        assert "Unknown command: /unknown" in captured

    def test_log_turn_success(self, fake_fs, mock_chat_ui):
        """Tests that a turn is correctly logged to a file."""
        log_path = config.CHATLOG_DIRECTORY / "test_log.jsonl"
        mock_chat_ui.session.state.history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        mock_chat_ui.session.state.model = "test-model"

        mock_chat_ui._log_turn(log_path)

        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
        assert data["model"] == "test-model"
        assert data["prompt"]["content"] == "Hello"
        assert data["response"]["content"] == "Hi"

    def test_log_turn_os_error(self, fake_fs, mock_chat_ui, caplog):
        """Tests graceful handling of an OSError during logging."""
        log_path = config.CHATLOG_DIRECTORY / "test_log.jsonl"
        # Make directory read-only to cause an error
        config.CHATLOG_DIRECTORY.chmod(0o555)
        mock_chat_ui.session.state.history = [{"role": "user"}, {"role": "assistant"}]

        mock_chat_ui._log_turn(log_path)
        assert "Could not write to session log file" in caplog.text


class TestMultiChatUI:
    """Test suite for the MultiChatUI class."""

    def test_handle_slash_command_ai_target(self, mock_multichat_ui):
        """Tests that the special /ai command is dispatched to process_turn."""
        history = InMemoryHistory()
        log_path = Path("/fake/log.jsonl")

        result = mock_multichat_ui._handle_slash_command(
            "/ai gpt question", history, log_path
        )

        assert result is False
        mock_multichat_ui.session.process_turn.assert_called_once_with(
            "/ai gpt question", log_path
        )

    def test_handle_slash_command_known(self, mocker, mock_multichat_ui):
        """Tests that a known command is dispatched to the multichat map."""
        mock_handler = mocker.MagicMock(return_value=True)
        mocker.patch.dict(commands.MULTICHAT_COMMAND_MAP, {"/exit": mock_handler})
        history = InMemoryHistory()
        log_path = Path("/fake/log.jsonl")

        result = mock_multichat_ui._handle_slash_command(
            "/exit name", history, log_path
        )

        assert result is True
        mock_handler.assert_called_once_with(
            ["name"], mock_multichat_ui.session, history
        )

    def test_handle_slash_command_unknown(self, mock_multichat_ui, capsys):
        """Tests that an unknown command prints an error."""
        history = InMemoryHistory()
        log_path = Path("/fake/log.jsonl")
        result = mock_multichat_ui._handle_slash_command("/unknown", history, log_path)
        assert result is False
        assert "Unknown command: /unknown" in capsys.readouterr().out
