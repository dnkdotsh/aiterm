# tests/test_cli_entrypoint.py
"""
Tests for the command-line argument parsing and dispatching in aiterm/aiterm.py.
"""

import io
import sys

import pytest

# Import the main module we want to test
from aiterm import aiterm, handlers


@pytest.fixture(autouse=True)
def mock_handlers(mocker):
    """
    A fixture to automatically mock all handler functions.
    This isolates the tests to only the argument parsing and dispatching logic
    in aiterm.py, preventing the actual handler code from running.
    """
    mocker.patch("aiterm.handlers.handle_chat")
    mocker.patch("aiterm.handlers.handle_image_generation")
    mocker.patch("aiterm.handlers.handle_multichat_session")
    mocker.patch("aiterm.handlers.handle_load_session")
    mocker.patch("aiterm.api_client.check_api_keys", return_value="fake_key")
    mocker.patch("aiterm.bootstrap.ensure_project_structure")
    mocker.patch("aiterm.engine.get_engine")
    mocker.patch("aiterm.review.main")


class TestCLIEntrypoint:
    """Test suite for the main CLI entrypoint."""

    def test_no_args_interactive_chat(self, monkeypatch):
        """Tests that running `aiterm` with no args starts an interactive chat."""
        monkeypatch.setattr(sys, "argv", ["aiterm"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        aiterm.main()
        handlers.handle_chat.assert_called_once()
        # Check that it's an interactive call (initial_prompt is None)
        call_args, _ = handlers.handle_chat.call_args
        assert call_args[0] is None

    def test_prompt_arg_calls_handle_chat(self, monkeypatch):
        """Tests that `aiterm -p "prompt"` calls the chat handler."""
        monkeypatch.setattr(sys, "argv", ["aiterm", "--prompt", "hello world"])
        monkeypatch.setattr(
            "sys.stdin.isatty", lambda: True
        )  # Ensure stdin is not read
        aiterm.main()
        handlers.handle_chat.assert_called_once()
        # Check that the prompt is passed correctly
        call_args, _ = handlers.handle_chat.call_args
        assert call_args[0] == "hello world"

    def test_piped_input_calls_handle_chat(self, monkeypatch):
        """Tests that piped input is correctly passed as a prompt."""
        monkeypatch.setattr(sys, "argv", ["aiterm"])
        # Simulate piped input
        monkeypatch.setattr("sys.stdin", io.StringIO("piped content"))
        aiterm.main()
        handlers.handle_chat.assert_called_once()
        # Check that the piped content is used as the prompt
        call_args, _ = handlers.handle_chat.call_args
        assert call_args[0] == "piped content"

    def test_image_mode_calls_handle_image(self, monkeypatch):
        """Tests that `aiterm --image` calls the image handler."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["aiterm", "--image", "-p", "a test image", "--engine", "openai"],
        )
        aiterm.main()
        handlers.handle_image_generation.assert_called_once()
        handlers.handle_chat.assert_not_called()

    def test_both_mode_calls_multichat(self, monkeypatch):
        """Tests that `aiterm --both` calls the multichat handler."""
        monkeypatch.setattr(sys, "argv", ["aiterm", "--both", "a test prompt"])
        aiterm.main()
        handlers.handle_multichat_session.assert_called_once()
        handlers.handle_chat.assert_not_called()

    def test_load_mode_calls_load_session(self, monkeypatch):
        """Tests that `aiterm --load` calls the load session handler."""
        monkeypatch.setattr(sys, "argv", ["aiterm", "--load", "my_session.json"])
        aiterm.main()
        handlers.handle_load_session.assert_called_once_with("my_session.json")
        handlers.handle_chat.assert_not_called()

    def test_invalid_args_exits(self, monkeypatch):
        """Tests that an invalid argument causes a system exit."""
        monkeypatch.setattr(sys, "argv", ["aiterm", "--non-existent-arg"])
        with pytest.raises(SystemExit):
            aiterm.main()

    def test_image_with_gemini_exits(self, monkeypatch, capsys):
        """Tests that using --image with the gemini engine correctly exits."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["aiterm", "chat", "--image", "--prompt", "test", "--engine", "gemini"],
        )
        with pytest.raises(SystemExit) as excinfo:
            aiterm.main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: --image mode is only supported" in captured.err

    def test_review_command_dispatches_to_review_main(self, monkeypatch):
        """Tests that the `review` command calls review.main."""
        monkeypatch.setattr(sys, "argv", ["aiterm", "review"])
        aiterm.main()
        aiterm.review.main.assert_called_once()
        # Ensure chat handlers are not called
        handlers.handle_chat.assert_not_called()
        handlers.handle_load_session.assert_not_called()

    def test_chat_subcommand_is_handled_correctly(self, monkeypatch):
        """Tests that `aiterm chat -p "prompt"` works the same as `aiterm -p "prompt"`."""
        monkeypatch.setattr(
            sys, "argv", ["aiterm", "chat", "--prompt", "hello from chat"]
        )
        aiterm.main()
        handlers.handle_chat.assert_called_once()
        call_args, _ = handlers.handle_chat.call_args
        assert call_args[0] == "hello from chat"
