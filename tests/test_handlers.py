# tests/test_handlers.py

import argparse
from pathlib import Path

import pytest

from src.aiterm import handlers
from src.aiterm.managers.context_manager import Attachment


# Mock dependencies that would be imported by handlers
@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    mocker.patch("src.aiterm.handlers.api_client")
    mocker.patch("src.aiterm.handlers.get_engine")
    mocker.patch("src.aiterm.handlers.persona_manager")
    mocker.patch("src.aiterm.handlers.ContextManager")
    mocker.patch("src.aiterm.handlers.SessionManager")
    mocker.patch("src.aiterm.handlers.SingleChatUI")
    mocker.patch("src.aiterm.handlers.MultiChatSession")
    mocker.patch("src.aiterm.handlers.MultiChatUI")
    mocker.patch("src.aiterm.handlers.workflows")
    mocker.patch("src.aiterm.handlers.settings", {"default_engine": "gemini"})
    mocker.patch("src.aiterm.handlers.resolve_config_precedence")
    mocker.patch("src.aiterm.handlers.read_system_prompt")
    mocker.patch("src.aiterm.handlers.config")
    yield


class TestHandlers:
    def test_handle_chat_single_shot(self, mock_dependencies):
        """Tests that single-shot mode runs correctly."""
        # Setup mock return values
        handlers.resolve_config_precedence.return_value = {
            "engine_name": "gemini",
            "model": "gemini-test",
            "max_tokens": 100,
            "stream": False,
            "memory_enabled": False,
            "debug_enabled": False,
            "persona": None,
            "session_name": None,
            "system_prompt_arg": None,
            "files_arg": [],
            "exclude_arg": [],
        }
        args = argparse.Namespace()
        handlers.handle_chat("hello", args)
        handlers.SessionManager.return_value.handle_single_shot.assert_called_once_with(
            "hello"
        )

    def test_handle_chat_interactive(self, mock_dependencies):
        """Tests that interactive mode runs correctly."""
        handlers.resolve_config_precedence.return_value = {
            "engine_name": "openai",
            "model": "gpt-test",
            "max_tokens": 200,
            "stream": True,
            "memory_enabled": True,
            "debug_enabled": True,
            "persona": None,
            "session_name": "my-session",
            "system_prompt_arg": "Be nice",
            "files_arg": [],
            "exclude_arg": [],
        }
        args = argparse.Namespace()
        handlers.handle_chat(None, args)  # None prompt means interactive
        handlers.SingleChatUI.assert_called_once()
        handlers.SingleChatUI.return_value.run.assert_called_once()

    def test_handle_load_session(self, mock_dependencies, mocker):
        """Tests the session loading handler."""
        # Mock the handle_load command function which does the actual loading logic
        mock_handle_load = mocker.patch("src.aiterm.handlers.handle_load")
        mock_handle_load.return_value = True  # Simulate successful load

        # Mock Path to control exists() check
        mocker.patch("src.aiterm.handlers.Path.is_absolute", return_value=True)

        filepath = "/path/to/session.json"
        handlers.handle_load_session(filepath)

        # Verify that handle_load was called with the session manager and path
        mock_handle_load.assert_called_once()
        # Verify that the UI was started
        handlers.SingleChatUI.assert_called_once()
        handlers.SingleChatUI.return_value.run.assert_called_once()

    def test_handle_multichat_session(self, mock_dependencies):
        """Tests the setup of a multichat session."""
        args = argparse.Namespace(
            file=[],
            exclude=[],
            system_prompt=None,
            model=None,
            max_tokens=None,
            debug=False,
            session_name="multi-test",
            persona_gpt=None,
            persona_gem=None,
        )
        handlers.handle_multichat_session(initial_prompt="compare these", args=args)
        handlers.MultiChatSession.assert_called_once()
        handlers.MultiChatUI.assert_called_once()
        handlers.MultiChatUI.return_value.run.assert_called_once()

        # Check if the initial prompt was passed to the UI
        _, kwargs = handlers.MultiChatUI.call_args
        assert kwargs["initial_prompt"] == "compare these"

    def test_handle_image_generation(self, mock_dependencies):
        """Tests the standalone image generation handler."""
        args = argparse.Namespace(model="dall-e-test", debug=True)
        handlers.handle_image_generation("a cat", args)
        handlers.workflows._perform_image_generation.assert_called_once()
        # Check that the arguments were passed correctly
        call_args, _ = handlers.workflows._perform_image_generation.call_args
        assert call_args[1] == "dall-e-test"  # model
        assert call_args[2] == "a cat"  # prompt
        assert call_args[3] is not None  # session_raw_logs

    def test_handle_image_generation_no_prompt(self, mocker, mock_dependencies):
        """Tests image generation when prompt is read from stdin."""
        args = argparse.Namespace(model="dall-e-test", debug=False)
        mocker.patch("sys.stdin.isatty", return_value=False)
        mocker.patch("sys.stdin.read", return_value="a dog")

        handlers.handle_image_generation(None, args)
        handlers.workflows._perform_image_generation.assert_called_once_with(
            handlers.api_client.check_api_keys.return_value,
            "dall-e-test",
            "a dog",
            session_raw_logs=None,
        )

    def test_handle_multichat_session_with_attachments(mocker, mock_dependencies):
        """Tests that multichat system prompts include attachment content."""
        args = argparse.Namespace(
            file=["/fake/file.txt"],
            exclude=[],
            system_prompt=None,
            model=None,
            max_tokens=None,
            debug=False,
            session_name=None,
            persona_gpt=None,
            persona_gem=None,
        )
        # Mock the context manager to return a file
        cm_instance = handlers.ContextManager.return_value
        cm_instance.attachments = {
            Path("/fake/file.txt"): Attachment(content="fake content", mtime=1.0)
        }
        cm_instance.image_data = []

        handlers.handle_multichat_session(initial_prompt=None, args=args)

        # Check that the MultiChatSession was initialized with the attachment
        _, kwargs = handlers.MultiChatSession.call_args
        state = kwargs["state"]
        assert Path("/fake/file.txt") in state.attachments
        assert state.attachments[Path("/fake/file.txt")].content == "fake content"
