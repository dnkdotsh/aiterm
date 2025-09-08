# tests/test_handlers.py
"""
Tests for the handler functions in aiterm/handlers.py.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest
from aiterm import handlers
from aiterm.managers.context_manager import Attachment
from aiterm.personas import Persona


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies for the handlers module."""
    mocker.patch("aiterm.handlers.resolve_config_precedence")
    mocker.patch("aiterm.handlers.api_client.check_api_keys", return_value="fake_key")
    mocker.patch("aiterm.handlers.get_engine")
    mocker.patch("aiterm.handlers.ContextManager")
    mocker.patch("aiterm.handlers.SessionManager")
    mocker.patch("aiterm.handlers.SingleChatUI")
    mocker.patch("aiterm.handlers.MultiChatSession")
    mocker.patch("aiterm.handlers.MultiChatUI")
    mocker.patch("aiterm.handlers.MultiChatSessionState")
    mocker.patch("aiterm.handlers.read_system_prompt", return_value="System prompt.")


class TestHandleChat:
    """Tests for the handle_chat function."""

    def test_interactive_mode_setup_and_run(
        self, mock_dependencies, mock_config_params
    ):
        """
        Tests that handle_chat correctly sets up and runs an interactive session
        when no initial prompt is provided.
        """
        # Arrange
        args = argparse.Namespace(prompt=None)
        # The mock_config_params fixture provides a standard return value
        handlers.resolve_config_precedence.return_value = mock_config_params

        # Act
        handlers.handle_chat(initial_prompt=None, args=args)

        # Assert
        # 1. Config resolution was called
        handlers.resolve_config_precedence.assert_called_once_with(args)

        # 2. Dependencies were initialized with resolved config
        handlers.api_client.check_api_keys.assert_called_once_with("gemini")
        handlers.get_engine.assert_called_once_with("gemini", "fake_key")
        handlers.ContextManager.assert_called_once()
        handlers.SessionManager.assert_called_once()

        # 3. The UI was instantiated and run
        ui_instance = handlers.SingleChatUI.return_value
        handlers.SingleChatUI.assert_called_once_with(
            handlers.SessionManager.return_value, "test_session"
        )
        ui_instance.run.assert_called_once()

    def test_single_shot_mode_setup_and_run(
        self, mock_dependencies, mock_config_params
    ):
        """
        Tests that handle_chat correctly sets up and runs a single-shot session
        when an initial prompt is provided.
        """
        # Arrange
        args = argparse.Namespace(prompt="Hello")
        handlers.resolve_config_precedence.return_value = mock_config_params

        # Act
        handlers.handle_chat(initial_prompt="Hello", args=args)

        # Assert
        # 1. Session manager was created
        session_manager_instance = handlers.SessionManager.return_value
        handlers.SessionManager.assert_called_once()

        # 2. `handle_single_shot` was called on the manager
        session_manager_instance.handle_single_shot.assert_called_once_with("Hello")

        # 3. The interactive UI was NOT run
        handlers.SingleChatUI.assert_not_called()

    def test_cli_overrides_persona_in_context_setup(self, mocker):
        """
        Verifies that files from both CLI and a persona are correctly passed
        to the ContextManager.
        """
        # Arrange: More specific mocking for this test
        mocker.patch("aiterm.handlers.api_client.check_api_keys")
        mocker.patch("aiterm.handlers.get_engine")
        mocker.patch("aiterm.handlers.SessionManager")
        mocker.patch("aiterm.handlers.SingleChatUI")
        mock_context_manager = mocker.patch("aiterm.handlers.ContextManager")
        mocker.patch("aiterm.handlers.read_system_prompt", return_value="")

        # Simulate a persona with its own attachments
        mock_persona = Persona(
            name="Test Persona",
            filename="test.json",
            attachments=["/persona/file.txt"],
        )
        # Simulate CLI args with different attachments
        args = argparse.Namespace(
            prompt=None,
            file=["/cli/file.txt"],
            persona="test",
            engine="gemini",
            model="test-model",
            max_tokens=100,
            stream=True,
            memory=True,
            debug=False,
            session_name="test",
            system_prompt=None,
            exclude=[],
        )

        # Use the real resolve_config_precedence but mock the persona loader inside it
        with patch(
            "aiterm.utils.config_loader.personas.load_persona",
            return_value=mock_persona,
        ):
            # Act
            handlers.handle_chat(initial_prompt=None, args=args)

        # Assert that ContextManager was called with a list containing BOTH files
        mock_context_manager.assert_called_once()
        _, call_kwargs = mock_context_manager.call_args
        # The first positional argument is 'files_arg'
        passed_files = call_kwargs.get("files_arg")

        assert "/cli/file.txt" in passed_files
        assert "/persona/file.txt" in passed_files
        assert len(passed_files) == 2

    def test_large_attachment_warning_confirm(self, mocker, mock_dependencies):
        """Tests that the large attachment warning allows the user to proceed."""
        mocker.patch("aiterm.handlers.config.LARGE_ATTACHMENT_THRESHOLD_BYTES", 10)
        mocker.patch("builtins.input", return_value="y")
        args = argparse.Namespace(prompt=None)
        handlers.resolve_config_precedence.return_value = {
            "engine_name": "gemini",
            "model": "test",
            "persona": None,
            "files_arg": ["large_file.txt"],
            "memory_enabled": False,
            "exclude_arg": [],
            "system_prompt_arg": None,
            "max_tokens": 1024,
            "debug_enabled": False,
            "stream": True,
            "session_name": None,
        }
        # Configure mocks to have the necessary state for the check
        cm_instance = handlers.ContextManager.return_value
        cm_instance.attachments = {
            Path("large_file.txt"): Attachment("this content is very long", 1.0)
        }
        cm_instance.image_data = []
        session_instance = handlers.SessionManager.return_value
        session_instance.state.attachments = cm_instance.attachments

        handlers.handle_chat(initial_prompt=None, args=args)
        handlers.SingleChatUI.return_value.run.assert_called_once()

    def test_large_attachment_warning_cancel(self, mocker, mock_dependencies):
        """Tests that the large attachment warning allows the user to cancel."""
        mocker.patch("aiterm.handlers.config.LARGE_ATTACHMENT_THRESHOLD_BYTES", 10)
        mocker.patch("builtins.input", return_value="n")
        args = argparse.Namespace(prompt=None)
        handlers.resolve_config_precedence.return_value = {
            "engine_name": "gemini",
            "model": "test",
            "persona": None,
            "files_arg": ["large_file.txt"],
            "memory_enabled": False,
            "exclude_arg": [],
            "system_prompt_arg": None,
            "max_tokens": 1024,
            "debug_enabled": False,
            "stream": True,
            "session_name": None,
        }
        # Configure mocks to have the necessary state for the check
        cm_instance = handlers.ContextManager.return_value
        cm_instance.attachments = {
            Path("large_file.txt"): Attachment("this content is very long", 1.0)
        }
        cm_instance.image_data = []
        session_instance = handlers.SessionManager.return_value
        session_instance.state.attachments = cm_instance.attachments

        with pytest.raises(SystemExit) as excinfo:
            handlers.handle_chat(initial_prompt=None, args=args)
        assert excinfo.value.code == 0
        handlers.SingleChatUI.return_value.run.assert_not_called()


def test_handle_load_session_failure_exits(mocker):
    """Tests that if handle_load fails, the application exits."""
    mocker.patch("aiterm.handlers.get_engine")
    mocker.patch("aiterm.handlers.api_client.check_api_keys")
    mocker.patch("aiterm.handlers.SessionManager")
    mocker.patch("aiterm.handlers.SingleChatUI")
    # Make the command fail by patching its source location
    mocker.patch("aiterm.commands.handle_load", return_value=False)

    with pytest.raises(SystemExit) as excinfo:
        handlers.handle_load_session("fake.json")
    assert excinfo.value.code == 1


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
    )
    # Mock the context manager to return a file
    cm_instance = handlers.ContextManager.return_value
    cm_instance.attachments = {
        Path("/fake/file.txt"): Attachment(content="fake content", mtime=1.0)
    }
    cm_instance.image_data = []

    handlers.handle_multichat_session(initial_prompt=None, args=args)

    # Assert that the state object was created with the correct prompts
    handlers.MultiChatSessionState.assert_called_once()
    _, call_kwargs = handlers.MultiChatSessionState.call_args
    system_prompts = call_kwargs["system_prompts"]

    assert "--- ATTACHED FILES ---" in system_prompts["openai"]
    assert "fake content" in system_prompts["openai"]
    assert "--- ATTACHED FILES ---" in system_prompts["gemini"]
    assert "fake content" in system_prompts["gemini"]
