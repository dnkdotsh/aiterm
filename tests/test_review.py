# tests/test_review.py
"""
Tests for the interactive review tool in aiterm/review.py.
"""

import argparse
import json
from unittest.mock import patch

import pytest
from aiterm import config, review
from aiterm.utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
)


@pytest.fixture
def setup_review_files(fake_fs):
    """Sets up fake log and session files for review tests."""
    # Single-chat log file
    log_path = config.CHATLOG_DIRECTORY / "chat_log_1.jsonl"
    log_content = [
        {
            "prompt": {"role": "user", "content": "Hello"},
            "response": {"role": "assistant", "content": "Hi there"},
        },
        {
            "prompt": {"role": "user", "content": "Bye"},
            "response": {"role": "assistant", "content": "See you"},
        },
    ]
    log_path.write_text("\n".join(json.dumps(line) for line in log_content))

    # Single-chat session file
    session_path = config.SESSIONS_DIRECTORY / "session_1.json"
    session_content = {
        "history": [
            construct_user_message("openai", "Let's start", []),
            construct_assistant_message("openai", "Ready!"),
        ]
    }
    session_path.write_text(json.dumps(session_content))

    # Multi-chat log file for testing different replay format
    multichat_path = config.CHATLOG_DIRECTORY / "multichat_log.jsonl"
    multichat_content = [
        {
            "history_slice": [
                {"role": "user", "content": "Director: Go!"},
                {"role": "assistant", "content": "[OpenAI]: I am ready."},
                {"role": "assistant", "content": "[Gemini]: As am I."},
            ]
        }
    ]
    multichat_path.write_text("\n".join(json.dumps(line) for line in multichat_content))

    # Malformed file
    (config.SESSIONS_DIRECTORY / "malformed.json").write_text("{not_json:")

    return {
        "log": log_path,
        "session": session_path,
        "multichat": multichat_path,
        "malformed": config.SESSIONS_DIRECTORY / "malformed.json",
    }


class TestReviewTool:
    """Test suite for the review tool."""

    @pytest.mark.parametrize(
        "file_key, expected_turns",
        [
            ("log", 2),
            ("session", 1),
            ("malformed", 0),
        ],
    )
    def test_get_turn_count(self, setup_review_files, file_key, expected_turns):
        """Tests turn counting for different file types."""
        file_path = setup_review_files[file_key]
        assert review.get_turn_count(file_path) == expected_turns

    def test_replay_file_jsonl(self, setup_review_files, capsys, mocker):
        """Tests replaying a .jsonl log file."""
        # Ensure the function thinks it's in an interactive session
        mocker.patch("sys.stdin.isatty", return_value=True)
        with patch(
            "aiterm.review.get_single_char", return_value="q"
        ):  # Quit after first turn
            review.replay_file(setup_review_files["log"])
        captured = capsys.readouterr().out
        assert "You:" in captured
        assert "Hello" in captured
        assert "Assistant:" in captured
        assert "Hi there" in captured
        assert "Bye" not in captured  # Should have quit before this

    def test_replay_file_json(self, setup_review_files, capsys, mocker):
        """Tests replaying a .json session file."""
        mocker.patch("sys.stdin.isatty", return_value=True)
        with patch("aiterm.review.get_single_char", return_value="c"):  # Continue
            review.replay_file(setup_review_files["session"])
        captured = capsys.readouterr().out
        assert "Let's start" in captured
        assert "Ready!" in captured

    def test_replay_file_multichat(self, setup_review_files, capsys, mocker):
        """Tests replaying a multichat log file."""
        mocker.patch("sys.stdin.isatty", return_value=True)
        with patch("aiterm.review.get_single_char", return_value="c"):
            review.replay_file(setup_review_files["multichat"])
        captured = capsys.readouterr().out
        assert "Director:" in captured
        assert "Go!" in captured
        assert "AI:" in captured
        assert "[OpenAI]: I am ready." in captured
        assert "[Gemini]: As am I." in captured

    def test_rename_file_success(self, setup_review_files):
        """Tests successful file renaming."""
        log_path = setup_review_files["log"]
        new_path = log_path.with_name("new_log_name.jsonl")
        with patch("builtins.input", return_value="new log name"):
            result_path = review.rename_file(log_path)
        assert not log_path.exists()
        assert new_path.exists()
        assert result_path == new_path

    def test_rename_file_exists_error(self, setup_review_files, fake_fs, capsys):
        """Tests that renaming to an existing filename fails."""
        log_path = setup_review_files["log"]
        # Create a file with the target name to cause a collision
        colliding_file = config.CHATLOG_DIRECTORY / "session_1.jsonl"
        fake_fs.create_file(colliding_file)

        with patch("builtins.input", return_value="session_1"):
            result_path = review.rename_file(log_path)
        assert result_path is None
        captured = capsys.readouterr().err
        assert "already exists" in captured

    def test_delete_file_confirm(self, setup_review_files):
        """Tests file deletion with user confirmation."""
        log_path = setup_review_files["log"]
        with patch("aiterm.review.get_single_char", return_value="y"):
            was_deleted = review.delete_file(log_path)
        assert was_deleted is True
        assert not log_path.exists()

    def test_delete_file_cancel(self, setup_review_files):
        """Tests file deletion cancellation."""
        log_path = setup_review_files["log"]
        with patch("aiterm.review.get_single_char", return_value="n"):
            was_deleted = review.delete_file(log_path)
        assert was_deleted is False
        assert log_path.exists()

    def test_reenter_session(self, setup_review_files, mocker):
        """Tests that re-entering a session calls the correct subprocess."""
        mock_subprocess = mocker.patch("aiterm.review.subprocess.run")
        session_path = setup_review_files["session"]
        review.reenter_session(session_path)
        mock_subprocess.assert_called_once_with(
            ["aiterm", "--load", str(session_path)], check=True
        )

    def test_reenter_session_file_not_found_error(
        self, setup_review_files, mocker, capsys
    ):
        """Tests handling of FileNotFoundError for the aiterm command."""
        mocker.patch("aiterm.review.subprocess.run", side_effect=FileNotFoundError)
        session_path = setup_review_files["session"]
        review.reenter_session(session_path)
        captured = capsys.readouterr().err
        assert "'aiterm' command not found" in captured

    @patch("aiterm.review.present_numbered_menu")
    @patch("aiterm.review.present_action_menu")
    def test_main_interactive_loop(
        self, mock_action_menu, mock_numbered_menu, setup_review_files, mocker
    ):
        """Tests a simple interactive session: select, replay, then back."""
        # Mock the main selection menu to choose the first file (index 0)
        # Then, choose the last option ('Quit') on the second call.
        mock_numbered_menu.side_effect = [0, len(setup_review_files)]
        # Mock the action menu for the full sequence: replay -> continue -> back
        mock_action_menu.side_effect = ["r", "c", "b"]
        mock_replay = mocker.patch("aiterm.review.replay_file")

        args = argparse.Namespace(file=None)
        review.main(args)

        assert mock_numbered_menu.call_count == 2
        # The action menu is called 3 times in our sequence
        assert mock_action_menu.call_count == 3
        mock_replay.assert_called_once()

    def test_main_direct_replay_mode(self, setup_review_files, mocker):
        """Tests that main calls replay_file directly when a file is provided."""
        mock_replay = mocker.patch("aiterm.review.replay_file")
        log_path = setup_review_files["log"]
        args = argparse.Namespace(file=log_path)
        review.main(args)
        mock_replay.assert_called_once_with(log_path)
