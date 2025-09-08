# tests/test_review.py
"""
Tests for the interactive review tool in aiterm/review.py.
"""

import argparse
import json
import os
from unittest.mock import patch

import pytest
from aiterm import config, review
from aiterm.utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
)


@pytest.fixture
def setup_review_files(fake_fs):
    """Sets up fake log and session files for review tests with predictable mtimes."""
    config.SESSIONS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config.CHATLOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Create files
    multichat_path = config.CHATLOG_DIRECTORY / "multichat_log_oldest.jsonl"
    multichat_content = [
        {"history_slice": [{"role": "user", "content": "Director: Go!"}]}
    ]
    multichat_path.write_text("\n".join(json.dumps(line) for line in multichat_content))

    log_path = config.CHATLOG_DIRECTORY / "chat_log_middle.jsonl"
    log_content = [
        {
            "prompt": {"role": "user", "content": "Hello"},
            "response": {"role": "assistant", "content": "Hi"},
        }
    ]
    log_path.write_text("\n".join(json.dumps(line) for line in log_content))

    session_path = config.SESSIONS_DIRECTORY / "session_newest.json"
    session_content = {
        "history": [
            construct_user_message("openai", "Let's start", []),
            construct_assistant_message("openai", "Ready!"),
        ]
    }
    session_path.write_text(json.dumps(session_content))

    # Set explicit, predictable modification times to control sorting
    os.utime(multichat_path, (1000, 1000))
    os.utime(log_path, (2000, 2000))
    os.utime(session_path, (3000, 3000))

    return {
        "log": log_path,
        "session": session_path,
        "multichat": multichat_path,
    }


class TestReviewTool:
    """Test suite for the review tool."""

    def test_get_turn_count(self, setup_review_files, fake_fs):
        """Tests turn counting for different file types."""
        # Test valid files
        assert review.get_turn_count(setup_review_files["log"]) == 1
        assert review.get_turn_count(setup_review_files["session"]) == 1
        # Test malformed file separately
        malformed_path = config.SESSIONS_DIRECTORY / "malformed.json"
        malformed_path.write_text("{not_json:")
        assert review.get_turn_count(malformed_path) == 0

    def test_replay_file_jsonl(self, setup_review_files, capsys, mocker):
        """Tests replaying a .jsonl log file."""
        mocker.patch("sys.stdin.isatty", return_value=True)
        with patch(
            "aiterm.review.get_single_char", return_value="q"
        ):  # Quit after first turn
            review.replay_file(setup_review_files["log"])
        captured = capsys.readouterr().out
        assert "You:" in captured
        assert "Hello" in captured

    def test_replay_file_json(self, setup_review_files, capsys, mocker):
        """Tests replaying a .json session file."""
        mocker.patch("sys.stdin.isatty", return_value=True)
        with patch("aiterm.review.get_single_char", return_value="c"):  # Continue
            review.replay_file(setup_review_files["session"])
        captured = capsys.readouterr().out
        assert "Let's start" in captured

    def test_replay_file_multichat(self, setup_review_files, capsys, mocker):
        """Tests replaying a multichat log file."""
        mocker.patch("sys.stdin.isatty", return_value=True)
        with patch("aiterm.review.get_single_char", return_value="c"):
            review.replay_file(setup_review_files["multichat"])
        captured = capsys.readouterr().out
        assert "Director:" in captured
        assert "Go!" in captured

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
        colliding_file = config.CHATLOG_DIRECTORY / "new_name.jsonl"
        fake_fs.create_file(colliding_file)
        with patch("builtins.input", return_value="new_name"):
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

    def test_main_direct_replay_mode(self, setup_review_files, mocker):
        """Tests that main calls replay_file directly when a file is provided."""
        mock_replay = mocker.patch("aiterm.review.replay_file")
        log_path = setup_review_files["log"]
        args = argparse.Namespace(file=log_path)
        review.main(args)
        mock_replay.assert_called_once_with(log_path)

    def test_get_single_char_fallback(self, mocker):
        """Tests the get_single_char returns None when stdin is not a TTY."""
        mocker.patch("sys.stdin.isatty", return_value=False)
        assert review.get_single_char() is None

    def test_action_menu_fallback(self, mocker):
        """Tests the action menu's fallback to input() when not a TTY."""
        mocker.patch("sys.stdin.isatty", return_value=False)
        mocker.patch("builtins.input", return_value="r ")
        choice = review.present_action_menu("Test", {"Replay": "r"})
        assert choice == "r"

    @patch("aiterm.review.present_numbered_menu")
    @patch("aiterm.review.present_action_menu")
    def test_main_interactive_loop(
        self, mock_action_menu, mock_numbered_menu, setup_review_files, mocker
    ):
        """Tests a simple interactive session: select, replay, then back."""
        # Sorted order: session (0), log (1), multichat (2). Quit is 3.
        # Select index 1 (log file) then quit (index 3)
        mock_numbered_menu.side_effect = [1, 3]
        # Actions: replay -> continue (post-replay) -> back to file list
        mock_action_menu.side_effect = ["r", "c", "b"]
        mock_replay = mocker.patch("aiterm.review.replay_file")
        args = argparse.Namespace(file=None)
        review.main(args)
        mock_replay.assert_called_once_with(setup_review_files["log"])
        assert mock_numbered_menu.call_count == 2
        assert mock_action_menu.call_count == 3

    @patch("aiterm.review.present_numbered_menu")
    @patch("aiterm.review.present_action_menu")
    def test_main_loop_reenter_session(
        self, mock_action_menu, mock_numbered_menu, setup_review_files, mocker
    ):
        """Tests selecting a session and choosing to re-enter."""
        # Sorted order: session (0), log (1), multichat (2). Select index 0.
        # The main loop will run only once because reenter_session causes main to return.
        mock_numbered_menu.return_value = 0
        mock_action_menu.return_value = "e"
        mock_reenter = mocker.patch("aiterm.review.reenter_session")
        args = argparse.Namespace(file=None)
        review.main(args)
        mock_reenter.assert_called_once_with(setup_review_files["session"])

    @patch("aiterm.review.present_numbered_menu")
    @patch("aiterm.review.present_action_menu")
    def test_main_loop_replay_then_delete(
        self, mock_action_menu, mock_numbered_menu, setup_review_files, mocker
    ):
        """Tests a sequence of replaying and then deleting a file."""
        # Sorted order: session (0), log (1), multichat (2). Quit is 3.
        # Main menu: select log file (index 1), then quit on the next loop.
        mock_numbered_menu.side_effect = [1, 3]
        # Action menu: 'replay', then post-replay 'delete'
        mock_action_menu.side_effect = ["r", "d"]
        mocker.patch("aiterm.review.replay_file")
        mock_delete = mocker.patch("aiterm.review.delete_file", return_value=True)
        args = argparse.Namespace(file=None)
        review.main(args)
        mock_delete.assert_called_once_with(setup_review_files["log"])
