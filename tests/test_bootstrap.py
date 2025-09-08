# tests/test_bootstrap.py
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiterm import bootstrap, config
from aiterm.theme_manager import USER_THEMES_DIR


@pytest.fixture
def mock_bootstrap_deps(mocker):
    """Mocks dependencies used within the bootstrap module."""
    # Prevent actual resource copying, we'll fake the source files instead
    mocker.patch("shutil.copy2")
    # Mock user input functions
    mocker.patch("builtins.input", return_value="y")
    mocker.patch("getpass.getpass", side_effect=["test_openai_key", "test_gemini_key"])
    # Mock persona creation as it's tested elsewhere
    mocker.patch("aiterm.bootstrap.persona_manager.create_default_persona_if_missing")


class TestBootstrap:
    """Test suite for the bootstrap module."""

    def test_perform_first_run_setup(self, fake_fs, mock_bootstrap_deps, mocker):
        """
        Tests the complete first-run setup process, verifying directory creation,
        .env file generation, and default content copying.
        """
        # Arrange
        # 1. Simulate the packaged resources that bootstrap tries to copy
        fake_themes_src_path = "/fake_pkg/aiterm/themes"
        fake_fs.create_file(f"{fake_themes_src_path}/default.json")
        fake_docs_src_path = "/fake_pkg/aiterm/docs_src"
        fake_fs.create_file(f"{fake_docs_src_path}/assistant_docs.md")

        # 2. Mock importlib.resources to find these fake files
        mock_theme_file = MagicMock()
        mock_theme_file.name = "default.json"
        mock_doc_file = MagicMock()
        mock_doc_file.name = "assistant_docs.md"

        mock_files = mocker.patch("importlib.resources.files")

        def files_side_effect(module_path):
            mock_traversable = MagicMock()
            if module_path == "aiterm.themes":
                mock_traversable.iterdir.return_value = [mock_theme_file]
            elif module_path == "aiterm.docs_src":
                mock_traversable.joinpath.return_value = mock_doc_file
            return mock_traversable

        mock_files.side_effect = files_side_effect

        mock_as_file = mocker.patch("importlib.resources.as_file")

        @contextmanager
        def as_file_side_effect(traversable):
            if traversable.name == "default.json":
                yield Path(f"{fake_themes_src_path}/default.json")
            elif traversable.name == "assistant_docs.md":
                yield Path(f"{fake_docs_src_path}/assistant_docs.md")

        mock_as_file.side_effect = as_file_side_effect

        # Act: The fake filesystem is empty, so this will trigger a first run.
        # We need to remove the directories created by the fixture.
        shutil.rmtree(config.CONFIG_DIR)
        bootstrap._perform_first_run_setup()

        # Assert directories were created
        assert config.CONFIG_DIR.exists()
        assert config.DATA_DIR.exists()
        assert config.LOG_DIRECTORY.exists()
        assert config.CHATLOG_DIRECTORY.exists()
        assert config.SESSIONS_DIRECTORY.exists()
        assert config.PERSONAS_DIRECTORY.exists()
        assert config.DOCS_DIRECTORY.exists()
        assert USER_THEMES_DIR.exists()

        # Assert .env file was created correctly
        assert config.DOTENV_FILE.exists()
        env_content = config.DOTENV_FILE.read_text()
        assert 'OPENAI_API_KEY="test_openai_key"' in env_content
        assert 'GEMINI_API_KEY="test_gemini_key"' in env_content
        assert os.stat(config.DOTENV_FILE).st_mode & 0o777 == 0o600

        # Assert default content was "copied"
        bootstrap.persona_manager.create_default_persona_if_missing.assert_called()
        assert shutil.copy2.call_count >= 2

    def test_ensure_project_structure_not_first_run(self, fake_fs, mock_bootstrap_deps):
        """
        Tests that `ensure_project_structure` creates missing subdirectories
        without triggering the full first-run setup if the main config dir exists.
        """
        # Arrange: The fake_fs fixture creates everything. We want to simulate a partial setup.
        # So, we remove some sub-directories but keep the main one.
        shutil.rmtree(config.PERSONAS_DIRECTORY)
        # The .env file may not exist, so handle potential FileNotFoundError
        if config.DOTENV_FILE.exists():
            os.remove(config.DOTENV_FILE)

        assert config.CONFIG_DIR.exists()  # This is true from the fixture
        assert not config.PERSONAS_DIRECTORY.exists()
        assert not config.DOTENV_FILE.exists()

        # Act
        bootstrap.ensure_project_structure()

        # Assert: The first-run input prompts should NOT have been called
        assert not bootstrap.getpass.getpass.called

        # Assert: Missing sub-directories and files should have been created silently
        assert config.PERSONAS_DIRECTORY.exists()
        assert config.DOTENV_FILE.exists()
        env_content = config.DOTENV_FILE.read_text()
        assert 'OPENAI_API_KEY=""' in env_content

        # Assert default persona creation is still called
        bootstrap.persona_manager.create_default_persona_if_missing.assert_called()

    def test_migrate_chat_logs(self, fake_fs):
        """
        Tests that old chat logs are correctly moved from LOG_DIRECTORY to
        CHATLOG_DIRECTORY.
        """
        # Arrange: The fake_fs fixture creates both dirs. Remove the target to simulate pre-migration state.
        shutil.rmtree(config.CHATLOG_DIRECTORY)
        assert config.LOG_DIRECTORY.exists()
        assert not config.CHATLOG_DIRECTORY.exists()

        fake_fs.create_file(config.LOG_DIRECTORY / "chat_log_1.jsonl")
        fake_fs.create_file(config.LOG_DIRECTORY / "multichat_log.jsonl")
        fake_fs.create_file(
            config.LOG_DIRECTORY / "aiterm.log"
        )  # This one should NOT move

        # Act
        bootstrap._migrate_chat_logs()

        # Assert
        assert config.CHATLOG_DIRECTORY.exists()
        assert (config.CHATLOG_DIRECTORY / "chat_log_1.jsonl").exists()
        assert (config.CHATLOG_DIRECTORY / "multichat_log.jsonl").exists()
        assert not (config.LOG_DIRECTORY / "chat_log_1.jsonl").exists()
        assert (config.LOG_DIRECTORY / "aiterm.log").exists()
