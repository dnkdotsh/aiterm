# tests/managers/test_context_manager.py
"""
Tests for the ContextManager class in aiterm/managers/context_manager.py.
"""

from pathlib import Path

import pytest
from aiterm import config
from aiterm.managers.context_manager import ContextManager


@pytest.fixture
def setup_fake_fs(fake_fs):
    """Sets up a common file structure for context manager tests."""
    # Create persistent memory file
    fake_fs.create_file(
        config.PERSISTENT_MEMORY_FILE, contents="Initial memory content."
    )

    # Create a project directory structure
    proj_dir = Path("/project")
    fake_fs.create_dir(proj_dir)
    fake_fs.create_file(proj_dir / "main.py", contents="print('hello')")
    fake_fs.create_file(proj_dir / "README.md", contents="# Project")
    fake_fs.create_file(proj_dir / "config.json", contents='{"key": "value"}')

    # Create a subdirectory with more files
    utils_dir = proj_dir / "utils"
    fake_fs.create_dir(utils_dir)
    fake_fs.create_file(utils_dir / "helpers.py", contents="def helper(): pass")
    fake_fs.create_file(utils_dir / "data.bin", contents="binary_data")  # Ignored

    # An excluded directory
    build_dir = proj_dir / "build"
    fake_fs.create_dir(build_dir)
    fake_fs.create_file(build_dir / "output.txt", contents="build output")

    return proj_dir


class TestContextManager:
    """Test suite for the ContextManager."""

    def test_init_with_single_file(self, setup_fake_fs):
        """Tests initialization with a single file path."""
        main_py_path = str(setup_fake_fs / "main.py")
        cm = ContextManager(
            files_arg=[main_py_path], memory_enabled=False, exclude_arg=None
        )

        assert len(cm.attachments) == 1
        assert Path(main_py_path) in cm.attachments
        assert cm.attachments[Path(main_py_path)].content == "print('hello')"
        assert cm.memory_content is None

    def test_init_with_directory(self, setup_fake_fs):
        """Tests initialization with a directory, including recursion."""
        proj_dir_path = str(setup_fake_fs)
        cm = ContextManager(
            files_arg=[proj_dir_path], memory_enabled=False, exclude_arg=None
        )

        # Should find 5 text files, ignoring the binary file
        assert len(cm.attachments) == 5
        assert setup_fake_fs / "main.py" in cm.attachments
        assert setup_fake_fs / "README.md" in cm.attachments
        assert setup_fake_fs / "config.json" in cm.attachments
        assert setup_fake_fs / "utils" / "helpers.py" in cm.attachments
        assert setup_fake_fs / "build" / "output.txt" in cm.attachments
        assert setup_fake_fs / "utils" / "data.bin" not in cm.attachments

    def test_init_with_memory_enabled(self, setup_fake_fs):
        """Tests that persistent memory is loaded when enabled."""
        cm = ContextManager(files_arg=None, memory_enabled=True, exclude_arg=None)
        assert cm.memory_content == "Initial memory content."

    def test_init_with_exclusions(self, setup_fake_fs):
        """Tests that excluded files and directories are ignored."""
        proj_dir_path = str(setup_fake_fs)
        exclusions = [
            str(setup_fake_fs / "config.json"),  # Exclude a file
            str(setup_fake_fs / "build"),  # Exclude a directory
        ]
        cm = ContextManager(
            files_arg=[proj_dir_path], memory_enabled=False, exclude_arg=exclusions
        )

        assert len(cm.attachments) == 3
        assert setup_fake_fs / "config.json" not in cm.attachments
        assert setup_fake_fs / "build" / "output.txt" not in cm.attachments
        assert setup_fake_fs / "main.py" in cm.attachments

    def test_refresh_files_modified(self, setup_fake_fs):
        """Tests that refresh_files re-reads a modified file."""
        main_py_path = setup_fake_fs / "main.py"
        cm = ContextManager(
            files_arg=[str(main_py_path)], memory_enabled=False, exclude_arg=None
        )

        # Modify the file on the fake filesystem
        main_py_path.write_text("print('updated')")

        updated_files = cm.refresh_files(None)
        assert updated_files == ["main.py"]
        assert cm.attachments[main_py_path].content == "print('updated')"

    def test_refresh_files_deleted(self, setup_fake_fs):
        """Tests that refresh_files removes a deleted file from context."""
        main_py_path = setup_fake_fs / "main.py"
        cm = ContextManager(
            files_arg=[str(main_py_path)], memory_enabled=False, exclude_arg=None
        )

        # Delete the file
        main_py_path.unlink()

        updated_files = cm.refresh_files(None)
        assert "main.py" not in updated_files
        assert main_py_path not in cm.attachments

    def test_attach_file(self, setup_fake_fs):
        """Tests adding a file after initialization."""
        cm = ContextManager(files_arg=[], memory_enabled=False, exclude_arg=None)
        assert len(cm.attachments) == 0

        readme_path_str = str(setup_fake_fs / "README.md")
        cm.attach_file(readme_path_str)

        assert len(cm.attachments) == 1
        assert Path(readme_path_str) in cm.attachments
        assert "# Project" in cm.attachments[Path(readme_path_str)].content

    def test_detach_file_by_name(self, setup_fake_fs):
        """Tests detaching a file by its base name."""
        cm = ContextManager(
            files_arg=[str(setup_fake_fs)], memory_enabled=False, exclude_arg=None
        )
        assert len(cm.attachments) == 5
        main_py_path = setup_fake_fs / "main.py"
        assert main_py_path in cm.attachments

        detached_paths = cm.detach("main.py")
        assert len(cm.attachments) == 4
        assert main_py_path not in cm.attachments
        assert detached_paths == [main_py_path]

    def test_detach_directory(self, setup_fake_fs):
        """Tests detaching all files within a directory."""
        cm = ContextManager(
            files_arg=[str(setup_fake_fs)], memory_enabled=False, exclude_arg=None
        )
        assert len(cm.attachments) == 5
        utils_dir_path_str = str(setup_fake_fs / "utils")
        helpers_py_path = setup_fake_fs / "utils" / "helpers.py"
        assert helpers_py_path in cm.attachments

        detached_paths = cm.detach(utils_dir_path_str)
        assert len(cm.attachments) == 4
        assert helpers_py_path not in cm.attachments
        assert detached_paths == [helpers_py_path]
