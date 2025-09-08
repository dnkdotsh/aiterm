# tests/managers/test_context_manager.py
"""
Tests for the ContextManager class in aiterm/managers/context_manager.py.
"""

import os
import tarfile
import zipfile
from io import BytesIO
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
    fake_fs.create_file(utils_dir / "logo.png", contents=b"\x89PNG\r\n\x1a\n")  # Png

    # An excluded directory
    build_dir = proj_dir / "build"
    fake_fs.create_dir(build_dir)
    fake_fs.create_file(build_dir / "output.txt", contents="build output")

    # Create a ZIP archive within the fake filesystem
    zip_path = proj_dir / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("zip_folder/zipped_file.txt", "zipped content")
        zf.writestr("zip_folder/ignored.dat", "binary")

    # Create a TAR.GZ archive within the fake filesystem
    tar_path = proj_dir / "archive.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tarinfo = tarfile.TarInfo(name="tar_folder/tarred_file.txt")
        file_content = b"tarred content"
        tarinfo.size = len(file_content)
        tf.addfile(tarinfo, BytesIO(file_content))

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

        # 5 text files + 2 archives = 7 attachments
        assert len(cm.attachments) == 7
        assert setup_fake_fs / "main.py" in cm.attachments
        assert setup_fake_fs / "README.md" in cm.attachments
        assert setup_fake_fs / "utils" / "helpers.py" in cm.attachments
        # 1 image file
        assert len(cm.image_data) == 1
        assert cm.image_data[0]["mime_type"] == "image/png"

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

        assert len(cm.attachments) == 5  # 3 files + 2 archives
        assert setup_fake_fs / "config.json" not in cm.attachments
        assert setup_fake_fs / "build" / "output.txt" not in cm.attachments
        assert setup_fake_fs / "main.py" in cm.attachments

    def test_process_zip_file(self, setup_fake_fs):
        """Tests that zip files are correctly processed."""
        zip_path_str = str(setup_fake_fs / "archive.zip")
        cm = ContextManager(
            files_arg=[zip_path_str], memory_enabled=False, exclude_arg=None
        )
        zip_path = Path(zip_path_str)
        assert zip_path in cm.attachments
        attachment = cm.attachments[zip_path]
        assert "zipped content" in attachment.content
        assert "zipped_file.txt" in attachment.content
        assert "ignored.dat" not in attachment.content

    def test_process_tar_file(self, setup_fake_fs):
        """Tests that tar.gz files are correctly processed."""
        tar_path_str = str(setup_fake_fs / "archive.tar.gz")
        cm = ContextManager(
            files_arg=[tar_path_str], memory_enabled=False, exclude_arg=None
        )
        tar_path = Path(tar_path_str)
        assert tar_path in cm.attachments
        attachment = cm.attachments[tar_path]
        assert "tarred content" in attachment.content
        assert "tarred_file.txt" in attachment.content

    def test_process_image_file(self, setup_fake_fs):
        """Tests that image files are correctly processed and added to image_data."""
        img_path_str = str(setup_fake_fs / "utils" / "logo.png")
        cm = ContextManager(
            files_arg=[img_path_str], memory_enabled=False, exclude_arg=None
        )
        assert len(cm.image_data) == 1
        assert cm.image_data[0]["mime_type"] == "image/png"
        assert isinstance(cm.image_data[0]["data"], str)  # Should be base64 string
        # Check that the text attachments are empty
        assert not cm.attachments

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

        cm.refresh_files(None)
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
            files_arg=[str(setup_fake_fs / "main.py")],
            memory_enabled=False,
            exclude_arg=None,
        )
        assert len(cm.attachments) == 1
        main_py_path = setup_fake_fs / "main.py"
        assert main_py_path in cm.attachments

        detached_paths = cm.detach("main.py")
        assert len(cm.attachments) == 0
        assert main_py_path not in cm.attachments
        assert detached_paths == [main_py_path]

    def test_detach_directory(self, setup_fake_fs):
        """Tests detaching all files within a directory."""
        cm = ContextManager(
            files_arg=[str(setup_fake_fs / "utils")],
            memory_enabled=False,
            exclude_arg=None,
        )
        assert len(cm.attachments) == 1
        helpers_py_path = setup_fake_fs / "utils" / "helpers.py"
        assert helpers_py_path in cm.attachments

        detached_paths = cm.detach(str(setup_fake_fs / "utils"))
        assert len(cm.attachments) == 0
        assert helpers_py_path not in cm.attachments
        assert detached_paths == [helpers_py_path]

    def test_list_files_output(self, setup_fake_fs, capsys):
        """Tests the formatted output of list_files."""
        cm = ContextManager(
            files_arg=[str(setup_fake_fs / "utils"), str(setup_fake_fs / "main.py")],
            memory_enabled=False,
            exclude_arg=None,
        )
        cm.list_files()
        captured = capsys.readouterr().out
        # Check for tree structure and file names
        assert f"{os.path.sep}project{os.path.sep}" in captured
        assert "+-- main.py" in captured
        assert "+-- utils" in captured
        assert "   +-- helpers.py" in captured
        # Check for byte formatting
        assert "(14.00 B)" in captured  # main.py
        assert "(18.00 B)" in captured  # helpers.py
