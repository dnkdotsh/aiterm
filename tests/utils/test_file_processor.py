# tests/utils/test_file_processor.py
"""
Tests for the file processing utilities in aiterm/utils/file_processor.py.
"""

from pathlib import Path

import pytest
from aiterm import config
from aiterm.utils import file_processor


class TestFileProcessor:
    """Test suite for file processing utilities."""

    def test_read_system_prompt_os_error(self, mocker, caplog):
        """Tests that an OSError when reading a system prompt file is handled."""
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("pathlib.Path.is_file", return_value=True)
        mocker.patch("pathlib.Path.read_text", side_effect=OSError("Permission denied"))
        # The function should fall back to treating the path as the prompt string itself
        result = file_processor.read_system_prompt("path/to/file.txt")
        assert result == "path/to/file.txt"
        assert "Could not read system prompt file" in caplog.text

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("archive.zip", True),
            ("archive.tar", True),
            ("archive.gz", True),
            ("archive.tgz", True),
            ("archive.tar.gz", True),
            ("document.txt", False),
            ("archive.rar", False),
        ],
    )
    def test_is_supported_archive_file(self, filename, expected):
        """Tests the archive file support check."""
        path = Path(filename)
        assert file_processor.is_supported_archive_file(path) is expected

    def test_save_image_and_get_path_no_session(self, fake_fs, mocker):
        """Tests the image filename format when no session_name is provided."""
        # Mock datetime within the file_processor module to get a predictable timestamp
        mock_datetime = mocker.patch("aiterm.utils.file_processor.datetime")
        mock_datetime.datetime.now.return_value.strftime.return_value = (
            "20250101_120000"
        )

        prompt = "a blue car"
        image_bytes = b"fakedata"
        path = file_processor.save_image_and_get_path(prompt, image_bytes, None)

        assert path.parent == config.IMAGE_DIRECTORY
        assert path.name == "image_a_blue_car_20250101_120000.png"
        assert path.read_bytes() == image_bytes

        # Also verify that our mock was called as expected
        mock_datetime.datetime.now.assert_called_once()
        mock_datetime.datetime.now.return_value.strftime.assert_called_once_with(
            "%Y%m%d_%H%M%S"
        )

    def test_log_image_generation_os_error(self, mocker, caplog):
        """Tests that an OSError when writing to the image log is handled."""
        # Mock `open` to raise an error when called
        mocker.patch("builtins.open", side_effect=OSError("Disk full"))
        file_processor.log_image_generation("dall-e-3", "prompt", "path", "session")
        assert "Could not write to image log file: Disk full" in caplog.text
