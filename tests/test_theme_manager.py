# tests/test_theme_manager.py
"""
Tests for the theme management module in aiterm/theme_manager.py.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiterm import theme_manager
from aiterm.theme_manager import USER_THEMES_DIR


@pytest.fixture
def mock_packaged_themes(mocker, fake_fs):
    """Mocks importlib.resources to simulate finding packaged theme files."""
    # 1. Create fake in-memory content and write to fake filesystem
    # This ensures the files exist for the `as_file` context manager to yield.
    packaged_themes_content = {
        "default.json": json.dumps({"description": "Packaged Default"}),
        "solarized.json": json.dumps({"description": "Packaged Solarized"}),
    }
    for name, content in packaged_themes_content.items():
        fake_fs.create_file(f"/fake_pkg/aiterm/themes/{name}", contents=content)

    # 2. Mock the top-level `files` and `as_file` functions
    mock_files = mocker.patch("importlib.resources.files")
    mock_as_file = mocker.patch("importlib.resources.as_file")

    # 3. Create mock traversable objects for iterdir()
    mock_default_traversable = MagicMock()
    mock_default_traversable.name = "default.json"
    mock_solarized_traversable = MagicMock()
    mock_solarized_traversable.name = "solarized.json"

    # 4. Define side effects
    def files_side_effect(module_path):
        if module_path == "aiterm.themes":
            mock_pkg_root = MagicMock()
            mock_pkg_root.iterdir.return_value = [
                mock_default_traversable,
                mock_solarized_traversable,
            ]
            # Handle the `traversable / "filename.json"` operation
            mock_pkg_root.__truediv__ = lambda self, child: {
                "default.json": mock_default_traversable,
                "solarized.json": mock_solarized_traversable,
            }.get(child, MagicMock())
            return mock_pkg_root
        return MagicMock()

    @contextmanager
    def as_file_side_effect(traversable):
        # Yield the path to the file on the fake filesystem
        yield Path(f"/fake_pkg/aiterm/themes/{traversable.name}")

    mock_files.side_effect = files_side_effect
    mock_as_file.side_effect = as_file_side_effect


@pytest.mark.usefixtures("mock_packaged_themes")
class TestThemeManager:
    """Test suite for the theme manager."""

    def test_get_theme_loads_default(self):
        """Tests that 'default' can be loaded correctly."""
        theme = theme_manager.get_theme("default")
        assert theme["description"] == "Packaged Default"

    def test_get_theme_user_overrides_packaged(self, fake_fs):
        """
        Tests that a user theme in their config directory takes precedence
        over a packaged theme with the same name.
        """
        # Arrange: User "solarized" theme with a different description
        user_solarized_content = json.dumps({"description": "User Solarized"})
        fake_fs.create_file(
            USER_THEMES_DIR / "solarized.json", contents=user_solarized_content
        )

        # Act
        theme = theme_manager.get_theme("solarized")

        # Assert
        assert theme["description"] == "User Solarized"

    def test_get_theme_fallback_on_not_found(self, caplog):
        """Tests that it falls back to the default theme if the requested one is not found."""
        theme = theme_manager.get_theme("nonexistent")
        assert theme["description"] == "Packaged Default"
        assert "Theme 'nonexistent' not found" in caplog.text

    def test_list_themes_combines_and_deduplicates(self, fake_fs):
        """
        Tests that list_themes combines packaged and user themes,
        with user themes taking precedence.
        """
        # Arrange: User themes (one new, one override)
        fake_fs.create_file(
            USER_THEMES_DIR / "solarized.json",
            contents=json.dumps({"description": "User Solarized Override"}),
        )
        fake_fs.create_file(
            USER_THEMES_DIR / "custom.json",
            contents=json.dumps({"description": "User Custom"}),
        )

        # Act
        all_themes = theme_manager.list_themes()

        # Assert
        assert len(all_themes) == 3
        assert all_themes["default"] == "Packaged Default"
        assert all_themes["solarized"] == "User Solarized Override"
        assert all_themes["custom"] == "User Custom"

    def test_get_theme_handles_malformed_json(self, fake_fs, caplog):
        """Tests that a malformed user theme file causes a fallback to default."""
        # Arrange
        fake_fs.create_file(USER_THEMES_DIR / "bad.json", contents="{not json")

        # Act
        theme = theme_manager.get_theme("bad")

        # Assert
        assert theme["description"] == "Packaged Default"
        assert "Could not load user theme 'bad'" in caplog.text
