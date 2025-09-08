# tests/test_personas.py
"""
Tests for the persona management module in aiterm/personas.py.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from aiterm import config, personas


class TestPersonas:
    """Test suite for persona management."""

    def test_create_default_persona_if_missing_exists(self, fake_fs):
        """Tests that the function does nothing if the default persona already exists."""
        default_path = config.PERSONAS_DIRECTORY / personas.DEFAULT_PERSONA_FILENAME
        fake_fs.create_file(default_path, contents='{"name": "Existing"}')
        personas.create_default_persona_if_missing()
        with open(default_path) as f:
            content = f.read()
        assert "AITERM Assistant" not in content  # Should not have been overwritten

    def test_create_default_persona_if_missing_creates(self, fake_fs):
        """Tests that the default persona is created if it doesn't exist."""
        default_path = config.PERSONAS_DIRECTORY / personas.DEFAULT_PERSONA_FILENAME
        assert not default_path.exists()
        personas.create_default_persona_if_missing()
        assert default_path.exists()
        with open(default_path) as f:
            data = json.load(f)
        assert data["name"] == "AITERM Assistant"

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Permission tests are POSIX-specific"
    )
    def test_create_default_persona_os_error(self, fake_fs, caplog):
        """Tests that an OSError during creation is handled gracefully."""
        # Make the directory read-only to cause an OSError
        config.PERSONAS_DIRECTORY.chmod(0o555)
        personas.create_default_persona_if_missing()
        assert "Failed to create default persona file" in caplog.text

    def test_resolve_attachment_paths(self, fake_fs, monkeypatch):
        """Tests various path resolution strategies for persona attachments."""
        persona_path = config.PERSONAS_DIRECTORY / "test.json"

        # Create a platform-specific home directory path for the mock
        home_dir_str = "C:\\home\\user" if sys.platform == "win32" else "/home/user"
        home_dir_path = Path(home_dir_str)

        # Monkeypatch os.path.expanduser to be platform-aware
        monkeypatch.setattr(
            os.path, "expanduser", lambda p: p.replace("~", home_dir_str)
        )

        # Create a file relative to the persona dir
        relative_file = config.PERSONAS_DIRECTORY / "relative.txt"
        fake_fs.create_file(relative_file)
        # Create an absolute file using an OS-agnostic method
        abs_path_str = os.path.abspath("/etc/abs.txt")
        abs_file = Path(abs_path_str)
        fake_fs.create_file(abs_file)
        # Create a file in the user's home dir
        home_file = home_dir_path / "docs/home.txt"
        fake_fs.create_file(home_file)

        raw_paths = ["relative.txt", abs_path_str, "~/docs/home.txt"]
        resolved_paths = personas._resolve_attachment_paths(persona_path, raw_paths)

        assert resolved_paths[0].resolve() == relative_file.resolve()
        assert resolved_paths[1].resolve() == abs_file.resolve()
        assert resolved_paths[2].resolve() == home_file.resolve()

    def test_load_persona_nonexistent(self):
        """Tests that loading a persona that doesn't exist returns None."""
        assert personas.load_persona("nonexistent") is None

    def test_load_persona_json_error(self, fake_fs, caplog):
        """Tests loading a persona with invalid JSON."""
        persona_path = config.PERSONAS_DIRECTORY / "bad.json"
        persona_path.write_text("not json")
        result = personas.load_persona("bad")
        assert result is None
        assert "Failed to load or parse persona file" in caplog.text

    def test_load_persona_missing_key(self, fake_fs, caplog):
        """Tests loading a persona missing a required key."""
        persona_path = config.PERSONAS_DIRECTORY / "incomplete.json"
        persona_path.write_text('{"system_prompt": "test"}')  # Missing "name"
        result = personas.load_persona("incomplete")
        assert result is None
        assert "is missing 'name' or 'system_prompt'" in caplog.text

    def test_load_persona_bad_attachments(self, fake_fs, caplog):
        """Tests loading a persona where 'attachments' is not a list."""
        persona_path = config.PERSONAS_DIRECTORY / "bad_attachments.json"
        content = {
            "name": "test",
            "system_prompt": "test",
            "attachments": "not_a_list",
        }
        persona_path.write_text(json.dumps(content))
        persona = personas.load_persona("bad_attachments")
        assert persona is not None
        assert persona.attachments == []  # Should default to empty list
        assert "Attachments in bad_attachments.json must be a list" in caplog.text

    def test_list_personas_no_dir(self, fake_fs):
        """Tests listing personas when the directory doesn't exist."""
        # The fake_fs fixture creates the directory by default, so we remove it for this test.
        shutil.rmtree(config.PERSONAS_DIRECTORY)
        result = personas.list_personas()
        assert result == []

    def test_list_personas_skips_invalid(self, fake_fs, caplog):
        """Tests that list_personas loads valid files and skips invalid ones."""
        # Valid persona
        valid_path = config.PERSONAS_DIRECTORY / "valid.json"
        valid_path.write_text('{"name": "Valid", "system_prompt": "p"}')
        # Invalid persona
        invalid_path = config.PERSONAS_DIRECTORY / "invalid.json"
        invalid_path.write_text('{"name": "Invalid"}')  # Missing system_prompt

        result = personas.list_personas()
        assert len(result) == 1
        assert result[0].name == "Valid"
        assert "is missing 'name' or 'system_prompt'" in caplog.text
