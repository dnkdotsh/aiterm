# tests/utils/test_ui_helpers.py
from unittest.mock import MagicMock

import pytest
from aiterm.engine import AIEngine
from aiterm.utils import ui_helpers


@pytest.fixture
def mock_engine(mocker):
    """Provides a mock AIEngine with a configurable list of models."""
    engine = MagicMock(spec=AIEngine)
    # This name must match a default defined in the mock_settings_patcher fixture
    engine.name = "gemini"
    engine.fetch_available_models.return_value = ["model-1", "model-2", "model-3"]
    return engine


@pytest.mark.usefixtures("mock_settings_patcher")
class TestUIHelpers:
    """Test suite for UI helper functions."""

    def test_select_model_use_default_yes(self, mock_engine, mock_prompt_toolkit):
        """Tests selecting the default model by typing 'y'."""
        mock_prompt_toolkit["input_queue"].append("y")
        result = ui_helpers.select_model(mock_engine, "image")
        assert result == "dall-e-default"
        mock_engine.fetch_available_models.assert_not_called()

    def test_select_model_use_default_enter(self, mock_engine, mock_prompt_toolkit):
        """Tests selecting the default model by pressing Enter."""
        mock_prompt_toolkit["input_queue"].append("")
        result = ui_helpers.select_model(mock_engine, "chat")
        assert result == "gemini-default"
        mock_engine.fetch_available_models.assert_not_called()

    def test_select_model_choose_from_list(self, mock_engine, mock_prompt_toolkit):
        """Tests choosing a model from the fetched list."""
        mock_prompt_toolkit["input_queue"].extend(["n", "2"])
        result = ui_helpers.select_model(mock_engine, "chat")
        assert result == "model-2"
        mock_engine.fetch_available_models.assert_called_once_with("chat")

    def test_select_model_invalid_choice(self, mock_engine, mock_prompt_toolkit):
        """Tests that invalid input falls back to the default model."""
        mock_prompt_toolkit["input_queue"].extend(["n", "99"])
        result = ui_helpers.select_model(mock_engine, "chat")
        assert result == "gemini-default"

    def test_select_model_fetch_fails(self, mock_engine, mock_prompt_toolkit):
        """Tests fallback to default when model fetching fails."""
        mock_engine.fetch_available_models.return_value = []
        mock_prompt_toolkit["input_queue"].append("n")
        result = ui_helpers.select_model(mock_engine, "chat")
        assert result == "gemini-default"
        mock_engine.fetch_available_models.assert_called_once_with("chat")

    def test_display_help_chat(self, capsys):
        """Tests that chat help displays relevant commands."""
        ui_helpers.display_help("chat")
        captured = capsys.readouterr().out
        assert "Interactive Chat Commands" in captured
        assert "/exit" in captured
        assert "/persona" in captured
        assert "/forget [N | <topic> --memory]" in captured
        assert "Warning: The --memory flag" in captured
        assert "/ai <gpt|gem>" not in captured

    def test_display_help_multichat(self, capsys):
        """Tests that multichat help displays relevant commands."""
        ui_helpers.display_help("multichat")
        captured = capsys.readouterr().out
        assert "Multi-Chat Commands" in captured
        assert "/ai <gpt|gem>" in captured
        assert "/persona" not in captured
