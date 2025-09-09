# tests/utils/test_ui_helpers.py

from unittest.mock import MagicMock, patch

from src.aiterm.engine import AIEngine
from src.aiterm.utils import ui_helpers


class TestUIHelpers:
    @patch("src.aiterm.utils.ui_helpers.prompt")
    def test_select_model_default(self, mock_prompt):
        """Tests that selecting the default model works."""
        mock_engine = MagicMock(spec=AIEngine)
        mock_prompt.return_value = "y"

        with patch.dict(
            ui_helpers.settings, {"default_openai_image_model": "dall-e-default"}
        ):
            result = ui_helpers.select_model(mock_engine, "image")
            assert result == "dall-e-default"

    @patch("src.aiterm.utils.ui_helpers.prompt")
    def test_select_model_choose_from_list(self, mock_prompt):
        """Tests selecting a model from the fetched list."""
        mock_engine = MagicMock(spec=AIEngine)
        mock_engine.fetch_available_models.return_value = ["model-a", "model-b"]
        # First prompt for default (n), second for choice (2)
        mock_prompt.side_effect = ["n", "2"]

        result = ui_helpers.select_model(mock_engine, "chat")
        assert result == "model-b"

    @patch("src.aiterm.utils.ui_helpers.prompt")
    def test_select_model_invalid_choice(self, mock_prompt):
        """Tests that an invalid choice falls back to the default."""
        mock_engine = MagicMock(spec=AIEngine)
        mock_engine.fetch_available_models.return_value = ["model-a", "model-b"]
        # First prompt for default (n), second for choice (invalid)
        mock_prompt.side_effect = ["n", "99"]
        with (
            patch.dict(ui_helpers.settings, {"default_gemini_model": "gemini-default"}),
            patch(
                "src.aiterm.utils.ui_helpers.get_default_model_for_engine"
            ) as mock_get_default,
        ):
            mock_get_default.return_value = "gemini-default"
            result = ui_helpers.select_model(mock_engine, "chat")
            assert result == "gemini-default"

    def test_display_help_chat(self, capsys):
        """Tests that chat help displays relevant commands."""
        ui_helpers.display_help("chat")
        captured = capsys.readouterr().out
        assert "Interactive Chat Commands" in captured
        assert "/exit" in captured
        assert "/persona" in captured
        assert "/forget" in captured
        assert "/ai" not in captured

    def test_display_help_multichat(self, capsys):
        """Tests that multichat help displays relevant commands."""
        ui_helpers.display_help("multichat")
        captured = capsys.readouterr().out
        assert "Multi-Chat Commands" in captured
        assert "/ai <gpt|gem>" in captured
        assert "/persona <gpt|gem> <name>" in captured
        assert "/personas" in captured

    def test_display_help_invalid(self, capsys):
        """Tests that an invalid context shows a generic message."""
        ui_helpers.display_help("invalid_context")
        captured = capsys.readouterr().out
        assert "No help available" in captured
