# tests/utils/test_message_builder.py
"""
Tests for the utility functions in aiterm/utils/message_builder.py.
"""

import pytest
from aiterm.utils.message_builder import (
    construct_assistant_message,
    construct_user_message,
    extract_text_from_message,
    translate_history,
)


class TestMessageBuilder:
    """Test suite for message construction and parsing utilities."""

    @pytest.mark.parametrize("engine_name", ["openai", "groq"])
    def test_construct_user_message_openai_compatible_text_only(self, engine_name):
        """Tests building an OpenAI-compatible user message with only text."""
        msg = construct_user_message(engine_name, "Hello", [])
        assert msg == {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }

    @pytest.mark.parametrize("engine_name", ["openai", "groq"])
    def test_construct_user_message_openai_compatible_with_image(self, engine_name):
        """Tests building an OpenAI-compatible user message with text and an image."""
        image_data = [{"mime_type": "image/png", "data": "base64data"}]
        msg = construct_user_message(engine_name, "Look at this", image_data)
        assert msg == {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,base64data"},
                },
            ],
        }

    def test_construct_user_message_anthropic_with_image(self):
        """Tests building an Anthropic user message with text and an image."""
        image_data = [{"mime_type": "image/jpeg", "data": "base64data"}]
        msg = construct_user_message("anthropic", "Look at this", image_data)
        assert msg == {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "base64data"
                    }
                },
            ],
        }

    def test_construct_user_message_gemini_text_only(self):
        """Tests building a Gemini user message with only text."""
        msg = construct_user_message("gemini", "Hello", [])
        assert msg == {"role": "user", "parts": [{"text": "Hello"}]}

    def test_construct_user_message_gemini_with_image(self):
        """Tests building a Gemini user message with text and an image."""
        image_data = [{"mime_type": "image/jpeg", "data": "base64data"}]
        msg = construct_user_message("gemini", "Look at this", image_data)
        assert msg == {
            "role": "user",
            "parts": [
                {"text": "Look at this"},
                {"inline_data": {"mime_type": "image/jpeg", "data": "base64data"}},
            ],
        }

    @pytest.mark.parametrize("engine_name", ["openai", "groq"])
    def test_construct_assistant_message_openai_compatible(self, engine_name):
        """Tests building an OpenAI-compatible assistant message."""
        msg = construct_assistant_message(engine_name, "I am a bot.")
        assert msg == {"role": "assistant", "content": "I am a bot."}

    def test_construct_assistant_message_anthropic(self):
        """Tests building an Anthropic assistant message."""
        msg = construct_assistant_message("anthropic", "I am Claude.")
        assert msg == {"role": "assistant", "content": "I am Claude."}

    def test_construct_assistant_message_gemini(self):
        """Tests building a Gemini assistant (model) message."""
        msg = construct_assistant_message("gemini", "I am a bot.")
        assert msg == {"role": "model", "parts": [{"text": "I am a bot."}]}

    @pytest.mark.parametrize(
        "message, expected_text",
        [
            # OpenAI / Anthropic formats
            (
                {"role": "assistant", "content": "Simple string content."},
                "Simple string content.",
            ),
            (
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Text from list."}],
                },
                "Text from list.",
            ),
            ({"role": "user", "content": [{"type": "image_url", "image_url": {}}]}, ""),
            # Gemini formats
            ({"role": "model", "parts": [{"text": "Gemini text."}]}, "Gemini text."),
            ({"role": "model", "parts": [{"inline_data": {}}]}, ""),
            # Fallback and edge cases
            ({"role": "user", "text": "Direct text field."}, "Direct text field."),
            ({"role": "user", "content": []}, ""),
            ({}, ""),
        ],
    )
    def test_extract_text_from_message(self, message, expected_text):
        """Tests extracting text from various message structures."""
        assert extract_text_from_message(message) == expected_text

    def test_translate_history_openai_to_gemini(self):
        """Tests translating a history from OpenAI to Gemini format."""
        openai_history = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": "Hello there"},
            {"role": "system", "content": "You are an assistant"},  # Should be skipped
        ]
        gemini_history = translate_history(openai_history, "gemini")
        assert len(gemini_history) == 2
        assert gemini_history[0] == {"role": "user", "parts": [{"text": "Hi"}]}
        assert gemini_history[1] == {
            "role": "model",
            "parts": [{"text": "Hello there"}],
        }

    def test_translate_history_gemini_to_openai(self):
        """Tests translating a history from Gemini to OpenAI format."""
        gemini_history = [
            {"role": "user", "parts": [{"text": "Hi"}]},
            {"role": "model", "parts": [{"text": "Hello there"}]},
        ]
        openai_history = translate_history(gemini_history, "openai")
        assert len(openai_history) == 2
        assert openai_history[0] == {
            "role": "user",
            "content": [{"type": "text", "text": "Hi"}],
        }
        assert openai_history[1] == {"role": "assistant", "content": "Hello there"}