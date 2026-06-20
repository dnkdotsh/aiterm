# tests/test_engine.py
"""
Unit tests for the AIEngine classes in aiterm/engine.py.
These tests validate the logic for building API-specific payloads and parsing responses.
"""

import logging
from unittest.mock import MagicMock

import pytest
import requests
from aiterm.engine import AnthropicEngine, GeminiEngine, OpenAICompatibleEngine, get_engine


class TestEngines:
    """Test suite for AI engine implementations."""

    def test_get_engine_factory(
        self, mock_gemini_engine, mock_anthropic_engine
    ):
        """Tests the factory function for creating engine instances."""
        engine_openai = get_engine("openai", "fake_key")
        assert isinstance(engine_openai, OpenAICompatibleEngine)
        assert engine_openai.provider_id == "openai"

        engine_groq = get_engine("groq", "fake_key")
        assert isinstance(engine_groq, OpenAICompatibleEngine)
        assert engine_groq.provider_id == "groq"

        assert isinstance(get_engine("gemini", "fake_key"), GeminiEngine)
        assert isinstance(get_engine("anthropic", "fake_key"), AnthropicEngine)
        with pytest.raises(ValueError):
            get_engine("unknown_engine", "fake_key")

    @pytest.mark.parametrize("provider_id", ["openai", "groq"])
    def test_openai_compatible_build_chat_payload(self, provider_id):
        """Tests that chat payloads are constructed correctly for older/standard models."""
        engine = OpenAICompatibleEngine(provider_id, "fake_key")
        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "Be brief."
        payload = engine.build_chat_payload(
            messages, system_prompt, 100, True, "gpt-3.5-turbo"
        )

        assert payload["model"] == "gpt-3.5-turbo"
        assert payload["stream"] is True
        assert payload["max_tokens"] == 100
        assert "max_completion_tokens" not in payload
        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {"role": "system", "content": "Be brief."}

    @pytest.mark.parametrize("provider_id", ["openai", "groq"])
    def test_openai_compatible_build_chat_payload_max_completion_tokens(self, provider_id):
        """Tests that max_completion_tokens is used for newer reasoning models."""
        engine = OpenAICompatibleEngine(provider_id, "fake_key")
        messages = [{"role": "user", "content": "Hello"}]

        # Reasoning models use max_completion_tokens
        payload_o1 = engine.build_chat_payload(
            messages, None, 150, False, "o1-mini"
        )
        assert payload_o1["model"] == "o1-mini"
        assert payload_o1["stream"] is False
        assert "max_tokens" not in payload_o1
        assert payload_o1["max_completion_tokens"] == 150
        assert len(payload_o1["messages"]) == 1

        # Standard models use max_tokens
        payload_std = engine.build_chat_payload(
            messages, None, 150, False, "gpt-4o-mini"
        )
        assert payload_std["max_tokens"] == 150
        assert "max_completion_tokens" not in payload_std

    @pytest.mark.parametrize("provider_id", ["openai", "groq"])
    def test_openai_compatible_parse_chat_response(
        self, provider_id, mock_openai_chat_response
    ):
        """Tests parsing of a standard chat response."""
        engine = OpenAICompatibleEngine(provider_id, "fake_key")
        response_text = engine.parse_chat_response(mock_openai_chat_response)
        assert response_text == "This is a test response."

    @pytest.mark.parametrize("provider_id", ["openai", "groq"])
    def test_openai_compatible_parse_chat_response_empty(self, provider_id):
        """Tests parsing of an empty chat response."""
        engine = OpenAICompatibleEngine(provider_id, "fake_key")
        assert engine.parse_chat_response({}) == ""
        assert engine.parse_chat_response({"choices": []}) == ""
        assert engine.parse_chat_response({"choices": [{"message": {}}]}) == ""

    @pytest.mark.parametrize("provider_id, expected_log", [
        ("openai", "Could not fetch OpenAI model list (HTTP Error)."),
        ("groq", "Could not fetch Groq model list (HTTP Error).")
    ])
    def test_openai_compatible_fetch_available_models_http_error(
        self, provider_id, expected_log, mocker, caplog
    ):
        """Tests HTTP error handling when fetching models."""
        engine = OpenAICompatibleEngine(provider_id, "fake_key")
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.RequestException("HTTP Error")
        )
        mocker.patch("requests.get", return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            models = engine.fetch_available_models("chat")
            assert models == []
            assert ("aiterm", logging.WARNING, expected_log) in caplog.record_tuples

    @pytest.mark.parametrize("provider_id", ["openai", "groq"])
    def test_openai_compatible_fetch_available_models_success(self, provider_id, mocker):
        """Tests successful fetching and filtering of models."""
        engine = OpenAICompatibleEngine(provider_id, "fake_key")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o-mini"},
                {"id": "dall-e-3"},
                {"id": "whisper-1"},
                {"id": "gpt-3.5-turbo"},
            ]
        }
        mocker.patch("requests.get", return_value=mock_response)

        chat_models = engine.fetch_available_models("chat")
        assert sorted(chat_models) == ["gpt-3.5-turbo", "gpt-4o-mini"]

        image_models = engine.fetch_available_models("image")
        assert sorted(image_models) == ["dall-e-3"]

    @pytest.mark.parametrize("provider_id, expected_url", [
        ("openai", "https://api.openai.com/v1/chat/completions"),
        ("groq", "https://api.groq.com/openai/v1/chat/completions")
    ])
    def test_openai_compatible_get_chat_url(self, provider_id, expected_url):
        """Tests endpoint URL resolution."""
        engine = OpenAICompatibleEngine(provider_id, "fake_key")
        assert engine.get_chat_url("some-model", stream=False) == expected_url
        assert engine.get_chat_url("some-model", stream=True) == expected_url

    @pytest.mark.parametrize("provider_id", ["openai", "groq"])
    def test_openai_compatible_get_headers(self, provider_id):
        """Tests header generation for OpenAI-compatible engines."""
        engine = OpenAICompatibleEngine(provider_id, "fake_key")
        headers = engine.get_headers()
        assert headers["Authorization"] == "Bearer fake_key"
        assert headers["Content-Type"] == "application/json"

    def test_gemini_build_chat_payload(self, mock_gemini_engine):
        """Tests that Gemini chat payloads are constructed correctly."""
        messages = [{"role": "user", "parts": [{"text": "Hello"}]}]
        system_prompt = "Be brief."
        payload = mock_gemini_engine.build_chat_payload(
            messages, system_prompt, 200, False, "gemini-1.5-flash"
        )

        assert payload["system_instruction"]["parts"][0]["text"] == "Be brief."
        assert payload["generationConfig"]["maxOutputTokens"] == 200
        assert payload["contents"] == messages

    def test_anthropic_build_chat_payload(self, mock_anthropic_engine):
        """Tests that Anthropic chat payloads are constructed correctly."""
        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "System instructions."
        payload = mock_anthropic_engine.build_chat_payload(
            messages, system_prompt, 1024, True, "claude-3-sonnet"
        )

        assert payload["model"] == "claude-3-sonnet"
        assert payload["stream"] is True
        assert payload["max_tokens"] == 1024
        assert payload["system"] == "System instructions."
        assert payload["messages"] == messages

    def test_gemini_parse_chat_response_empty_or_malformed(
        self, mock_gemini_engine, caplog
    ):
        """Tests parsing of empty or malformed Gemini chat responses."""
        test_cases = [
            ({}, "Could not extract Gemini response part or finish reason"),
            (
                {"candidates": []},
                "Could not parse Gemini response due to unexpected structure",
            ),
            (
                {"candidates": [{"content": {}}]},
                "Could not extract Gemini response part or finish reason",
            ),
            (
                {"candidates": [{"content": {"parts": []}}]},
                "Could not extract Gemini response part or finish reason",
            ),
            (
                {"candidates": [{"content": {"parts": [{"not_text": "data"}]}}]},
                "Could not extract Gemini response part or finish reason",
            ),
        ]

        for response_data, expected_log_msg in test_cases:
            with caplog.at_level(logging.WARNING):
                caplog.clear()  # Clear logs for each case
                assert mock_gemini_engine.parse_chat_response(response_data) == ""
                assert expected_log_msg in caplog.text

    def test_anthropic_parse_chat_response(
        self, mock_anthropic_engine, mock_anthropic_chat_response
    ):
        """Tests parsing of a standard Anthropic chat response."""
        response_text = mock_anthropic_engine.parse_chat_response(
            mock_anthropic_chat_response
        )
        assert response_text == "This is a test response."

    def test_anthropic_parse_chat_response_empty(self, mock_anthropic_engine):
        """Tests parsing of an empty Anthropic chat response."""
        assert mock_anthropic_engine.parse_chat_response({}) == ""
        assert mock_anthropic_engine.parse_chat_response({"content": []}) == ""

    def test_gemini_fetch_available_models_http_error(
        self, mock_gemini_engine, mocker, caplog
    ):
        """Tests HTTP error handling when fetching Gemini models."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.RequestException("HTTP Error")
        )
        mocker.patch("requests.get", return_value=mock_response)
        with caplog.at_level(logging.WARNING):
            models = mock_gemini_engine.fetch_available_models("chat")
            assert models == []
            assert (
                "aiterm",
                logging.WARNING,
                "Could not fetch Gemini model list (HTTP Error).",
            ) in caplog.record_tuples

    def test_anthropic_fetch_available_models_http_error(
        self, mock_anthropic_engine, mocker, caplog
    ):
        """Tests HTTP error handling when fetching Anthropic models."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.RequestException("HTTP Error")
        )
        mocker.patch("requests.get", return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            models = mock_anthropic_engine.fetch_available_models("chat")
            assert models == []
            assert (
                "aiterm",
                logging.WARNING,
                "Could not fetch Anthropic model list (HTTP Error).",
            ) in caplog.record_tuples

    def test_gemini_fetch_available_models_success(self, mock_gemini_engine, mocker):
        """Tests successful fetching and filtering of Gemini models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "models/gemini-1.5-flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/embedding-001",
                    "supportedGenerationMethods": ["embedContent"],
                },
                {
                    "name": "models/gemini-pro",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        }
        mocker.patch("requests.get", return_value=mock_response)

        chat_models = mock_gemini_engine.fetch_available_models("chat")
        assert sorted(chat_models) == [
            "gemini-1.5-flash",
            "gemini-pro",
        ]  # "models/" prefix should be removed

    def test_anthropic_fetch_available_models_success(self, mock_anthropic_engine, mocker):
        """Tests successful fetching and filtering of Anthropic models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "claude-3-5-sonnet-20241022"},
                {"id": "claude-3-opus-20240229"}
            ]
        }
        mocker.patch("requests.get", return_value=mock_response)

        chat_models = mock_anthropic_engine.fetch_available_models("chat")
        assert sorted(chat_models) == [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229"
        ]

        # Anthropic doesn't support image generation models
        image_models = mock_anthropic_engine.fetch_available_models("image")
        assert image_models == []