# tests/test_api_client.py

import json
from unittest.mock import MagicMock

import pytest
import requests
from aiterm import api_client
from aiterm.api_client import (
    ApiRequestError,
    MissingApiKeyError,
    check_api_keys,
    make_api_request,
)
from aiterm.engine import AIEngine


@pytest.fixture
def mock_settings(mocker):
    """Fixture to mock the settings object."""
    return mocker.patch("aiterm.api_client.settings", {"api_timeout": 10})


def test_check_api_keys_openai_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    assert check_api_keys("openai") == "test_key"


def test_check_api_keys_gemini_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    assert check_api_keys("gemini") == "test_key"


def test_check_api_keys_missing_openai(monkeypatch, mocker):
    # Mock load_dotenv to prevent it from reading a real .env file
    mocker.patch("aiterm.api_client.load_dotenv")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError) as excinfo:
        check_api_keys("openai")
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_check_api_keys_missing_gemini(monkeypatch, mocker):
    # Mock load_dotenv to prevent it from reading a real .env file
    mocker.patch("aiterm.api_client.load_dotenv")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError) as excinfo:
        check_api_keys("gemini")
    assert "GEMINI_API_KEY" in str(excinfo.value)


def test_make_api_request_success(mocker, mock_settings):
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.return_value = {"success": True}
    mock_post.return_value = mock_response

    response = make_api_request("http://test.com", {}, {})
    assert response == {"success": True}
    mock_post.assert_called_once()
    mock_response.raise_for_status.assert_called_once()


def test_make_api_request_streaming_success(mocker, mock_settings):
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200  # Add status_code to the mock
    mock_post.return_value = mock_response

    response = make_api_request("http://test.com", {}, {}, stream=True)
    assert response is mock_response
    mock_post.assert_called_once_with(
        "http://test.com", headers={}, json={}, stream=True, timeout=10
    )
    mock_response.raise_for_status.assert_called_once()


def test_make_api_request_api_error_in_payload(mocker, mock_settings):
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": {"message": "API Error Occurred"}}
    mock_post.return_value = mock_response

    with pytest.raises(ApiRequestError, match="API Error Occurred"):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_http_error_with_json(mocker, mock_settings):
    """Test that an HTTP error with a valid JSON body is handled correctly."""
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

    # Create an HTTPError instance and attach the mock response to it
    http_error = requests.exceptions.HTTPError()
    http_error.response = mock_response
    mock_response.raise_for_status.side_effect = http_error
    mock_post.return_value = mock_response

    with pytest.raises(ApiRequestError, match="Invalid API key"):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_http_error_with_non_json(mocker, mock_settings):
    """Test that an HTTP error with a non-JSON body is handled."""
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
    mock_response.text = "Internal Server Error"

    # Create an HTTPError instance and attach the mock response to it
    http_error = requests.exceptions.HTTPError()
    http_error.response = mock_response
    mock_response.raise_for_status.side_effect = http_error
    mock_post.return_value = mock_response

    with pytest.raises(ApiRequestError, match="Internal Server Error"):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_connection_error(mocker, mock_settings):
    """Test that a generic RequestException is caught and wrapped."""
    mocker.patch(
        "requests.post",
        side_effect=requests.exceptions.RequestException("Connection failed"),
    )
    with pytest.raises(ApiRequestError, match="Connection failed"):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_success_with_bad_json(mocker, mock_settings):
    """Test handling of a 200 OK response with an invalid JSON body."""
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
    mock_post.return_value = mock_response

    with pytest.raises(ApiRequestError, match="Failed to decode API response."):
        make_api_request("http://test.com", {}, {})


def test_make_api_request_logs_on_failure(mocker, mock_settings, fs):
    """Test that the request is logged even when an exception occurs."""
    mocker.patch(
        "requests.post",
        side_effect=requests.exceptions.RequestException("Connection failed"),
    )
    # Mock the log file path
    log_file_path = "/fake/home/.local/share/aiterm/logs/raw.log"
    fs.create_file(log_file_path, contents="")

    mocker.patch("aiterm.api_client.config.RAW_LOG_FILE", log_file_path)
    # We also need to mock the logger setup to use the fake path
    mocker.patch("aiterm.api_client._setup_raw_logger")

    session_logs = []
    with pytest.raises(ApiRequestError):
        make_api_request(
            "http://test.com",
            {},
            {"payload": "data"},
            debug_active=True,
            session_raw_logs=session_logs,
        )

    # Check that the in-memory session log was appended to
    assert len(session_logs) == 1
    assert session_logs[0]["request"]["payload"] == {"payload": "data"}
    assert "Connection failed" in session_logs[0]["response"]["error"]


def test_parse_token_counts_openai(mock_openai_chat_response):
    """Tests parsing token counts from an OpenAI response."""
    p, c, r, t = api_client._parse_token_counts("openai", mock_openai_chat_response)
    assert p == 10
    assert c == 20
    assert r == 0
    assert t == 30


def test_parse_token_counts_gemini(mock_gemini_chat_response):
    """Tests parsing token counts from a Gemini response."""
    p, c, r, t = api_client._parse_token_counts("gemini", mock_gemini_chat_response)
    assert p == 15
    assert c == 25
    assert r == 5
    assert t == 40


def test_process_stream_openai(mock_streaming_response_factory):
    """Tests successful processing of an OpenAI stream."""
    openai_stream_chunks = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " "}}]}',
        'data: {"choices": [{"delta": {"content": "world"}}]}',
        'data: {"usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}',
        "data: [DONE]",
    ]
    mock_response = mock_streaming_response_factory(openai_stream_chunks)
    full_text, tokens = api_client._process_stream(
        "openai", mock_response, print_stream=False
    )
    assert full_text == "Hello world"
    assert tokens == {"prompt": 5, "completion": 10, "reasoning": 0, "total": 15}


def test_process_stream_gemini(mock_streaming_response_factory):
    """Tests successful processing of a Gemini stream."""
    gemini_stream_chunks = [
        'data: {"candidates": [{"content": {"parts": [{"text": "This "}]}}]}',
        'data: {"candidates": [{"content": {"parts": [{"text": "is "}]}}]}',
        'data: {"candidates": [{"content": {"parts": [{"text": "a test."}]}}]}',
        'data: {"usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 12, "cachedContentTokenCount": 2, "totalTokenCount": 22}}',
    ]
    mock_response = mock_streaming_response_factory(gemini_stream_chunks)
    full_text, tokens = api_client._process_stream(
        "gemini", mock_response, print_stream=False
    )
    assert full_text == "This is a test."
    assert tokens == {"prompt": 8, "completion": 12, "reasoning": 2, "total": 22}


def test_process_stream_keyboard_interrupt(mock_streaming_response_factory):
    """Tests that a KeyboardInterrupt during streaming is handled gracefully."""

    def iter_lines_with_interrupt(*args, **kwargs):
        yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}'
        raise KeyboardInterrupt

    mock_response = MagicMock(spec=requests.Response)
    mock_response.iter_lines.side_effect = iter_lines_with_interrupt

    full_text, tokens = api_client._process_stream(
        "openai", mock_response, print_stream=False
    )

    assert full_text == "Hello"
    assert tokens == {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0}


def test_perform_chat_request_streaming_error(mocker):
    """Tests that a top-level streaming request handles upstream errors."""
    mocker.patch(
        "aiterm.api_client.make_api_request",
        side_effect=ApiRequestError("Stream connection failed"),
    )
    mock_engine = MagicMock(spec=AIEngine)
    mock_engine.name = "openai"
    mock_engine.api_key = "fake_key"
    mock_engine.get_chat_url.return_value = "http://fake.url"
    mock_engine.build_chat_payload.return_value = {}

    response_text, tokens = api_client.perform_chat_request(
        engine=mock_engine,
        model="test-model",
        messages_or_contents=[],
        system_prompt=None,
        max_tokens=100,
        stream=True,
    )

    assert response_text == ""
    assert tokens == {}
