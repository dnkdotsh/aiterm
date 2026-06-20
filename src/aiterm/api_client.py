# aiterm/api_client.py
# aiterm: A command-line interface for interacting with AI models.
# Copyright (C) 2025-2026 Dank A. Saurus

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import datetime
import json
import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler

import requests
from dotenv import load_dotenv

from . import config
from .engine import AIEngine, PROVIDER_CONFIGS, _is_groq_reasoning_model
from .logger import log
from .settings import settings
from .utils.formatters import RESET_COLOR, SYSTEM_MSG
from .utils.redaction import redact_sensitive_info

# Dim grey, used to render reasoning/chain-of-thought during the /debug reveal.
REASONING_COLOR = "\033[90m"


class MissingApiKeyError(Exception):
    """Custom exception for missing API keys."""

    pass


class ApiRequestError(Exception):
    """Custom exception for failed API requests."""

    pass


def _setup_raw_logger() -> logging.Logger:
    """Sets up a rotating file logger specifically for raw API logs."""
    raw_logger = logging.getLogger("aiterm_raw")
    raw_logger.setLevel(logging.INFO)
    raw_logger.propagate = False
    if not raw_logger.handlers:
        try:
            # Rotates when the log reaches 2MB, keeping up to 5 backup logs.
            handler = RotatingFileHandler(
                config.RAW_LOG_FILE,
                maxBytes=2 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            raw_logger.addHandler(handler)
        except OSError as e:
            log.warning("Could not create raw log file handler: %s", e)
    return raw_logger


raw_api_logger = _setup_raw_logger()


def check_api_keys(engine: str):
    """
    Checks for the required API key in environment variables, loading the .env
    file if necessary, and returns the key.
    """
    load_dotenv(dotenv_path=config.DOTENV_FILE)

    if engine in PROVIDER_CONFIGS:
        key_name = PROVIDER_CONFIGS[engine]["api_key_env"]
    else:
        key_map = {
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        key_name = key_map.get(engine)

    if not key_name:
        raise ValueError(f"Unknown engine provided to key checker: {engine}")

    api_key = os.getenv(key_name)
    if not api_key:
        raise MissingApiKeyError(
            f"Error: Environment variable '{key_name}' is not set."
        )
    return api_key


def make_api_request(
    url: str,
    headers: dict,
    payload: dict,
    stream: bool = False,
    debug_active: bool = False,
    session_raw_logs: list | None = None,
) -> dict | requests.Response:
    """
    Makes a POST request to the specified API endpoint and handles errors.
    Returns a dictionary for non-streaming responses or a requests.Response object for streaming.
    Raises ApiRequestError on failure.
    """
    response_data = None
    error_details_for_log = None
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "request": {"url": url, "headers": headers, "payload": payload},
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=settings["api_timeout"],
        )
        response.raise_for_status()
        if stream:
            log_entry["response"] = {
                "status_code": response.status_code,
                "streaming": True,
            }
            return response
        response_data = response.json()
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown API error")
            log.error("API Error: %s", error_msg)
            raise ApiRequestError(error_msg)
        return response_data
    except requests.exceptions.HTTPError as e:
        error_details = "No specific error message provided by the API."
        try:
            error_json = e.response.json()
            if (
                isinstance(error_json, dict)
                and "error" in error_json
                and "message" in error_json.get("error", {})
            ):
                error_details = error_json["error"]["message"]
            else:
                error_details = e.response.text
        except json.JSONDecodeError:
            error_details = e.response.text
        log.error("HTTP Request Error: %s\nDETAILS: %s", e, error_details)
        error_details_for_log = error_details
        raise ApiRequestError(error_details) from e
    except json.JSONDecodeError as e:
        log.error("Failed to decode API response.")
        error_details_for_log = "Failed to decode API response."
        raise ApiRequestError("Failed to decode API response.") from e
    except requests.exceptions.RequestException as e:
        log.error("Request Error: %s", e)
        error_details_for_log = str(e)
        raise ApiRequestError(str(e)) from e
    finally:
        if not stream:
            if error_details_for_log:
                log_entry["response"] = {"error": error_details_for_log}
            else:
                log_entry["response"] = response_data or {
                    "error": "Request failed, see logs for details."
                }

        if debug_active:
            safe_log_entry = redact_sensitive_info(log_entry)
            if session_raw_logs is not None:
                session_raw_logs.append(safe_log_entry)
            raw_api_logger.info(json.dumps(safe_log_entry))


# --- Reasoning / <think>-block handling -------------------------------------

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_TAGS = (_THINK_OPEN, _THINK_CLOSE)


def _split_safe(text: str) -> tuple[str, str]:
    """
    Splits 'text' into (emittable, carry), holding back any trailing substring
    that could be the start of a think tag split across stream chunks.
    """
    max_partial = max(len(t) for t in _THINK_TAGS) - 1
    for k in range(min(len(text), max_partial), 0, -1):
        suffix = text[-k:]
        if any(tag.startswith(suffix) for tag in _THINK_TAGS):
            return text[:-k], suffix
    return text, ""


def _filter_think_stream(text: str, state: dict) -> tuple[str, str]:
    """
    Stateful removal of balanced <think>...</think> spans from streamed content.

    Returns (visible_text, reasoning_text). 'state' carries 'in_think' (bool)
    and 'carry' (str) across chunks so tags split between deltas are handled.
    Used for the answer portion of a reasoning response (any stray pairs) and
    for non-reasoning engines as a cheap safety net.
    """
    buf = state["carry"] + text
    state["carry"] = ""
    visible: list[str] = []
    reasoning: list[str] = []
    i = 0
    while True:
        if not state["in_think"]:
            idx = buf.find(_THINK_OPEN, i)
            if idx == -1:
                emit, carry = _split_safe(buf[i:])
                visible.append(emit)
                state["carry"] = carry
                break
            visible.append(buf[i:idx])
            i = idx + len(_THINK_OPEN)
            state["in_think"] = True
        else:
            idx = buf.find(_THINK_CLOSE, i)
            if idx == -1:
                emit, carry = _split_safe(buf[i:])
                reasoning.append(emit)
                state["carry"] = carry
                break
            reasoning.append(buf[i:idx])
            i = idx + len(_THINK_CLOSE)
            state["in_think"] = False
    return "".join(visible), "".join(reasoning)


def _parse_token_counts(
    engine_name: str, response_data: dict
) -> tuple[int, int, int, int]:
    """Parses token counts from a non-streaming API response."""
    p, c, r, t = 0, 0, 0, 0
    if not response_data:
        return 0, 0, 0, 0
    if engine_name in ["openai", "groq"]:
        if "usage" in response_data:
            usage = response_data["usage"]
            p = usage.get("prompt_tokens", 0)
            c = usage.get("completion_tokens", 0)
            t = usage.get("total_tokens", 0)
            # Groq/OpenAI reasoning models report reasoning tokens here.
            details = usage.get("completion_tokens_details") or {}
            r = details.get("reasoning_tokens", 0)
    elif engine_name == "gemini":
        usage = response_data.get("usageMetadata", {})
        p = usage.get("promptTokenCount", 0)
        c = usage.get("candidatesTokenCount", 0)
        r = usage.get("cachedContentTokenCount", 0)
        t = usage.get("totalTokenCount", 0)
    elif engine_name == "anthropic":
        usage = response_data.get("usage", {})
        p = usage.get("input_tokens", 0)
        c = usage.get("output_tokens", 0)
        t = p + c
    return p, c, r, t


def _process_stream(
    engine: str,
    response: requests.Response,
    print_stream: bool = True,
    show_reasoning: bool = False,
    reasoning_expected: bool = False,
) -> tuple[str, dict]:
    """
    Processes a streaming API response.

    When 'reasoning_expected' is True (Groq reasoning models), leading content
    is buffered until the first </think> so chain-of-thought is separated from
    the answer even when the model omits the opening <think> tag (a known
    Groq/qwen quirk). The reasoning is shown only under /debug (show_reasoning);
    it never enters the returned response that becomes conversation history.
    """
    full_response, p, c, r, t = "", 0, 0, 0, 0
    think_state = {"in_think": False, "carry": ""}
    answer_mode = False  # reasoning prefix resolved? (reasoning models only)
    prefix_buffer = ""   # ambiguous leading content held until </think> or EOS

    def _emit_answer(text: str) -> None:
        """Filters any stray balanced think pairs, prints, and records answer."""
        nonlocal full_response
        visible, think = _filter_think_stream(text, think_state)
        if think and show_reasoning and print_stream:
            sys.stdout.write(f"{REASONING_COLOR}{think}{RESET_COLOR}")
        if visible and print_stream:
            sys.stdout.write(visible)
        if (visible or think) and print_stream:
            sys.stdout.flush()
        full_response += visible

    def _show_reasoning(text: str) -> None:
        if text and show_reasoning and print_stream:
            sys.stdout.write(f"{REASONING_COLOR}{text}{RESET_COLOR}")
            sys.stdout.flush()

    try:
        for chunk in response.iter_lines():
            if not chunk:
                continue
            decoded_chunk = chunk.decode("utf-8")
            if engine in ["openai", "groq"]:
                if decoded_chunk.startswith("data:"):
                    if "[DONE]" in decoded_chunk:
                        break
                    try:
                        data = json.loads(decoded_chunk.split("data: ", 1)[1])
                        choices = data.get("choices") or []
                        delta = choices[0].get("delta", {}) if choices else {}

                        reasoning_chunk = delta.get("reasoning") or delta.get(
                            "reasoning_content"
                        )
                        content_chunk = delta.get("content")

                        if not reasoning_expected:
                            # Standard path: stream content directly.
                            _show_reasoning(reasoning_chunk)
                            if content_chunk:
                                if print_stream:
                                    sys.stdout.write(content_chunk)
                                    sys.stdout.flush()
                                full_response += content_chunk
                        else:
                            # Reasoning model. Reasoning may arrive as a separate
                            # field, or inline as <think>...</think> (possibly
                            # with the opening tag missing).
                            if reasoning_chunk:
                                _show_reasoning(reasoning_chunk)
                                # Separate-field reasoning => content is clean.
                                if not answer_mode and prefix_buffer:
                                    _emit_answer(prefix_buffer)
                                    prefix_buffer = ""
                                    answer_mode = True
                            if content_chunk:
                                if answer_mode:
                                    _emit_answer(content_chunk)
                                else:
                                    prefix_buffer += content_chunk
                                    idx = prefix_buffer.find(_THINK_CLOSE)
                                    if idx != -1:
                                        reasoning_part = prefix_buffer[:idx].replace(
                                            _THINK_OPEN, ""
                                        )
                                        remainder = prefix_buffer[
                                            idx + len(_THINK_CLOSE):
                                        ]
                                        _show_reasoning(reasoning_part)
                                        prefix_buffer = ""
                                        answer_mode = True
                                        if remainder:
                                            _emit_answer(remainder)

                        usage = data.get("usage")
                        if usage:
                            p = usage.get("prompt_tokens", 0)
                            c = usage.get("completion_tokens", 0)
                            t = usage.get("total_tokens", 0)
                            details = usage.get("completion_tokens_details") or {}
                            r = details.get("reasoning_tokens", 0)
                    except (json.JSONDecodeError, IndexError):
                        continue
            elif engine == "gemini":
                try:
                    data = json.loads(decoded_chunk.split("data: ", 1)[1])
                    if "candidates" in data:
                        text_chunk = (
                            data["candidates"][0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        if print_stream:
                            sys.stdout.write(text_chunk)
                            sys.stdout.flush()
                        full_response += text_chunk
                    if "usageMetadata" in data:
                        p = data["usageMetadata"].get("promptTokenCount", 0)
                        c = data["usageMetadata"].get("candidatesTokenCount", 0)
                        r = data["usageMetadata"].get("cachedContentTokenCount", 0)
                        t = data["usageMetadata"].get("totalTokenCount", 0)
                except (json.JSONDecodeError, IndexError):
                    continue
            elif engine == "anthropic":
                if decoded_chunk.startswith("data:"):
                    data_str = decoded_chunk.split("data: ", 1)[1].strip()
                    if not data_str:
                        continue
                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type")
                        if event_type == "message_start":
                            p = data.get("message", {}).get("usage", {}).get("input_tokens", 0)
                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_chunk = delta.get("text", "")
                                if print_stream:
                                    sys.stdout.write(text_chunk)
                                    sys.stdout.flush()
                                full_response += text_chunk
                        elif event_type == "message_delta":
                            c = data.get("usage", {}).get("output_tokens", 0)
                            t = p + c
                    except json.JSONDecodeError:
                        continue
    except KeyboardInterrupt:
        if print_stream:
            # A newline is needed to move the cursor to the next line after the partial response.
            print(f"\n{SYSTEM_MSG}--> Stream interrupted by user.{RESET_COLOR}")
    except Exception as e:
        if print_stream:
            print(
                f"\n{SYSTEM_MSG}--> Stream interrupted by network/API error: {e}{RESET_COLOR}"
            )
        log.warning("Stream processing error: %s", e)

    # No </think> ever arrived: the reasoning model answered without thinking,
    # so the buffered prefix is the actual answer.
    if reasoning_expected and not answer_mode and prefix_buffer:
        if print_stream:
            sys.stdout.write(prefix_buffer)
            sys.stdout.flush()
        full_response += prefix_buffer
        prefix_buffer = ""

    # Flush any partial-tag carry held back by the answer-portion filter.
    if think_state["carry"] and not think_state["in_think"]:
        if print_stream:
            sys.stdout.write(think_state["carry"])
            sys.stdout.flush()
        full_response += think_state["carry"]

    # Final guard: strip any complete balanced think span that slipped through
    # (no-op on content with no think tags).
    full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL)

    tokens = {"prompt": p, "completion": c, "reasoning": r, "total": t}
    return full_response, tokens


def perform_chat_request(
    engine: AIEngine,
    model: str,
    messages_or_contents: list,
    system_prompt: str | None,
    max_tokens: int,
    stream: bool,
    debug_active: bool = False,
    session_raw_logs: list | None = None,
    print_stream: bool = True,
    show_reasoning: bool = False,
) -> tuple[str, dict]:
    """Executes a single chat request and returns the response text and token dictionary."""
    url = engine.get_chat_url(model, stream)
    payload = engine.build_chat_payload(
        messages_or_contents, system_prompt, max_tokens, stream, model
    )
    headers = engine.get_headers()

    try:
        response_obj = make_api_request(
            url,
            headers,
            payload,
            stream,
            debug_active=debug_active,
            session_raw_logs=session_raw_logs,
        )
    except ApiRequestError as e:
        # For non-streaming, return the error in the response tuple
        if not stream:
            return f"API Error: {e}", {}
        # For streaming, the error is already logged, so we just return empty
        return "", {}

    if stream:
        reasoning_expected = engine.name == "groq" and _is_groq_reasoning_model(model)
        return _process_stream(
            engine.name,
            response_obj,
            print_stream=print_stream,
            show_reasoning=show_reasoning,
            reasoning_expected=reasoning_expected,
        )

    response_data = response_obj
    assistant_response = engine.parse_chat_response(response_data)

    # Reveal reasoning on non-streaming requests when /debug is active. Groq's
    # 'parsed' mode places it in message.reasoning, separate from content.
    if show_reasoning and print_stream and engine.name in ("openai", "groq"):
        choices = response_data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        reasoning_text = message.get("reasoning")
        if reasoning_text:
            print(f"{REASONING_COLOR}{reasoning_text}{RESET_COLOR}")

    p, c, r, t = _parse_token_counts(engine.name, response_data)
    tokens = {"prompt": p, "completion": c, "reasoning": r, "total": t}
    return assistant_response, tokens