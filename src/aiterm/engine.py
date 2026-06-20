# aiterm/engine.py
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


import abc
from typing import Any

import requests

from .logger import log
from .settings import settings

# Configuration block for OpenAI-compatible providers
PROVIDER_CONFIGS = {
    "openai": {
        "id": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "headers": {},
    },
    "groq": {
        "id": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "headers": {},
    }
}

# Substrings identifying Groq models that emit reasoning and accept the
# 'reasoning_format' parameter. Non-reasoning models (llama, gemma, kimi)
# reject that parameter with an HTTP 400.
GROQ_REASONING_MARKERS = ("qwen", "gpt-oss", "deepseek", "qwq")


def _is_groq_reasoning_model(model: str) -> bool:
    """True if the given Groq model accepts the reasoning_format parameter."""
    return any(marker in model.lower() for marker in GROQ_REASONING_MARKERS)


class AIEngine(abc.ABC):
    """Abstract base class for an AI engine provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the engine (e.g., 'openai')."""
        pass

    @abc.abstractmethod
    def get_headers(self) -> dict[str, str]:
        """Get the HTTP headers required for requests to this engine."""
        pass

    @abc.abstractmethod
    def get_chat_url(self, model: str, stream: bool) -> str:
        """Get the API endpoint URL for chat completions."""
        pass

    @abc.abstractmethod
    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int | None,
        stream: bool,
        model: str,
    ) -> dict[str, Any]:
        """Build the JSON payload for a chat request."""
        pass

    @abc.abstractmethod
    def parse_chat_response(self, response_data: dict[str, Any]) -> str:
        """Extract the assistant's text response from the API response."""
        pass

    @abc.abstractmethod
    def fetch_available_models(self, task: str) -> list[str]:
        """Fetch a list of available models for a given task."""
        pass


class OpenAICompatibleEngine(AIEngine):
    """AI Engine implementation for OpenAI-compatible providers."""

    def __init__(self, provider_id: str, api_key: str):
        super().__init__(api_key)
        self.provider_id = provider_id
        self.config = PROVIDER_CONFIGS[provider_id]

    @property
    def name(self) -> str:
        return self.provider_id

    def get_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.get("headers", {}))
        return headers

    def get_chat_url(self, model: str, stream: bool) -> str:
        return f"{self.config['base_url']}/chat/completions"

    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int | None,
        stream: bool,
        model: str,
    ) -> dict[str, Any]:
        all_messages = messages.copy()
        if system_prompt:
            all_messages.insert(0, {"role": "system", "content": system_prompt})

        payload = {"model": model, "messages": all_messages, "stream": stream}

        if stream:
            payload["stream_options"] = {"include_usage": True}

        # Groq reasoning models emit chain-of-thought. 'parsed' routes it to a
        # separate field. Only reasoning-capable models accept this parameter;
        # standard models (llama, gemma) return a 400 if it is present.
        if self.provider_id == "groq" and _is_groq_reasoning_model(model):
            payload["reasoning_format"] = "parsed"

        if max_tokens:
            # Newer OpenAI reasoning models use 'max_completion_tokens'.
            # Standard chat models and Groq defaults to 'max_tokens'.
            if model.startswith("o1") or model.startswith("o3"):
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        return payload

    def parse_chat_response(self, response_data: dict[str, Any]) -> str:
        # Local import to break circular dependency: engine -> utils -> engine
        from .utils.message_builder import extract_text_from_message

        if "choices" in response_data and response_data["choices"]:
            message = response_data["choices"][0].get("message")
            if message:
                return extract_text_from_message(message)
        return ""

    def fetch_available_models(self, task: str) -> list[str]:
        try:
            url = f"{self.config['base_url']}/models"
            response = requests.get(
                url, headers=self.get_headers(), timeout=settings["api_timeout"]
            )
            response.raise_for_status()
            model_list = response.json().get("data", [])
            model_ids = [m["id"] for m in model_list]

            if task == "chat":
                # Use a blacklist to ensure o1, o3, and future architectures are included.
                excluded_prefixes = (
                    "dall-e", "tts", "whisper", "text-embedding", "embedding",
                    "babbage", "davinci", "omni-moderation", "text-search", "code-search"
                )
                chat_models = [mid for mid in model_ids if not mid.startswith(excluded_prefixes)]
                return sorted(chat_models)
            elif task == "image":
                image_models = [
                    mid
                    for mid in model_ids
                    if mid.startswith("dall-e") or "image" in mid
                ]
                return sorted(image_models)
        except requests.exceptions.RequestException as e:
            log.warning("Could not fetch %s model list (%s).", "OpenAI" if self.provider_id == "openai" else self.provider_id.capitalize(), e)
        return []


class GeminiEngine(AIEngine):
    """AI Engine implementation for Google Gemini."""

    @property
    def name(self) -> str:
        return "gemini"

    def get_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def get_chat_url(self, model: str, stream: bool) -> str:
        if stream:
            return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int | None,
        stream: bool,
        model: str,
    ) -> dict[str, Any]:
        payload = {"contents": messages}
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
        if max_tokens:
            payload["generationConfig"] = {"maxOutputTokens": max_tokens}
        return payload

    def parse_chat_response(self, response_data: dict[str, Any]) -> str:
        """Safely extracts text from a Gemini API response."""
        try:
            # Check for content and parts before accessing them
            candidate = response_data.get("candidates", [{}])[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"]

            # If no text is found, log the reason but return an empty string
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if finish_reason not in ["UNKNOWN", None]:
                log.info(
                    "Gemini response finished with reason '%s' but no text content.",
                    finish_reason,
                )
            else:
                log.warning(
                    "Could not extract Gemini response part or finish reason. Full response: %s",
                    response_data,
                )
            return ""
        except (KeyError, IndexError):
            log.warning(
                "Could not parse Gemini response due to unexpected structure. Full response: %s",
                response_data,
            )
            return ""

    def fetch_available_models(self, task: str) -> list[str]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            response = requests.get(url, timeout=settings["api_timeout"])
            response.raise_for_status()
            model_list = response.json().get("models", [])
            return sorted(
                [
                    m["name"].replace("models/", "")
                    for m in model_list
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
            )
        except requests.exceptions.RequestException as e:
            log.warning("Could not fetch Gemini model list (%s).", e)
        return []


class AnthropicEngine(AIEngine):
    """AI Engine implementation for Anthropic Claude."""

    @property
    def name(self) -> str:
        return "anthropic"

    def get_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def get_chat_url(self, model: str, stream: bool) -> str:
        return "https://api.anthropic.com/v1/messages"

    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int | None,
        stream: bool,
        model: str,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            # Anthropic strictly requires a max_tokens parameter
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def parse_chat_response(self, response_data: dict[str, Any]) -> str:
        content_blocks = response_data.get("content", [])
        for block in content_blocks:
            if block.get("type") == "text":
                return block.get("text", "")
        return ""

    def fetch_available_models(self, task: str) -> list[str]:
        # Anthropic doesn't have an endpoint for image generation models
        if task != "chat":
            return []
        try:
            url = "https://api.anthropic.com/v1/models"
            response = requests.get(
                url, headers=self.get_headers(), timeout=settings["api_timeout"]
            )
            response.raise_for_status()
            model_list = response.json().get("data", [])
            return sorted([m["id"] for m in model_list if "id" in m])
        except requests.exceptions.RequestException as e:
            log.warning("Could not fetch Anthropic model list (%s).", e)
        return []


def get_engine(engine_name: str, api_key: str) -> AIEngine:
    """Factory function to get an engine instance by name."""
    if engine_name in PROVIDER_CONFIGS:
        return OpenAICompatibleEngine(engine_name, api_key)
    if engine_name == "gemini":
        return GeminiEngine(api_key)
    if engine_name == "anthropic":
        return AnthropicEngine(api_key)
    raise ValueError(f"Unknown engine: {engine_name}")