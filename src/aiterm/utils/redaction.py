# aiterm/utils/redaction.py
# aiterm: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

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

"""
Centralized utility for redacting sensitive information from data structures.
"""

import copy
import re
from typing import Any

# Regex for known API key patterns.
OPENAI_KEY_PATTERN = re.compile(r"sk-(proj-)?\w{20,}")
GEMINI_KEY_PATTERN = re.compile(r"AIzaSy[A-Za-z0-9\-_]{20,}")

# A set of dictionary keys whose values should always be redacted.
SENSITIVE_KEYS = {"api_key", "key", "token", "authorization"}


def redact_sensitive_info(data: Any) -> Any:
    """
    Recursively traverses a dict or list to redact sensitive information.
    This function is non-destructive and returns a new, redacted object.
    It redacts:
    1. Values of keys found in the SENSITIVE_KEYS set (case-insensitive).
    2. String values that match known API key patterns or URL key parameters.
    """

    def _redact_recursive(sub_data: Any) -> Any:
        """Inner recursive function that operates on parts of the copied data."""
        if isinstance(sub_data, dict):
            new_dict = {}
            for key, value in sub_data.items():
                # Rule 1: Check for sensitive key names (case-insensitive)
                if key.lower() in SENSITIVE_KEYS:
                    # Rule 1a: Special format for Authorization header
                    if key == "Authorization":
                        new_dict[key] = "Bearer [REDACTED]"
                    else:
                        new_dict[key] = "[REDACTED]"
                else:
                    # Rule 2: If key is not sensitive, recurse on the value
                    new_dict[key] = _redact_recursive(value)
            return new_dict

        if isinstance(sub_data, list):
            return [_redact_recursive(item) for item in sub_data]

        if isinstance(sub_data, str):
            # Rule 3: Apply pattern-based redaction to all strings
            sub_data = re.sub(r"key=([^&]+)", "key=[REDACTED]", sub_data)
            sub_data = OPENAI_KEY_PATTERN.sub("[REDACTED_OPENAI_KEY]", sub_data)
            sub_data = GEMINI_KEY_PATTERN.sub("[REDACTED_GEMINI_KEY]", sub_data)
            return sub_data

        return sub_data

    # Start with a deepcopy to ensure the original data is not mutated
    data_copy = copy.deepcopy(data)
    return _redact_recursive(data_copy)
