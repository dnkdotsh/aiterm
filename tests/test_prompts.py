# tests/test_prompts.py
"""
Tests for the prompt constants in aiterm/prompts.py.
"""

from aiterm import prompts


class TestPrompts:
    """Test suite for prompt constants."""

    def test_memory_scrub_prompt_placeholders(self):
        """
        Tests that MEMORY_SCRUB_PROMPT contains the required placeholders.
        """
        prompt_text = prompts.MEMORY_SCRUB_PROMPT
        assert "{existing_ltm}" in prompt_text
        assert "{topic}" in prompt_text
