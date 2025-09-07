# tests/utils/test_redaction.py
"""
Tests for the redaction utility in aiterm/utils/redaction.py.
"""

from aiterm.utils.redaction import redact_sensitive_info


class TestRedaction:
    def test_redact_sensitive_info_deep(self):
        """
        Tests recursive redaction of API keys in URLs, headers, and payloads,
        including nested structures.
        """
        log_entry = {
            "timestamp": "2024-01-01T12:00:00",
            "request": {
                "url": "https://api.gemini.com/v1/models?key=AIzaSy_supersecretkey_1234567890",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer sk-proj-anothersecretkey1234567890",
                },
                "payload": {
                    "data": "test",
                    "nested": {"api_key": "payload_secret"},
                    "tokens": [{"type": "auth", "token": "a_third_secret"}],
                },
            },
            "response": {
                "error": "Invalid key provided: AIzaSy_supersecretkey_1234567890. Please check your credentials."
            },
        }
        redacted = redact_sensitive_info(log_entry)

        # Test request redaction
        assert "key=[REDACTED]" in redacted["request"]["url"]
        assert "AIzaSy" not in redacted["request"]["url"]
        assert redacted["request"]["headers"]["Authorization"] == "Bearer [REDACTED]"
        assert "sk-proj-" not in redacted["request"]["headers"]["Authorization"]
        assert redacted["request"]["payload"]["nested"]["api_key"] == "[REDACTED]"
        assert redacted["request"]["payload"]["tokens"][0]["token"] == "[REDACTED]"

        # Test response body redaction
        assert "[REDACTED_GEMINI_KEY]" in redacted["response"]["error"]
        assert "AIzaSy_supersecretkey" not in redacted["response"]["error"]

        # Ensure original data is not mutated
        assert "AIzaSy_supersecretkey" in log_entry["request"]["url"]
        assert "sk-proj-" in log_entry["request"]["headers"]["Authorization"]
        assert "AIzaSy_supersecretkey" in log_entry["response"]["error"]
        assert log_entry["request"]["payload"]["nested"]["api_key"] == "payload_secret"

    def test_redact_non_sensitive_info_is_untouched(self):
        """Tests that data structures without sensitive info are not changed."""
        log_entry = {
            "request": {
                "url": "http://example.com/api/v1/resource",
                "payload": {"data": "some content", "user_id": 123},
            },
            "response": {"status": "ok", "message": "Success"},
        }
        # Use a simple way to check for equality without deepdiff
        original_json = str(log_entry)
        redacted = redact_sensitive_info(log_entry)
        redacted_json = str(redacted)

        assert original_json == redacted_json
