# tests/test_logger.py
"""
Tests for the logger setup in aiterm/logger.py.
"""

import logging

from aiterm import logger


def test_setup_logger_file_creation_fails(mocker, capsys):
    """
    Tests that the logger setup handles an OSError during file handler
    creation and falls back to a console handler.
    """
    # Arrange: Mock RotatingFileHandler to fail on instantiation
    mocker.patch(
        "aiterm.logger.RotatingFileHandler",
        side_effect=OSError("Permission denied"),
    )
    # The logger instance is a singleton; we need to reset it to re-trigger setup
    # by removing existing handlers and resetting its internal state.
    log_instance = logging.getLogger("aiterm")
    log_instance.handlers = []
    log_instance.propagate = False  # Reset to initial state

    # Act
    # The setup_logger function is called when the module is imported,
    # so we need to reload it to trigger the logic with our mock in place.
    reloaded_log = logger.setup_logger()

    # Assert
    # 1. A critical error was printed to stderr for the user.
    captured = capsys.readouterr()
    assert "CRITICAL: Could not create log file" in captured.err
    assert "Permission denied" in captured.err

    # 2. The logger still has a handler (the console fallback).
    assert len(reloaded_log.handlers) == 1
    assert isinstance(reloaded_log.handlers[0], logging.StreamHandler)
