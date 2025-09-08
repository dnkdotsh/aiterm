# tests/test_workflows.py
"""
Tests for the complex, multi-step workflows in aiterm/workflows.py.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiterm import api_client, prompts, workflows


class TestAIHelperWorkflows:
    """Tests for workflows that use a helper AI model."""

    def test_consolidate_memory(self, mocker, mock_session_manager):
        """Tests that consolidate_memory formats the prompt and writes the result."""
        # Arrange
        mocker.patch.object(
            mock_session_manager.context_manager,
            "_read_memory_file",
            return_value="existing",
        )
        mocker.patch.object(
            mock_session_manager, "_get_history_for_helpers", return_value="history"
        )
        mock_helper = mocker.patch.object(
            mock_session_manager,
            "_perform_helper_request",
            return_value=("updated memory", {}),
        )
        mock_write = mocker.patch.object(
            mock_session_manager.context_manager, "_write_memory_file"
        )
        expected_prompt = prompts.MEMORY_INTEGRATION_PROMPT.format(
            existing_ltm="existing", session_content="history"
        )

        # Act
        workflows.consolidate_memory(mock_session_manager)

        # Assert
        mock_helper.assert_called_once_with(expected_prompt, None)
        mock_write.assert_called_once_with("updated memory")

    def test_inject_memory(self, mocker, mock_session_manager):
        """Tests that inject_memory formats the prompt and writes the result."""
        # Arrange
        mocker.patch.object(
            mock_session_manager.context_manager,
            "_read_memory_file",
            return_value="existing",
        )
        mock_helper = mocker.patch.object(
            mock_session_manager,
            "_perform_helper_request",
            return_value=("updated memory", {}),
        )
        mock_write = mocker.patch.object(
            mock_session_manager.context_manager, "_write_memory_file"
        )
        fact = "new fact"
        expected_prompt = prompts.DIRECT_MEMORY_INJECTION_PROMPT.format(
            existing_ltm="existing", new_fact=fact
        )

        # Act
        workflows.inject_memory(mock_session_manager, fact)

        # Assert
        mock_helper.assert_called_once_with(expected_prompt, None)
        mock_write.assert_called_once_with("updated memory")

    def test_rename_log_with_ai(self, mocker, mock_session_manager):
        """Tests that rename_log_with_ai gets a name and calls the rename method."""
        # Arrange
        mocker.patch.object(
            mock_session_manager, "_get_history_for_helpers", return_value="history"
        )
        mock_helper = mocker.patch.object(
            mock_session_manager,
            "_perform_helper_request",
            return_value=("ai_suggested_name", {}),
        )
        mock_rename = mocker.patch.object(mock_session_manager, "_rename_log_file")
        log_path = Path("/fake/log.jsonl")

        # Act
        workflows.rename_log_with_ai(mock_session_manager, log_path)

        # Assert
        mock_helper.assert_called_once()
        mock_rename.assert_called_once_with(log_path, "ai_suggested_name")

    def test_scrub_memory(self, mocker, mock_session_manager):
        """Tests that scrub_memory formats the prompt and writes the result."""
        # Arrange
        mocker.patch.object(
            mock_session_manager.context_manager,
            "_read_memory_file",
            return_value="existing memory about aiterm",
        )
        mock_helper = mocker.patch.object(
            mock_session_manager,
            "_perform_helper_request",
            return_value=("rewritten memory", {}),
        )
        mock_write = mocker.patch.object(
            mock_session_manager.context_manager, "_write_memory_file"
        )
        topic = "aiterm"
        expected_prompt = prompts.MEMORY_SCRUB_PROMPT.format(
            existing_ltm="existing memory about aiterm", topic=topic
        )

        # Act
        workflows.scrub_memory(mock_session_manager, topic)

        # Assert
        mock_helper.assert_called_once_with(expected_prompt, None)
        mock_write.assert_called_once_with("rewritten memory")

    def test_scrub_memory_empty(self, mocker, mock_session_manager, capsys):
        """Tests that scrub_memory does nothing if memory is empty."""
        mocker.patch.object(
            mock_session_manager.context_manager,
            "_read_memory_file",
            return_value="",
        )
        mock_helper = mocker.patch.object(
            mock_session_manager,
            "_perform_helper_request",
        )

        workflows.scrub_memory(mock_session_manager, "aiterm")

        mock_helper.assert_not_called()
        captured = capsys.readouterr().out
        assert "Persistent memory is empty" in captured


class TestPerformImageGeneration:
    """Tests for the core _perform_image_generation function."""

    @pytest.fixture(autouse=True)
    def mock_deps(self, mocker):
        """Auto-mock dependencies for this test class."""
        self.mock_api = mocker.patch("aiterm.workflows.api_client.make_api_request")
        self.mock_save = mocker.patch(
            "aiterm.workflows.save_image_and_get_path",
            return_value=Path("/fake/image.png"),
        )
        self.mock_log = mocker.patch("aiterm.workflows.log_image_generation")
        self.mock_b64 = mocker.patch("aiterm.workflows.base64.b64decode")

    def test_success(self):
        """Tests the successful image generation path."""
        # Arrange
        self.mock_api.return_value = {"data": [{"b64_json": "fakedata"}]}

        # Act
        success, filepath = workflows._perform_image_generation(
            "fake_key", "dall-e-3", "a prompt"
        )

        # Assert
        assert success is True
        assert filepath == "/fake/image.png"
        self.mock_api.assert_called_once()
        self.mock_b64.assert_called_once_with("fakedata")
        self.mock_save.assert_called_once()
        self.mock_log.assert_called_once()

    def test_api_error(self, capsys):
        """Tests handling of an ApiRequestError."""
        # Arrange
        self.mock_api.side_effect = api_client.ApiRequestError("API failed")

        # Act
        success, filepath = workflows._perform_image_generation(
            "fake_key", "dall-e-3", "a prompt"
        )

        # Assert
        assert success is False
        assert filepath is None
        assert "Error: API failed" in capsys.readouterr().err
        self.mock_save.assert_not_called()

    def test_no_image_data_in_response(self, capsys):
        """Tests handling of a successful API call that returns no image data."""
        # Arrange
        self.mock_api.return_value = {"data": []}  # Empty data list

        # Act
        success, filepath = workflows._perform_image_generation(
            "fake_key", "dall-e-3", "a prompt"
        )

        # Assert
        assert success is False
        assert filepath is None
        assert "did not contain image data" in capsys.readouterr().err
        self.mock_save.assert_not_called()

    def test_save_error(self, capsys):
        """Tests handling of an OSError during image saving."""
        # Arrange
        self.mock_api.return_value = {"data": [{"b64_json": "fakedata"}]}
        self.mock_save.side_effect = OSError("Disk full")

        # Act
        success, filepath = workflows._perform_image_generation(
            "fake_key", "dall-e-3", "a prompt"
        )

        # Assert
        assert success is False
        assert filepath is None
        assert "Error saving image: Disk full" in capsys.readouterr().err


class TestImageGenerationWorkflow:
    """Tests for the ImageGenerationWorkflow class."""

    @pytest.fixture
    def mock_workflow(self, mock_openai_session_manager):
        """Provides an ImageGenerationWorkflow instance with a mocked session."""
        return workflows.ImageGenerationWorkflow(mock_openai_session_manager)

    def test_run_starts_crafting_with_prompt(self, mocker, mock_workflow):
        """Tests that `run` with a prompt starts the crafting process."""
        mock_start = mocker.patch.object(mock_workflow, "_start_prompt_crafting")
        mock_workflow.run(["a cool prompt"])
        mock_start.assert_called_once_with("a cool prompt")

    def test_run_send_immediately(self, mocker, mock_workflow):
        """Tests that `run` with --send-prompt generates immediately."""
        mock_generate = mocker.patch.object(
            mock_workflow, "_generate_image_from_session"
        )
        mock_workflow.run(["a cool prompt", "--send-prompt"])
        mock_generate.assert_called_once_with("a cool prompt")

    def test_run_handles_engine_switch(self, mocker, mock_session_manager):
        """Tests that the engine is switched to OpenAI if not already active."""
        mock_handle_engine = mocker.patch("aiterm.commands.handle_engine")
        # Use a non-OpenAI session manager for this test
        workflow = workflows.ImageGenerationWorkflow(mock_session_manager)
        assert workflow.session.state.engine.name == "gemini"

        workflow.run(["a prompt"])

        mock_handle_engine.assert_called_once_with(["openai"], mock_session_manager)
        assert workflow.pre_image_engine == "gemini"

    def test_process_prompt_input_generate(self, mock_workflow):
        """Tests that 'yes' input triggers generation."""
        mock_workflow.img_prompt = "a final prompt"
        should_generate, final_prompt = mock_workflow.process_prompt_input("yes")
        assert should_generate is True
        assert final_prompt == "a final prompt"

    def test_process_prompt_input_cancel(self, mock_workflow):
        """Tests that 'no' input cancels the workflow."""
        mock_revert = MagicMock()
        mock_workflow._revert_engine_after_crafting = mock_revert
        mock_workflow.img_prompt_crafting = True

        should_generate, final_prompt = mock_workflow.process_prompt_input("cancel")

        assert should_generate is False
        assert final_prompt is None
        assert mock_workflow.img_prompt_crafting is False
        mock_revert.assert_called_once()

    def test_process_prompt_input_refine(self, mock_workflow):
        """Tests that other input refines the prompt."""
        mock_workflow.session._perform_helper_request = MagicMock(
            return_value=("a refined prompt", {})
        )
        mock_workflow.img_prompt = "an initial prompt"
        should_generate, final_prompt = mock_workflow.process_prompt_input(
            "make it better"
        )
        assert should_generate is False
        assert final_prompt is None
        assert mock_workflow.img_prompt == "a refined prompt"

    def test_generate_image_from_session_success(self, mocker, mock_workflow):
        """Tests a successful generation and history update."""
        mocker.patch(
            "aiterm.workflows._perform_image_generation",
            return_value=(True, "/path/to/image.png"),
        )
        initial_history_len = len(mock_workflow.session.state.history)

        result = mock_workflow._generate_image_from_session("a prompt")

        assert result is True
        assert len(mock_workflow.session.state.history) == initial_history_len + 2
        last_msg = mock_workflow.session.state.history[-1]
        assert "I've generated an image" in last_msg["content"]
        assert "/path/to/image.png" in last_msg["content"]

    def test_revert_engine_after_crafting(self, mocker, mock_workflow):
        """Tests that the engine and model are reverted correctly."""
        mock_handle_engine = mocker.patch("aiterm.commands.handle_engine")
        mock_handle_model = mocker.patch("aiterm.commands.handle_model")

        mock_workflow.pre_image_engine = "gemini"
        mock_workflow.pre_image_model = "gemini-1.5-flash"

        mock_workflow._revert_engine_after_crafting()

        mock_handle_engine.assert_called_once_with(["gemini"], mock_workflow.session)
        mock_handle_model.assert_called_once_with(
            ["gemini-1.5-flash"], mock_workflow.session
        )
        assert mock_workflow.pre_image_engine is None
        assert mock_workflow.pre_image_model is None
