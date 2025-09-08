# tests/utils/test_config_loader.py
import argparse

import pytest
from aiterm import config
from aiterm.personas import Persona
from aiterm.utils.config_loader import resolve_config_precedence


@pytest.fixture
def mock_args():
    """Provides a basic argparse.Namespace object with all values as None."""
    return argparse.Namespace(
        engine=None,
        model=None,
        prompt=None,
        persona=None,
        system_prompt=None,
        file=None,
        exclude=None,
        memory=None,
        session_name=None,
        stream=None,
        max_tokens=None,
        debug=False,
    )


@pytest.fixture
def mock_persona_loader(mocker):
    """Fixture to mock the persona loading functions."""
    return mocker.patch("aiterm.utils.config_loader.personas.load_persona")


# Use the centralized settings patcher from conftest.py
@pytest.mark.usefixtures("mock_settings_patcher")
class TestConfigLoader:
    def test_defaults_only(self, mock_args, mock_persona_loader):
        """Test that default settings are used when no CLI args or persona are provided."""
        # Simulate interactive mode, which should try to load default persona
        mock_persona_loader.return_value = None
        result = resolve_config_precedence(mock_args)
        assert result["engine_name"] == "gemini"
        assert result["model"] == "gemini-default"
        assert result["max_tokens"] == 1000
        assert result["stream"] is True
        assert result["memory_enabled"] is True

    def test_persona_overrides_defaults(self, mock_args, mock_persona_loader):
        """Test that persona settings override the default settings."""
        mock_args.persona = "coder"
        persona = Persona(
            name="Coder",
            filename="coder.json",
            engine="openai",
            model="gpt-4-turbo",
            max_tokens=2048,
            stream=False,
        )
        mock_persona_loader.return_value = persona

        result = resolve_config_precedence(mock_args)

        assert result["engine_name"] == "openai"
        assert result["model"] == "gpt-4-turbo"
        assert result["max_tokens"] == 2048
        assert result["stream"] is False
        assert result["memory_enabled"] is True  # Persona didn't override this
        assert result["persona"] == persona
        mock_persona_loader.assert_called_with("coder")

    def test_cli_overrides_persona(self, mock_args, mock_persona_loader):
        """Test that CLI arguments override both persona and default settings."""
        mock_args.persona = "coder"
        mock_args.engine = "gemini"
        mock_args.model = "gemini-1.5-pro"
        mock_args.max_tokens = 4096
        mock_args.stream = True
        mock_args.memory = False  # --no-memory

        persona = Persona(
            name="Coder",
            filename="coder.json",
            engine="openai",
            model="gpt-4-turbo",
            max_tokens=2048,
            stream=False,
        )
        mock_persona_loader.return_value = persona

        result = resolve_config_precedence(mock_args)

        assert result["engine_name"] == "gemini"
        assert result["model"] == "gemini-1.5-pro"
        assert result["max_tokens"] == 4096
        assert result["stream"] is True
        assert result["memory_enabled"] is False  # CLI override

    def test_single_shot_disables_memory(self, mock_args):
        """Test that providing a prompt via CLI disables memory."""
        mock_args.prompt = "What is the capital of France?"
        # memory=None means neither --memory nor --no-memory was passed
        mock_args.memory = None

        result = resolve_config_precedence(mock_args)
        assert result["memory_enabled"] is False

    def test_no_memory_flag(self, mock_args):
        """Test that --no-memory flag correctly disables memory."""
        mock_args.memory = False
        result = resolve_config_precedence(mock_args)
        assert result["memory_enabled"] is False

    def test_default_persona_loaded_in_interactive(
        self, mock_args, mock_persona_loader
    ):
        """Test that the default persona is loaded in interactive mode if available."""
        # No persona, no system prompt, interactive mode implied by prompt=None
        default_persona = Persona(
            name="AITERM Assistant", filename="aiterm_assistant.json"
        )
        mock_persona_loader.return_value = default_persona
        result = resolve_config_precedence(mock_args)
        mock_persona_loader.assert_called_with("aiterm_assistant")
        assert result["persona"] == default_persona

    def test_system_prompt_prevents_default_persona(
        self, mock_args, mock_persona_loader
    ):
        """Test that providing a system prompt prevents loading the default persona."""
        mock_args.system_prompt = "You are a pirate."
        result = resolve_config_precedence(mock_args)
        mock_persona_loader.assert_not_called()
        assert result["persona"] is None
        assert result["system_prompt_arg"] == "You are a pirate."

    def test_persona_model_mismatch_engine(self, mock_args, mock_persona_loader):
        """Test that a persona's model is ignored if its engine doesn't match the final engine."""
        mock_args.engine = "gemini"  # CLI forces gemini
        persona = Persona(
            name="Coder",
            filename="coder.json",
            engine="openai",  # Persona is for openai
            model="gpt-4-turbo",
        )
        mock_persona_loader.return_value = persona

        result = resolve_config_precedence(mock_args)

        assert result["engine_name"] == "gemini"
        # The model should fall back to the default for the CLI-specified engine
        assert result["model"] == "gemini-default"

    def test_persona_object_passed_through_with_attachments(
        self, fake_fs, mock_args, mock_persona_loader
    ):
        """
        Test that the persona object, with its pre-resolved attachment paths,
        is passed through correctly.
        """
        # Create fake persona and doc files for context
        persona_path = config.PERSONAS_DIRECTORY / "doc_expert.json"
        # The doc path that would be resolved by `load_persona`
        doc_path = config.CONFIG_DIR / "docs/file.md"
        fake_fs.create_file(persona_path)
        fake_fs.create_file(doc_path, contents="doc content")

        mock_args.persona = "doc_expert"
        # Simulate that `load_persona` has already resolved the path.
        resolved_path = str(doc_path.resolve())
        persona_with_resolved_paths = Persona(
            name="Doc Expert",
            filename="doc_expert.json",
            attachments=[resolved_path],
        )
        mock_persona_loader.return_value = persona_with_resolved_paths

        result = resolve_config_precedence(mock_args)

        # The test now simply checks if the correct persona object is in the result.
        # The handler (`handlers.py`) will then be responsible for reading the
        # `attachments` from this object.
        assert result["persona"] is not None
        assert result["persona"].attachments == [resolved_path]
