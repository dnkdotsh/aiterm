# AITERM: A Unified Command-Line AI Client

A flexible and robust command-line interface for interacting with multiple AI models, including OpenAI (GPT series) and Google (Gemini series). `aiterm` is designed for developers, enabling powerful workflows that integrate local files, persistent memory, and session management directly into your terminal.

## Key Features

-   **Multi-Engine Chat**: Engage with OpenAI, Gemini, or both simultaneously for comparative analysis.
-   **Dynamic File Context**: Attach, detach, and refresh local files or entire directories mid-conversation to provide deep, evolving context for the AI.
-   **Reusable Personas**: Define AI "personas" with pre-configured system prompts, models, and settings for different tasks (e.g., `code_reviewer`, `technical_writer`).
-   **Powerful Memory System**: Consolidate conversations into a long-term memory file. The AI learns from this memory across sessions, and you can selectively remove information with an AI-powered `/forget` command.
-   **Full Session Management**: Save interactive sessions to a file and resume them later, preserving the full conversation history, model state, and file context.
-   **Built-in Review Tool**: An interactive TUI (`aiterm review`) to browse, replay, rename, or re-enter past chat logs and saved sessions.
-   **Image Generation**: Generate images using DALL-E 3 directly from the command line or within an interactive chat.

---

## Installation

### Requirements
-   Python 3.10+
-   API keys for [OpenAI](https://platform.openai.com/api-keys) and/or [Google AI](https://aistudio.google.com/app/apikey)

### Recommended (via pipx)
`pipx` installs Python applications in isolated environments, making them available globally without interfering with other Python projects.

```bash
# Install pipx if you don't have it
pip install pipx

# Install from the project's root directory
pipx install .
```

### Alternate (via pip)
You can also install directly with `pip` into your environment of choice.

```bash
# Install from the project's root directory
pip install .
```
---

## Quick Start

1.  **First Run & Configuration**: The first time you run `aiterm`, it will guide you through creating the necessary config files and a `.env` file for your API keys in its configuration directory (e.g., `~/.config/aiterm` on Linux).

2.  **Start an Interactive Chat**:
    ```bash
    aiterm
    ```
    Once in a session, type `/help` to see a full list of interactive commands.

3.  **Ask a Single-Shot Question with File Context**:
    ```bash
    aiterm -f ./src/main.py -p "Find any potential bugs in this Python code."
    ```

4.  **Start a Chat with a Persona**: Personas are pre-configured roles for the AI.
    ```bash
    # Use the built-in assistant to ask questions about the tool itself
    aiterm -P aiterm_assistant
    ```

5.  **Compare Models Side-by-Side**:
    ```bash
    aiterm --both "Compare and contrast Python's asyncio with traditional threading."
    ```
---

## Full Documentation

-   **[USAGE.md](docs/USAGE.md)**: Practical "cookbook" guides for real-world developer workflows.
-   **[MANUAL.md](docs/MANUAL.md)**: A comprehensive reference for all command-line flags and interactive `/commands`.
-   **[CONTRIBUTING.md](docs/CONTRIBUTING.md)**: Instructions for setting up a development environment and contributing to the project.

---

### Final Thoughts
<small>I literally just made this to make the AI argue against each other. I hope someone else can find a better use for it.</small>
