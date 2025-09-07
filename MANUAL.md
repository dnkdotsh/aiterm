# AITERM User Manual

This document provides a comprehensive reference for all command-line arguments and interactive commands available in `aiterm`. For practical examples and workflow guides, please see [USAGE.md](USAGE.md).

### Table of Contents
1.  [Command-Line Usage](#command-line-usage)
    -   [`chat` Command](#chat-command-default)
    -   [`review` Command](#review-command)
2.  [Interactive Commands](#interactive-commands)
    -   [General & Session Control](#general--session-control)
    -   [Context & Memory Management](#context--memory-management)
    -   [AI & Model Control](#ai--model-control)
    -   [UI & Settings](#ui--settings)
    -   [Multi-Chat Only Commands](#multi-chat-only-commands)

---

## Command-Line Usage

The application has two main commands: `chat` (the default) and `review`.

```
aiterm [chat|review] [OPTIONS]
```

### `chat` Command (Default)
If no command is specified, `chat` is assumed.

```
aiterm [-e {openai,gemini}] [-m MODEL] [--system-prompt PROMPT_OR_PATH]
      [-P PERSONA]
      [-c | -i | -b [PROMPT] | -l FILEPATH]
      [-p PROMPT] [-f PATH] [-x PATH] [--memory | --no-memory]
      [-s NAME] [--stream | --no-stream] [--max-tokens INT] [--debug]
```

#### Modes (mutually exclusive)
-   `-c, --chat`: Interactive chat mode (this is the default behavior if no other mode is selected).
-   `-i, --image`: Image generation mode (OpenAI only). Requires a prompt via `-p` or piped input.
-   `-b, --both [PROMPT]`: Multi-chat mode to query both OpenAI and Gemini simultaneously. Can be started with an optional initial prompt.
-   `-l, --load FILEPATH`: Load a saved chat session from a `.json` file and resume interaction.

#### Context & Input Arguments
-   `-p, --prompt PROMPT`: A single, non-interactive prompt for a one-shot answer. If this is used, the application will exit after the response.
-   `-P, --persona PERSONA`: Start the session with a specific persona (e.g., `code_reviewer`).
-   `--system-prompt PROMPT_OR_PATH`: Provide a system instruction as a string or a path to a text file. This overrides a persona's system prompt.
-   `-f, --file PATH`: Attach a file or directory to the conversation context. Can be used multiple times.
-   `-x, --exclude PATH`: Exclude a file or directory from being processed by `-f`.
-   `--memory / --no-memory`: Explicitly enable or disable the use of the persistent memory file for the session, overriding the default in `settings.json`.

#### Session Control Arguments
-   `-s, --session-name NAME`: Provide a custom name for the chat log file.
-   `--stream / --no-stream`: Explicitly enable or disable streaming for chat responses.
-   `--max-tokens INT`: Set the maximum number of tokens the AI should generate in its response.
-   `--debug`: Start the session with raw API request/response logging enabled.

### `review` Command
Launch an interactive TUI to browse, replay, rename, or delete past chat logs and saved sessions.

```bash
# Launch the interactive review tool
aiterm review

# Directly replay a specific session file without entering the menu
aiterm review ~/.local/share/aiterm/sessions/my_session.json
```

---

## Interactive Commands

During an interactive session, type these commands at the `You:` or `Director>` prompt.

### General & Session Control
-   `/help`: Display this help message.
-   `/exit [name]`: Save and end the session, updating persistent memory by default. Optionally provide a final name for the log file.
-   `/quit`: Exit immediately without updating memory or renaming the log.
-   `/save <name> [--stay] [--remember]`: Save the current state of the session to a `.json` file. By default, this also exits.
    -   `--stay`: Saves the session but continues the conversation (creates a checkpoint).
    -   `--remember`: Explicitly updates persistent memory. Without this, saving does not consolidate the chat into memory.
-   `/load <filename>`: Load a session from a `.json` file, replacing the current one.
-   `/clear`: Clear the current conversation history from the session.
-   `/history`: Print the full JSON of the current conversation history.
-   `/state`: Print the current session's configuration (model, persona, attachments, etc.).
-   `/debug`: Toggle raw API logging for the current session.

### Context & Memory Management
-   `/attach <path>`: Attach a file or directory to the session context.
-   `/detach <name>`: Detach a file from the context by its filename.
-   `/files`: List all currently attached text files in a tree view.
-   `/refresh [term]`: Re-read attached files. If `[term]` is given, only refreshes files whose names contain that term.
-   `/memory`: View the contents of the persistent memory file.
-   `/remember [text]`: Injects `[text]` directly into persistent memory. If run without text, it consolidates the current chat into memory.

### AI & Model Control
-   `/engine [name]`: Switch AI engine (`openai` or `gemini`). Toggles between them if no name is given.
-   `/model [name]`: Select a new model for the current engine. Prompts with a list if no name is given.
-   `/persona <name>`: Switch to a different persona. Use `/persona clear` to remove the active persona.
-   `/personas`: List all available personas.
-   `/stream`: Toggle response streaming on/off for the session.
-   `/max-tokens <num>`: Set the max output tokens for the session.
-   `/image [prompt]`: Initiate the interactive image generation workflow.

### UI & Settings
-   `/set <key> <value>`: Change an application setting permanently (e.g., `/set stream false`).
-   `/toolbar [on|off|toggle <comp>]`: Control the bottom toolbar. Components: `io`, `live`, `model`, `persona`.
-   `/theme <name>`: Switch the display theme. Run without a name to list available themes.

### Multi-Chat Only Commands
-   `/ai <gpt|gem> [prompt]`: Send a targeted prompt to only one AI. If no prompt is given, the AI is asked to continue the conversation.
