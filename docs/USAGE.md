# AITERM: Practical Workflows and Examples

This document goes beyond the basic commands found in `README.md` to demonstrate how `aiterm` can be integrated into powerful, real-world developer workflows. The true strength of this tool lies in combining its features—personas, context management, and session control—to solve complex problems efficiently.

### Table of Contents
1.  [Core Concepts Refresher](#core-concepts-refresher)
2.  [Workflow 1: Deep Code Review and Iterative Debugging](#workflow-1-deep-code-review-and-iterative-debugging)
3.  [Workflow 2: Querying a Project's Documentation](#workflow-2-querying-a-projects-documentation)
4.  [Workflow 3: Long-Form Content Creation Across Sessions](#workflow-3-long-form-content-creation-across-sessions)
5.  [Workflow 4: Comparative Analysis of AI Models](#workflow-4-comparative-analysis-of-ai-models)
6.  [Tips and Best Practices](#tips-and-best-practices)

---

### Core Concepts Refresher

The following workflows are built on three core features:

*   **Personas (`-P`, `/persona`):** Pre-configuring the AI's role, model, and instructions is the key to getting high-quality, specialized responses. A good persona saves you from retyping the same system prompt every time. The built-in `aiterm_assistant` persona is a great way to learn about the tool itself.
*   **Context Management (`-f`, `-x`, `/attach`, `/refresh`):** The ability to provide the AI with the *exact* files it needs to reason about a problem. Attaching a directory (`-f ./src`) is often the fastest way to get started.
*   **Session & Memory (`/save`, `/load`, `/remember`, `/forget`):** For tasks that take more than a few minutes, saving your session preserves the entire conversation history and context. Using `/remember` allows the AI to build a long-term understanding of your project, while `/forget` can be used to correct the AI's path or remove sensitive information.

---

### Workflow 1: Deep Code Review and Iterative Debugging

**Scenario:** You've been working on a new feature in your Python project located in `src/feature_x/`. It's not working as expected, and you suspect there's a bug in how `api_handler.py` and `data_processor.py` interact.

**Step-by-Step:**

1.  **Start with a specialized persona.** Create a `code_reviewer.json` persona if you don't have one. Launch `aiterm`, attaching the entire feature directory for full context.

    ```bash
    aiterm -P code_reviewer -f ./src/feature_x/
    ```

2.  **Give the initial high-level prompt.** At the `You:` prompt, type:

    > I'm having an issue with the new feature. The data processing seems to fail when it receives data from the API handler. Can you review the attached code, specifically `api_handler.py` and `data_processor.py`, and look for potential race conditions or data mismatch errors?

    *The AI will now analyze the files you provided and give you its initial feedback.*

3.  **Correct a wrong turn (Optional).** The AI suggests a complex refactor, but you realize your prompt was misleading. Instead of continuing, you can remove the last conversational turn to get it back on track.

    > **You:** `/forget`

    *The system confirms the last exchange was removed. You can now re-phrase your prompt for a better result.*

4.  **Modify the code.** Based on the AI's new, more accurate suggestion, you edit `data_processor.py` in your IDE to fix the bug.

5.  **Refresh the context.** Back in `aiterm`, enter the following command at the `You:` prompt:

    > `/refresh data_processor.py`

    *The system will confirm the file has been re-read and the AI has been notified of the update.*

6.  **Re-evaluate the code.** Now that the AI's context is fresh, ask it to review your changes.

    > **You:** I've updated the file as we discussed. Does the new logic in `data_processor.py` resolve the potential issue you identified?

7.  **End the session.** Once you're done, save the log with a descriptive name.

    > **You:** `/exit feature_x_debug_session`

**Why this is powerful:** You never left the terminal. The `/refresh` command is the key to an iterative workflow, allowing the AI to see your changes in real-time. The `/forget` command acts as an "undo" for the conversation, preventing the AI from getting stuck on an incorrect path.

---

### Workflow 2: Querying a Project's Documentation

**Scenario:** You're new to a team and need to understand the project's custom deployment process. The documentation is spread across multiple markdown files in the `/docs` directory.

**Step-by-Step:**

1.  **Start `aiterm` and attach the entire docs folder.** A persona isn't strictly necessary here, but a targeted system prompt is very effective.

    ```bash
    aiterm -f ./docs --system-prompt "You are a helpful assistant. Answer questions based *only* on the attached files. If the answer is not in the files, say so."
    ```

2.  **Ask your questions at the prompt.**

    > **You:** How do I deploy a new service to the staging environment?

    *The AI will read through all the `.md` files and synthesize an answer.*

    > **You:** What are the required environment variables for the authentication service?

**Why this is powerful:** `aiterm` effectively becomes a natural language interface for your project's knowledge base. It's often faster than manually searching and reading through multiple files.

---

### Workflow 3: Long-Form Content Creation Across Sessions

**Scenario:** You are writing a technical blog post about a new technology. You plan to work on it over several days.

**Step-by-Step:**

1.  **Start with a persona** to set the tone and voice for your article.

    ```bash
    aiterm -P technical_writer
    ```

2.  **Brainstorm and draft.** Have a conversation with the AI to create an outline and draft the first few sections.

3.  **Save your session before you finish for the day.** At the prompt, type:

    > `/save blog_post_draft`

    *This command saves the entire session, including conversation history and attached files, to `blog_post_draft.json` and then exits the application.*

    *(Note: Use `/save --stay` if you only want to create a checkpoint without exiting.)*

4.  **Resume your work the next day.** Launch `aiterm` using the `-l` flag.

    ```bash
    # You don't need to specify a persona or files; they are stored in the session.
    aiterm -l blog_post_draft.json
    ```

    *The session is loaded, and you can see the last few messages of your previous conversation, ready to continue.*

    > **You:** Okay, let's continue with the section on "Advanced Techniques".

5.  **Consolidate key points into memory.** Once the draft is complete, you can distill the core concepts into long-term memory for future articles.

    > **You:** `/remember`

**Why this is powerful:** The `/save` and `/load` commands turn a stateless conversation into a persistent, stateful project. The AI's context and your conversation history are never lost, allowing for deep, long-running tasks.

---

### Workflow 4: Comparative Analysis of AI Models

**Scenario:** You need to implement a complex text classification algorithm. You're not sure if OpenAI's or Gemini's model will provide a more nuanced result.

**Step-by-Step:**

1.  **Attach the relevant context.** This could be a requirements document or a sample dataset.

    ```bash
    # Attach the project requirements and start in multi-chat mode
    aiterm --both -f ./requirements.md
    ```

2.  **Pose the complex question to both models.** At the `Director>` prompt, type:

    > Based on the attached requirements, propose a high-level strategy for a text classification system. Pay close attention to the need for handling ambiguous language. Which specific techniques or models in your arsenal would you recommend?

3.  **Analyze and compare the responses.** The `[OpenAI]` and `[Gemini]` models will both respond. You can directly compare their reasoning, suggestions, and approaches in the same terminal view.

4.  **Ask follow-up questions to a specific model.** If one model gives a more promising answer, you can target it directly using the `/ai` command.

    > **Director>** `/ai gpt Your suggestion to use a fine-tuned model is interesting. Could you elaborate on the potential costs and data requirements for that approach?`

**Why this is powerful:** This workflow leverages the original "fun" purpose of the tool for a serious professional goal: model evaluation. It provides a direct, side-by-side comparison of AI capabilities on the specific problems you care about.

---

### Tips and Best Practices

*   **Be Mindful of Cost:** Attaching large directories can consume a lot of tokens, which costs money. Use the `-x` flag to exclude irrelevant subdirectories (like `node_modules` or `__pycache__`).
*   **Invest in Personas:** A well-crafted persona is the single best way to improve the quality of the AI's output. Be specific in your instructions.
*   **Name Your Sessions:** When you `/exit` or `/save`, give your sessions descriptive names (`/exit my_feature_refactor`). This makes them much easier to find and review later with `aiterm review`.
*   **Use Shell Piping:** For single-shot questions about a specific file, shell piping can be very convenient: `cat my_file.py | aiterm -p "Find any bugs in this Python code."`
