# aiterm/prompts.py
# aiterm: A command-line interface for interacting with AI models.
# Copyright (C) 2025 Dank A. Saurus

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY;
# without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


"""
A centralized collection of prompts for automated AI tasks.
"""

# --- From session_manager.py ---

HISTORY_SUMMARY_PROMPT = (
    "Concisely summarize the key facts and takeaways from the following conversation excerpt in the third person. "
    "This summary will be used as context for the rest of the conversation.\n\n"
    "--- EXCERPT ---\n{log_content}\n---"
)

MEMORY_INTEGRATION_PROMPT = (
    "You are a memory consolidation agent. Your task is to distill the crucial information from the 'NEW CHAT SESSION' "
    "and integrate it into the 'EXISTING PERSISTENT MEMORY'. Synthesize related topics, update existing facts with new "
    "information, and discard conversational fluff or trivial data. The final output must be a dense, factual summary, "
    "optimized for conciseness and relevance for a future AI to use as context. Eliminate all verbosity and unnecessary "
    "formatting and markdown.\n\n"
    "--- EXISTING PERSISTENT MEMORY ---\n{existing_ltm}\n\n"
    "--- NEW CHAT SESSION TO INTEGRATE ---\n{session_content}\n\n"
    "--- UPDATED PERSISTENT MEMORY ---"
)

DIRECT_MEMORY_INJECTION_PROMPT = (
    "You are a memory integration agent. Your task is to intelligently integrate the 'NEW FACT' into the "
    "'EXISTING PERSISTENT MEMORY'. If the new fact updates or contradicts existing information, modify the memory "
    "accordingly. If it's a new topic, add it concisely. The goal is to maintain a dense, coherent, and accurate "
    "knowledge base. The final output must be the complete, updated memory, presented as a dense, factual summary. "
    "Eliminate all verbosity and unnecessary formatting and markdown.\n\n"
    "--- EXISTING PERSISTENT MEMORY ---\n{existing_ltm}\n\n"
    "--- NEW FACT TO INTEGRATE ---\n{new_fact}\n\n"
    "--- UPDATED PERSISTENT MEMORY ---"
)

MEMORY_SCRUB_PROMPT = (
    "You are a memory management agent. Your task is to rewrite the 'EXISTING PERSISTENT MEMORY' to completely remove "
    "any information related to the 'TOPIC TO FORGET'. You must preserve all other facts, maintain the original tone "
    "and conciseness, and ensure the rewritten memory is a coherent whole. The final output must be ONLY the rewritten "
    "memory text, with no preamble, conversational text, or markdown formatting.\n\n"
    "--- EXISTING PERSISTENT MEMORY ---\n{existing_ltm}\n\n"
    "--- TOPIC TO FORGET ---\n{topic}\n\n"
    "--- REWRITTEN PERSISTENT MEMORY ---"
)


LOG_RENAMING_PROMPT = (
    "Based on the following chat log, generate a concise, descriptive, filename-safe title. "
    "Use snake_case. The title should be 3-5 words. "
    "Do not include any file extension like '.jsonl'. "
    "Example response: 'python_script_debugging_and_refactoring'\n\n"
    "CHAT LOG EXCERPT:\n---\n{log_content}\n---"
)

SESSION_FINALIZATION_PROMPT = (
    "You are a session finalization agent. Based on the provided chat session and existing memory, perform two tasks:\n"
    "1. **Memory Consolidation**: Distill the crucial information from the 'NEW CHAT SESSION' and integrate it into the 'EXISTING PERSISTENT MEMORY'. "
    "Synthesize related topics, update existing facts, and discard conversational fluff. The final output must be a dense, factual summary.\n"
    "2. **Log Renaming**: Generate a concise, descriptive, filename-safe title for the chat session. Use snake_case. "
    "The title should be 3-5 words. Do not include any file extension like '.jsonl'.\n\n"
    'Your final response MUST be a valid JSON object with two string keys: "updated_memory" and "log_filename".\n\n'
    "--- EXISTING PERSISTENT MEMORY ---\n{existing_ltm}\n\n"
    "--- NEW CHAT SESSION TO INTEGRATE ---\n{session_content}\n\n"
    "--- JSON RESPONSE ---"
)


CONTINUATION_PROMPT = (
    "Please continue the conversation based on the history provided. "
    "Offer a new insight, ask a follow-up question, or rebut the last point made."
)

# Prompts for the image crafting workflow in SessionManager
IMAGE_PROMPT_INITIAL_REFINEMENT = (
    "The user wants to generate an image with this description: '{initial_prompt}'\n\n"
    "Provide a gently refined version that keeps their core idea intact, adds helpful visual "
    "details, and is concise. Respond with only the refined prompt."
)

IMAGE_PROMPT_SUBSEQUENT_REFINEMENT = (
    "Current prompt: '{current_prompt}'\n\n"
    "User refinement: '{user_input}'\n\n"
    "Incorporate the user's feedback into an updated prompt. Respond with only the updated prompt."
)


# --- From handlers.py ---

MULTICHAT_SYSTEM_PROMPT_OPENAI = (
    "You are the OpenAI model. You are not Gemini. The user is the 'Director'.\n"
    "**MANDATORY INSTRUCTION: Your response MUST NOT begin with a label like `[OpenAI]:` or `[Gemini]:`.** "
    "The client application adds your `[OpenAI]` label automatically. Do not duplicate it.\n"
    "Acknowledge and address points made by Gemini, but speak only for yourself."
)

MULTICHAT_SYSTEM_PROMPT_GEMINI = (
    "You are the Gemini model. You are not OpenAI. The user is the 'Director'.\n"
    "**MANDATORY INSTRUCTION: Your response MUST NOT begin with a label like `[Gemini]:` or `[OpenAI]:`.** "
    "The client application adds your `[Gemini]` label automatically. Do not duplicate it.\n"
    "Acknowledge and address points made by OpenAI, but speak only for yourself."
)
