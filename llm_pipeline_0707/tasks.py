"""Shared task definitions and prompt formatting for the 5 binary tasks.

Kept model-agnostic: the instruction/input/output triples produced here can be
rendered into any model's chat template at training/eval time.
"""

from __future__ import annotations

# System prompt used for every task.
SYSTEM_PROMPT = (
    "You are a clinical assistant that analyzes Arabic crisis-helpline call "
    "transcripts. Read the transcript and answer the yes/no question. "
    "Reply with exactly one word: Yes or No."
)

# Positive/negative surface forms (label 1 -> Yes, label 0 -> No).
POSITIVE = "Yes"
NEGATIVE = "No"

# task_key (matches the dataset folder name) -> metadata.
# `question` is the per-task instruction; `column` is the source metadata label.
TASKS: dict[str, dict[str, str]] = {
    "wish_to_be_dead": {
        "column": "Wish To Be Dead",
        "question": "Based on the call, does the caller express a wish to be dead "
        "(a passive desire to die)?",
    },
    "non_specific_active_suicidal_thoughts": {
        "column": "Non Specific Active Suicidal Thoughts",
        "question": "Based on the call, does the caller express non-specific active "
        "suicidal thoughts (thoughts of killing oneself without a method)?",
    },
    "active_suicidal_ideation_with_any_methods": {
        "column": "Active Suicidal Ideation With Any Methods",
        "question": "Based on the call, does the caller express active suicidal "
        "ideation with any methods (has thought of how, but no plan or intent)?",
    },
    "active_suicidal_with_some_intent_to_act": {
        "column": "Active Suicidal With Some Intent To Act",
        "question": "Based on the call, does the caller express active suicidal "
        "ideation with some intent to act?",
    },
    "active_suicidal_ideation_with_specific_plan_and_intent": {
        "column": "Active Suicidal Ideation With Specific Plan And Intent",
        "question": "Based on the call, does the caller express active suicidal "
        "ideation with a specific plan and intent?",
    },
}


def label_to_text(label: int) -> str:
    return POSITIVE if int(label) == 1 else NEGATIVE


def text_to_label(text: str) -> int | None:
    """Parse a model answer into a binary label, or None if unparseable."""
    t = text.strip().lower()
    # Look at the first informative token.
    for token in t.replace(".", " ").replace(",", " ").split():
        if token in {"yes", "true", "1", "positive"}:
            return 1
        if token in {"no", "false", "0", "negative"}:
            return 0
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def build_instruction(question: str) -> str:
    return f"{question}\nAnswer with exactly one word: Yes or No."


def messages_from_instruction(instruction: str, transcript: str) -> list[dict[str, str]]:
    """Build chat messages (without the assistant answer) from an example.

    ``instruction`` is the already-built instruction stored in the processed
    dataset; ``transcript`` is the ``input`` field. Used by train.py / evaluate.py
    so all stages format prompts identically.
    """
    user = f"{instruction}\n\nCall transcript:\n{transcript}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_messages(question: str, transcript: str) -> list[dict[str, str]]:
    """Build chat messages (without the assistant answer) from a raw question."""
    return messages_from_instruction(build_instruction(question), transcript)
