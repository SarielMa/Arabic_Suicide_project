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


# Coarser two-level grouping over the 5 C-SSRS tasks, produced by
# build_merged_data.py. A merged label is the OR of its constituents.
#
# The constituent tasks do not all annotate the same calls, so a call whose
# available constituent labels are all 0 but which is missing at least one
# annotation has an *undetermined* merged label (0 or 1 depending on the value
# that was never recorded). build_merged_data.py drops those rather than
# assuming a negative; see its docstring.
MERGED_TASKS: dict[str, dict] = {
    "med_risk": {
        "components": [
            "wish_to_be_dead",
            "non_specific_active_suicidal_thoughts",
        ],
        "question": "Based on the call, does the caller express suicidal ideation at "
        "a moderate level, that is, either a wish to be dead (a passive desire to "
        "die) or non-specific active suicidal thoughts (thoughts of killing oneself "
        "without a method)?",
    },
    "high_risk": {
        "components": [
            "active_suicidal_ideation_with_any_methods",
            "active_suicidal_with_some_intent_to_act",
            "active_suicidal_ideation_with_specific_plan_and_intent",
        ],
        "question": "Based on the call, does the caller express suicidal ideation at "
        "a high level, that is, active suicidal ideation with any method, with some "
        "intent to act, or with a specific plan and intent?",
    },
}


# AceGPT-v2 (Llama-3-based, FreedomIntelligence) ships no chat_template but was
# trained on a plain "<User>: ... <Assistant>: ..." format with no system role
# (see its model card). We fold the system prompt into the first user turn and
# terminate each assistant turn with eos_token. Rendered form:
#
#   <User>: {system}\n\n{user} <Assistant>: {answer}{eos}
#
# NOTE: the generation prompt ends at "<Assistant>:" with NO trailing space; the
# space that separates the marker from the answer is emitted as part of the
# assistant turn instead. This keeps the tokenized prompt an exact *token-level*
# prefix of the full sequence under Llama-3 BPE (a trailing prompt space would
# otherwise merge with the first answer token, corrupting train.py's label mask).
ACEGPT_CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}"
    "{{ '<User>: ' + message['content'] + '\n\n' }}"
    "{%- elif message['role'] == 'user' -%}"
    "{%- if loop.first -%}{{ '<User>: ' }}{%- endif -%}"
    "{{ message['content'] }}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{ ' <Assistant>: ' + message['content'] + eos_token }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ ' <Assistant>:' }}{%- endif -%}"
)


def set_chat_template_if_missing(tokenizer) -> None:
    """Ensure the tokenizer can render chat messages.

    Some fine-tunes (e.g. AceGPT-v2) do not ship a ``chat_template``, which makes
    ``apply_chat_template`` raise. When one is missing we install a template that
    matches the model's documented prompt format so train.py / evaluate.py format
    prompts identically. Models that already carry a template are left untouched,
    so this is a no-op for Qwen/Llama-Instruct and friends.
    """
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = ACEGPT_CHAT_TEMPLATE


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


def answer_first_token_ids(tokenizer) -> tuple[int, int]:
    """Return the (yes_id, no_id) token ids the model emits *first* in its answer.

    Not simply ``encode("Yes")``: under AceGPT's template the assistant turn opens
    with a leading space, so the first answer token is " Yes", a different id than
    "Yes". We therefore render a dummy example through the chat template both with
    and without the answer, and read off the first token past the generation
    prompt---the same construction train.py uses to place its label mask, so the
    id we score is exactly the id that was trained.
    """
    messages = messages_from_instruction("dummy question", "dummy transcript")
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    ids = {}
    for label, word in ((1, POSITIVE), (0, NEGATIVE)):
        full_ids = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": word}],
            add_generation_prompt=False,
            tokenize=True,
        )
        if full_ids[: len(prompt_ids)] != list(prompt_ids):
            raise ValueError(
                "Generation prompt is not a token-level prefix of the full sequence "
                f"for answer {word!r}; first-token scoring would be misaligned."
            )
        ids[label] = full_ids[len(prompt_ids)]
    if ids[1] == ids[0]:
        raise ValueError("Yes and No map to the same first token; cannot score.")
    return ids[1], ids[0]


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
