"""Task list shared by the BERT pipeline.

The 5 binary tasks correspond to the dataset folder names produced by
``build_training_datasets.py``. BERT needs no prompts — each task is a plain
text -> {0,1} sequence-classification problem.
"""

from __future__ import annotations

TASK_KEYS = [
    "wish_to_be_dead",
    "non_specific_active_suicidal_thoughts",
    "active_suicidal_ideation_with_any_methods",
    "active_suicidal_with_some_intent_to_act",
    "active_suicidal_ideation_with_specific_plan_and_intent",
]
