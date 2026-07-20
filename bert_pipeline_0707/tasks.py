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

# Coarse two-level grouping of the 5 tasks, built by the LLM pipeline's
# build_merged_data.py (which writes processed_datasets_merged{,_en}/). Each is
# the OR of its constituents:
#     med_risk  = wish_to_be_dead OR non_specific_active_suicidal_thoughts
#     high_risk = any_methods OR some_intent_to_act OR specific_plan_and_intent
# Same text -> {0,1} shape as the 5 base tasks, so the pipeline treats them
# identically; only the dataset directory differs.
MERGED_TASK_KEYS = [
    "med_risk",
    "high_risk",
]

# Everything --task will accept. Kept separate from TASK_KEYS so callers that
# mean "the 5 C-SSRS tasks" still get exactly those.
ALL_TASK_KEYS = TASK_KEYS + MERGED_TASK_KEYS
