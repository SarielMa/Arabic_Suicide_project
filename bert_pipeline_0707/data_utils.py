"""Shared data loading / text preprocessing for the BERT pipeline."""

from __future__ import annotations

import json
from pathlib import Path


def load_split(data_dir: Path, task: str, split: str) -> tuple[list[str], list[int], list[str]]:
    """Load a task split. Returns (texts, labels, file_ids).

    Supports the raw JSON datasets produced by build_training_datasets.py and
    the instruction JSONL datasets produced by the LLM pipeline.
    """
    json_path = data_dir / task / f"{split}.json"
    jsonl_path = data_dir / task / f"{split}.jsonl"

    if json_path.exists():
        with json_path.open(encoding="utf-8") as handle:
            records = json.load(handle)
    elif jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        raise FileNotFoundError(f"Missing split file: {json_path} or {jsonl_path}")

    texts = [r.get("text", r.get("input", "")) for r in records]
    labels = [int(r["label"]) for r in records]
    file_ids = [r["file_id"] for r in records]
    return texts, labels, file_ids


def maybe_arabert_preprocessor(model_name: str):
    """Return an AraBERT preprocessor for AraBERT models, else None.

    AraBERT models expect their own text segmentation. If the model looks like
    an AraBERT checkpoint and the `arabert` package is installed, we apply it;
    otherwise we skip it (with a warning) so the pipeline still runs.
    """
    if "arabert" not in model_name.lower():
        return None
    try:
        from arabert.preprocess import ArabertPreprocessor

        print(f"Applying AraBERT preprocessing for {model_name}")
        return ArabertPreprocessor(model_name=model_name)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: '{model_name}' looks like AraBERT but the 'arabert' "
            f"package is unavailable ({type(exc).__name__}); continuing without "
            f"AraBERT preprocessing. `pip install arabert` to enable it."
        )
        return None


def preprocess_texts(texts: list[str], preprocessor) -> list[str]:
    if preprocessor is None:
        return texts
    return [preprocessor.preprocess(t) for t in texts]
