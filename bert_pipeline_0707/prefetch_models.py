#!/usr/bin/env python3
"""Pre-download every model in models.txt into the HF cache.

Run this ON THE LOGIN NODE (which has internet) before submitting the SLURM
job, so the compute node can load models from ~/.cache/huggingface offline.

Uses snapshot_download, which fetches the repo files into the cache without
loading the weights into memory (so it is safe for large models too).

Usage::

    python prefetch_models.py                 # uses models.txt
    MODELS_FILE=other.txt python prefetch_models.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

# Skip alternative weight formats we do not use, to save bandwidth/disk.
IGNORE = ["*.gguf", "*.onnx", "*.onnx_data", "*.msgpack", "*.h5"]


def read_models(models_file: Path) -> list[str]:
    models = []
    for line in models_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            models.append(line)
    return models


def main() -> int:
    models_file = Path(os.environ.get("MODELS_FILE", "models.txt"))
    if not models_file.exists():
        raise SystemExit(f"Missing model list: {models_file}")

    models = read_models(models_file)
    print(f"Prefetching {len(models)} model(s) from {models_file} into the HF cache:")
    for name in models:
        print(f"  {name} ...", flush=True)
        path = snapshot_download(repo_id=name, ignore_patterns=IGNORE)
        print(f"  {name}  OK -> {path}")

    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    print(f"\nDone. Cached under: {cache}")
    print("The SLURM job can now load these models offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
