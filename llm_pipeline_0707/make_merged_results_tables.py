#!/usr/bin/env python3
"""Generate LaTeX result tables for merged LLM and BERT runs."""

from __future__ import annotations

import csv
from pathlib import Path


METRICS = (
    ("macro_f1", "Macro $F_1$"),
    ("recall_pos", "$R_{+}$"),
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("accuracy", "Acc"),
)
TASKS = (
    ("med_risk", "MR"),
    ("high_risk", "HR"),
)

LLM_MODELS = (
    ("qwen2.5-1.5b-instruct", "Qwen2.5-1.5B"),
    ("qwen2.5-14b-instruct", "Qwen2.5-14B"),
    ("llama-3.3-70b-instruct", "Llama-3.3-70B"),
    ("acegpt-v2-8b-chat", "AceGPT-v2-8B"),
    ("acegpt-v2-70b-chat", "AceGPT-v2-70B"),
)

AR_BERT_MODELS = (
    ("bert-base-arabic-camelbert-da", "CAMeLBERT-DA"),
    ("bert-base-arabertv02", "AraBERTv0.2"),
    ("marbert-sakenni-sentiment", "MARBERT-Sakenni"),
    ("bert-base-multilingual-cased", "mBERT"),
)

EN_BERT_MODELS = (
    ("bert-large-uncased", "BERT-large"),
    ("bert-base-uncased", "BERT-base"),
)

TABLE_SPECS = (
    ("llm", Path("runs_merged_ep3"), "Arabic", 3, "llm-results-merged-ar-ep3"),
    ("llm", Path("runs_en_merged_ep3"), "English", 3, "llm-results-merged-en-ep3"),
    ("llm", Path("runs_merged_ep10"), "Arabic", 10, "llm-results-merged-ar-ep10"),
    ("llm", Path("runs_en_merged_ep10"), "English", 10, "llm-results-merged-en-ep10"),
    (
        "bert",
        Path("../bert_pipeline_0707/runs_merged_ep3"),
        "Arabic",
        3,
        "bert-results-merged-ar-ep3",
    ),
    (
        "bert",
        Path("../bert_pipeline_0707/runs_en_merged_ep3"),
        "English",
        3,
        "bert-results-merged-en-ep3",
    ),
    (
        "bert",
        Path("../bert_pipeline_0707/runs_merged_ep10"),
        "Arabic",
        10,
        "bert-results-merged-ar-ep10",
    ),
    (
        "bert",
        Path("../bert_pipeline_0707/runs_en_merged_ep10"),
        "English",
        10,
        "bert-results-merged-en-ep10",
    ),
)


def read_summary(path: Path) -> dict[str, dict[str, float | int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, float | int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            if task in rows:
                raise ValueError(f"duplicate task {task!r} in {path}")
            rows[task] = {
                metric: float(row[metric])
                for metric, _ in METRICS
            } | {
                "support_pos": int(row["support_pos"]),
                "support_neg": int(row["support_neg"]),
            }
    missing = [task for task, _ in TASKS if task not in rows]
    if missing:
        raise ValueError(f"missing task(s) {missing} in {path}")
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def prevalence(row: dict[str, float | int]) -> int:
    support_pos = int(row["support_pos"])
    support_neg = int(row["support_neg"])
    return round(100.0 * support_pos / (support_pos + support_neg))


def maybe_bold(value: str, should_bold: bool) -> str:
    return rf"\textbf{{{value}}}" if should_bold else value


def llm_table(root: Path, language: str, epoch: int, label: str) -> str:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Macro $F_1$, positive-class recall ($R_{+}$), PR-AUC, ROC-AUC,",
        r"and overall accuracy (Acc), on the held-out test split, in percent. Positive prevalence",
        r"per task is given in parentheses. Bold marks SFT cells where",
        rf"SFT strictly improves on that model's own zero-shot condition. This is on the {language} merged dataset. Epoch is {epoch}.}}",
        rf"\label{{tab:{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\small",
        r"\begin{tabular}{l ccccc c ccccc}",
        r"\toprule",
        r" & \multicolumn{5}{c}{Zero-shot} & & \multicolumn{5}{c}{SFT} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){8-12}",
        r"Task (\% pos)",
        rf" & {' & '.join(label for _, label in METRICS)} & & {' & '.join(label for _, label in METRICS)} \\",
        r"\midrule",
    ]

    for idx, (run_name, display_name) in enumerate(LLM_MODELS):
        zero = read_summary(root / "zeroshot" / run_name / "summary.csv")
        sft = read_summary(root / run_name / "summary.csv")
        if idx:
            lines.extend([r"\midrule", r"\midrule"])
        lines.extend([
            rf"\multicolumn{{12}}{{c}}{{\textbf{{{display_name}}}}} \\",
            r"\midrule",
        ])
        for task, short_name in TASKS:
            z = zero[task]
            s = sft[task]
            task_name = rf"{short_name} \ ({prevalence(s)}\%)"
            z_vals = [pct(float(z[metric])) for metric, _ in METRICS]
            s_vals = []
            for metric, _ in METRICS:
                value = pct(float(s[metric]))
                should_bold = float(s[metric]) > float(z[metric])
                s_vals.append(maybe_bold(value, should_bold))
            lines.append(
                rf"{task_name} & {' & '.join(z_vals)} & & {' & '.join(s_vals)} \\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def bert_models_for(language: str) -> tuple[tuple[str, str], ...]:
    return EN_BERT_MODELS if language == "English" else AR_BERT_MODELS


def bert_table(root: Path, language: str, epoch: int, label: str) -> str:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\caption{Macro $F_1$, positive-class recall ($R_{+}$), PR-AUC, ROC-AUC,",
        r"and overall accuracy (Acc), on the held-out test split, in percent. Positive prevalence",
        rf"per task is given in parentheses. This is on the {language} merged dataset. Epoch is {epoch}.}}",
        rf"\label{{tab:{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\small",
        r"\begin{tabular}{l ccccc}",
        r"\toprule",
        rf"Task (\% pos) & {' & '.join(label for _, label in METRICS)} \\",
        r"\midrule",
    ]

    for idx, (run_name, display_name) in enumerate(bert_models_for(language)):
        rows = read_summary(root / run_name / "summary.csv")
        if idx:
            lines.extend([r"\midrule", r"\midrule"])
        lines.extend([
            rf"\multicolumn{{6}}{{c}}{{\textbf{{{display_name}}}}} \\",
            r"\midrule",
        ])
        for task, short_name in TASKS:
            row = rows[task]
            task_name = rf"{short_name} \ ({prevalence(row)}\%)"
            vals = [pct(float(row[metric])) for metric, _ in METRICS]
            lines.append(rf"{task_name} & {' & '.join(vals)} \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def main() -> int:
    tables = []
    for kind, root, language, epoch, label in TABLE_SPECS:
        if kind == "llm":
            tables.append(llm_table(root, language, epoch, label))
        else:
            tables.append(bert_table(root, language, epoch, label))

    output = Path("merged_results_tables.tex")
    output.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"Wrote {len(tables)} tables to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
