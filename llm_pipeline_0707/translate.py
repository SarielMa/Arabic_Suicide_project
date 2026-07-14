#!/usr/bin/env python3
"""Translate the Arabic call transcripts into English with an instruct LLM.

The five tasks are five label sets over the SAME calls, so we translate the 438
*unique* transcripts (keyed by file_id) rather than the 1766 task rows. Besides
being 4x cheaper, this guarantees a given call reads identically across tasks --
translating it once per task would yield five different Englishes for one call.

The hard part is not the translation, it is *not silently losing the label*. The
cues that separate IM/SI/PI are exactly the graphic specifics (method, plan,
intent) that a safety-tuned chat model is most likely to soften, summarize or
refuse. A sanitized translation would delete the positive-class signal
*selectively*, and the resulting English scores would be uninterpretable. So every
translation is checked (see check_translation) and anything suspicious is retried
and then flagged; nothing is quietly accepted.

Runs on vLLM. The transcripts are independent, so the whole corpus is handed to
the engine at once and continuous batching keeps the GPUs saturated; the previous
transformers path decoded one transcript at a time and left both B200s mostly idle.
Weights are served in bf16 rather than 4-bit NF4: Qwen2.5-72B is ~145GB and two
B200s hold 360GB, so the quantization was buying memory we already had while
paying a dequantize cost on every forward pass.

Resumable: already-translated file_ids are skipped, and results are appended after
each chunk, so a timed-out job can be resubmitted and will pick up where it left off.

Example::

    python translate.py --model Qwen/Qwen2.5-72B-Instruct \\
        --out translations/qwen2.5-72b-instruct.jsonl --limit 20   # pilot
    python translate.py --model Qwen/Qwen2.5-72B-Instruct \\
        --out translations/qwen2.5-72b-instruct.jsonl              # all 438
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "You are a professional Arabic-to-English translator working on clinical "
    "crisis-helpline transcripts for a research study approved by an ethics board. "
    "The audio is colloquial spoken Arabic (Levantine dialect) produced by "
    "automatic speech recognition, so it is disfluent, unpunctuated and repetitive."
)

# The instruction is deliberately blunt about faithfulness. A softened or
# summarized translation is worse than useless here: it removes exactly the
# risk-bearing content the downstream classifier is trained to detect.
USER_TEMPLATE = """Translate the following Arabic crisis-helpline transcript into English.

Rules:
1. Translate FAITHFULLY and COMPLETELY. This is clinical data: do NOT summarize, \
soften, censor, omit or add anything. Statements about suicide, self-harm, methods, \
plans or intent MUST be translated literally and in full detail -- they are the \
clinical content of the record, and omitting them would corrupt the study.
2. Preserve every <PERS> placeholder exactly as written.
3. Preserve the turn/line structure: one line in, one line out.
4. Keep the register colloquial and the disfluencies (repetitions, false starts). \
Do not clean up the speech.
5. Output ONLY the English translation. No preamble, no commentary, no notes.

Transcript:
{transcript}"""

REFUSAL_PATTERNS = re.compile(
    r"\b(i'?m sorry|i cannot|i can'?t (help|assist|provide)|as an ai|i am unable"
    r"|cannot fulfill|i must decline|it is not appropriate|if you are in crisis"
    r"|please (seek|contact|reach out to) )",
    re.IGNORECASE,
)


def arabic_ratio(text: str) -> float:
    """Fraction of letters that are still Arabic script (catches partial output)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if "ARABIC" in unicodedata.name(c, ""))
    return arabic / len(letters)


def check_translation(src: str, out: str) -> list[str]:
    """Return a list of quality flags; empty means the translation looks sound."""
    flags = []
    if not out.strip():
        flags.append("empty")
        return flags
    ratio = len(out) / max(len(src), 1)
    # A refusal is short by definition -- the model declines instead of translating.
    # The phrases themselves ("I'm sorry", "if you are in crisis", "please reach out
    # to") are also what distressed callers and helpline staff actually say, so on a
    # full-length output they are transcript content, not a refusal. Requiring a
    # short output as well keeps this from firing on faithful translations of exactly
    # the most emotional calls -- which would strip the positive class.
    if REFUSAL_PATTERNS.search(out[:400]) and ratio < 0.9:
        flags.append("refusal")
    # English of Arabic runs roughly 0.8-1.6x the source length. Far below that is
    # summarization or truncation -- i.e. exactly the label-destroying failure.
    if ratio < 0.55:
        flags.append(f"too_short({ratio:.2f})")
    elif ratio > 2.5:
        flags.append(f"too_long({ratio:.2f})")
    ar = arabic_ratio(out)
    if ar > 0.10:
        flags.append(f"arabic_residue({ar:.2f})")
    if src.count("<PERS>") != out.count("<PERS>"):
        flags.append(f"pers({src.count('<PERS>')}->{out.count('<PERS>')})")
    return flags


# A hard flag means the translation lost content the label depends on: the model
# refused, summarized, truncated, or left the text in Arabic. Those are worth a
# retry and must not enter the training data. The rest are cosmetic -- a dropped
# <PERS> placeholder or a verbose-but-complete rendering changes no risk evidence,
# and at 72B it is not worth re-decoding a transcript over.
HARD_FLAGS = ("empty", "refusal", "too_short", "arabic_residue")


def hard_flags(flags: list[str]) -> list[str]:
    return [f for f in flags if f.split("(")[0] in HARD_FLAGS]


def load_unique_transcripts(data_dir: Path) -> dict[str, str]:
    """file_id -> Arabic transcript, deduplicated across the five task datasets."""
    transcripts: dict[str, str] = {}
    for task_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for split in ("train", "test"):
            path = task_dir / f"{split}.jsonl"
            if not path.exists():
                continue
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                transcripts.setdefault(rec["file_id"], rec["input"])
    return transcripts


def read_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            done[rec["file_id"]] = rec
    return done


@torch.no_grad()
def build_prompt(tokenizer, transcript: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript)},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def translate_batch(llm, prompts: list[str], max_new_tokens: int,
                    temperature: float, seed: int) -> list[str]:
    """Translate a list of prompts in one vLLM call, preserving input order.

    Attempt 0 is greedy (temperature=0), which reproduces the deterministic
    behaviour of the old transformers path. Retries resample with a little
    temperature -- re-running a greedy decode would only redraw the identical
    bad text.
    """
    params = SamplingParams(
        temperature=temperature,
        top_p=0.9 if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
        seed=seed if temperature > 0 else None,
    )
    outputs = llm.generate(prompts, params)
    # vLLM returns results in the order of the prompts it was given.
    return [o.outputs[0].text.strip() for o in outputs]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    p.add_argument("--data-dir", type=Path, default=Path("processed_datasets"))
    p.add_argument("--out", type=Path, required=True,
                   help="JSONL of translations; appended to, and resumed from.")
    p.add_argument("--limit", type=int, default=0,
                   help="Translate only the first N transcripts (0 = all). Use for a pilot.")
    p.add_argument("--max-new-tokens", type=int, default=3072)
    p.add_argument("--retries", type=int, default=2,
                   help="Re-attempts for a transcript whose translation fails the checks.")
    p.add_argument("--seed", type=int, default=42)
    # --- vLLM engine ---
    p.add_argument("--tensor-parallel-size", type=int, default=0,
                   help="0 = use every visible GPU.")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="Longest transcript is ~3.7k tokens; +3k generated fits in 8k.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--chunk-size", type=int, default=64,
                   help="Transcripts per write. Smaller = more resumable, no speed cost "
                        "beyond a brief drain of the batch between chunks.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    transcripts = load_unique_transcripts(args.data_dir)
    print(f"Unique transcripts across all tasks: {len(transcripts)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(args.out)
    todo = [(fid, txt) for fid, txt in transcripts.items() if fid not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Already translated: {len(done)}   To do now: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tp_size = args.tensor_parallel_size or max(torch.cuda.device_count(), 1)
    print(f"vLLM: tensor_parallel_size={tp_size} dtype=bfloat16 "
          f"max_model_len={args.max_model_len}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )

    n_flagged = 0
    n_written = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start : start + args.chunk_size]
            prompts = [build_prompt(tokenizer, src) for _, src in chunk]

            # Attempt 0: one greedy pass over the whole chunk.
            english = translate_batch(
                llm, prompts, args.max_new_tokens, temperature=0.0, seed=args.seed
            )
            flags = [check_translation(src, en) for (_, src), en in zip(chunk, english)]

            # Retries: re-decode only the transcripts whose translation lost content,
            # again as one batch per round rather than one call per transcript.
            for attempt in range(1, args.retries + 1):
                bad = [i for i, f in enumerate(flags) if hard_flags(f)]
                if not bad:
                    break
                for i in bad:
                    print(f"  [{chunk[i][0]}] attempt {attempt} flagged: {flags[i]}")
                retry_out = translate_batch(
                    llm, [prompts[i] for i in bad], args.max_new_tokens,
                    temperature=0.3, seed=args.seed + attempt,
                )
                for i, en in zip(bad, retry_out):
                    english[i] = en
                    flags[i] = check_translation(chunk[i][1], en)

            for (fid, src), en, fl in zip(chunk, english, flags):
                if fl:
                    n_flagged += 1
                n_written += 1
                rec = {
                    "file_id": fid,
                    "arabic": src,
                    "english": en,
                    "flags": fl,
                    "len_ratio": round(len(en) / max(len(src), 1), 3),
                    "arabic_ratio": round(arabic_ratio(en), 3),
                    "pers_src": src.count("<PERS>"),
                    "pers_out": en.count("<PERS>"),
                    "model": args.model,
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                status = "OK" if not fl else f"FLAGGED {fl}"
                print(f"[{n_written}/{len(todo)}] {fid}  "
                      f"{len(src)}->{len(en)} chars  {status}")
            handle.flush()  # resumable even if the job is killed between chunks

    print(f"\n=== Done. {len(todo) - n_flagged}/{len(todo)} clean, {n_flagged} flagged.")
    print(f"Translations: {args.out}")
    print("Inspect flagged ones before building the English datasets:")
    print(f"  python inspect_translations.py --pred {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
