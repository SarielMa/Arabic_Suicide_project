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


def looping_ratio(text: str, window: int = 40, uniq_floor: float = 0.30) -> float:
    """Fraction of overlapping word windows that are mostly one repeated token.

    Detects decoder degeneration: the model latches onto a token and emits it
    until it hits max_new_tokens ("hello hello hello ...", "now, now, now ...").
    The real translation stops wherever the loop began, so the tail of the call
    -- where a helpline caller is most likely to escalate -- is simply gone.

    Length alone cannot catch this. Of the 29 degenerate outputs in the 72B run,
    only 15 exceeded the too_long ratio; the rest looped *within* a normal overall
    length, one at a ratio of 1.05. Hence a shape check rather than a size check.
    """
    words = re.findall(r"\S+", text.lower())
    if len(words) < window:
        return 0.0
    starts = list(range(0, len(words) - window, window // 2))
    if not starts:
        return 0.0
    looped = sum(
        1 for i in starts if len(set(words[i : i + window])) / window < uniq_floor
    )
    return looped / len(starts)


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
    # Decoder degeneration. The source is ASR of a real call and genuinely repeats
    # ("الو الو الو" opens most of them), so a bare repetition count would fire on
    # faithful translations. We flag only when the OUTPUT loops far more than the
    # INPUT does -- that difference is the model's invention, not the caller's speech.
    out_loop, src_loop = looping_ratio(out), looping_ratio(src)
    if out_loop > 0.15 and out_loop > src_loop + 0.15:
        flags.append(f"degenerate({out_loop:.2f}>{src_loop:.2f})")
    if src.count("<PERS>") != out.count("<PERS>"):
        flags.append(f"pers({src.count('<PERS>')}->{out.count('<PERS>')})")
    return flags


# A hard flag means the translation lost content the label depends on: the model
# refused, summarized, truncated, degenerated, or left the text in Arabic. Those
# are worth a retry and must not enter the training data. The rest are cosmetic --
# a dropped <PERS> placeholder or a verbose-but-complete rendering changes no risk
# evidence, and at 72B it is not worth re-decoding a transcript over.
#
# `degenerate` was NOT originally hard, and `too_long` still is not. That gap let
# 29 looped translations (6.6%) into the English data: the loop starts early --
# median 10% of the way in -- so ~90% of those documents is filler and the rest of
# the call was never translated. That is a content-destroying failure, exactly like
# too_short, so it belongs here. `too_long` stays soft: genuine verbosity loses
# nothing, and every looped output is now caught by shape instead.
HARD_FLAGS = ("empty", "refusal", "too_short", "arabic_residue", "degenerate")


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
                    temperature: float, seed: int,
                    repetition_penalty: float = 1.0) -> list[str]:
    """Translate a list of prompts in one vLLM call, preserving input order.

    Attempt 0 is greedy (temperature=0), which reproduces the deterministic
    behaviour of the old transformers path. Retries resample with a little
    temperature -- re-running a greedy decode would only redraw the identical
    bad text.

    ``repetition_penalty`` > 1 is what actually breaks a decoder loop. Temperature
    alone is not enough: once the model is inside "hello hello hello" the repeated
    token dominates the distribution, so sampling re-draws it anyway. Penalising
    already-emitted tokens is what lets the decode escape. Kept mild (~1.1) and
    used only on retries, because the source is disfluent ASR whose real
    repetitions we are instructed to preserve.
    """
    params = SamplingParams(
        temperature=temperature,
        top_p=0.9 if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
        seed=seed if temperature > 0 else None,
        repetition_penalty=repetition_penalty,
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
    p.add_argument("--repetition-penalty", type=float, default=1.1,
                   help="Applied on retries only; >1 is what breaks a decoder loop.")
    p.add_argument("--redo", action="store_true",
                   help="Re-translate transcripts already in --out whose stored "
                        "translation fails the CURRENT hard checks, rewriting the "
                        "file in place (a .bak is kept). Use after tightening the "
                        "checks; without it, existing entries are never revisited.")
    p.add_argument("--dry-run", action="store_true",
                   help="With --redo: report what would be re-translated, load no model.")
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

    # --redo re-checks what is already on disk against the CURRENT rules and queues
    # the failures. Ordinary runs only ever translate file_ids absent from the file,
    # so a check tightened after the fact would otherwise never be applied.
    redo_ids: list[str] = []
    if args.redo:
        for fid, rec in done.items():
            if fid not in transcripts:
                continue
            if hard_flags(check_translation(transcripts[fid], rec.get("english", ""))):
                redo_ids.append(fid)
        redo_ids.sort()
        print(f"--redo: {len(redo_ids)}/{len(done)} stored translations fail the current checks")
        for fid in redo_ids:
            fl = check_translation(transcripts[fid], done[fid].get("english", ""))
            print(f"   {fid}  {hard_flags(fl)}")
        if args.dry_run:
            print("\n--dry-run: nothing re-translated, no model loaded.")
            return 0
        if not redo_ids:
            print("Nothing to redo.")
            return 0

    todo = [(fid, txt) for fid, txt in transcripts.items() if fid not in done]
    if args.limit:
        todo = todo[: args.limit]
    todo = [(fid, transcripts[fid]) for fid in redo_ids] + todo
    print(f"Already translated: {len(done)}   To do now: {len(todo)}"
          f"{f' ({len(redo_ids)} redo + {len(todo) - len(redo_ids)} new)' if redo_ids else ''}")
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

    redo_set = set(redo_ids)
    updated: dict[str, dict] = {}  # redone records, merged back into the file at the end

    n_flagged = 0
    n_written = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start : start + args.chunk_size]
            prompts = [build_prompt(tokenizer, src) for _, src in chunk]

            # Attempt 0: one greedy pass over the whole chunk. On --redo the stored
            # text came from exactly this decode, so repeating it would reproduce the
            # same loop; go straight to the sampled+penalised path instead.
            if args.redo and all(fid in redo_set for fid, _ in chunk):
                # Offset well clear of the retry seeds (seed+1, seed+2, ...) below,
                # so a retry never redraws the sample this pass already produced.
                english = translate_batch(
                    llm, prompts, args.max_new_tokens,
                    temperature=0.3, seed=args.seed + 1000,
                    repetition_penalty=args.repetition_penalty,
                )
            else:
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
                    repetition_penalty=args.repetition_penalty,
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
                if fid in redo_set:
                    # Held back and merged in below: appending would leave two records
                    # for one file_id, and consumers that keep the first would silently
                    # go on using the broken text.
                    updated[fid] = rec
                else:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                status = "OK" if not fl else f"FLAGGED {fl}"
                print(f"[{n_written}/{len(todo)}] {fid}  "
                      f"{len(src)}->{len(en)} chars  {status}")
            handle.flush()  # resumable even if the job is killed between chunks

    # Merge redone records back in place, preserving the original line order so the
    # file stays a stable artifact. The previous version is kept as .bak.
    if updated:
        backup = args.out.with_suffix(args.out.suffix + ".bak")
        args.out.replace(backup)
        seen: set[str] = set()
        n_replaced = 0
        with backup.open(encoding="utf-8") as src_h, \
             args.out.open("w", encoding="utf-8") as dst_h:
            for line in src_h:
                if not line.strip():
                    continue
                rec = json.loads(line)
                fid = rec["file_id"]
                if fid in seen:
                    continue  # collapse any pre-existing duplicates
                seen.add(fid)
                if fid in updated:
                    rec = updated.pop(fid)
                    n_replaced += 1
                dst_h.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for fid, rec in updated.items():  # redone but not previously on file
                dst_h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nReplaced {n_replaced} record(s) in place; previous file kept at {backup}")

    print(f"\n=== Done. {len(todo) - n_flagged}/{len(todo)} clean, {n_flagged} flagged.")
    print(f"Translations: {args.out}")
    print("Inspect flagged ones before building the English datasets:")
    print(f"  python inspect_translations.py --pred {args.out}")
    if redo_set:
        print("\nThen rebuild the downstream datasets:")
        print(f"  python build_english_data.py --pred {args.out}")
        print("  python build_merged_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
