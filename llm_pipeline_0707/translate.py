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

Resumable: already-translated file_ids are skipped, so a timed-out job can be
resubmitted and will pick up where it left off.

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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

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
    if REFUSAL_PATTERNS.search(out[:400]):
        flags.append("refusal")
    ratio = len(out) / max(len(src), 1)
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
def translate_one(model, tokenizer, transcript: str, max_new_tokens: int,
                  temperature: float) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(transcript=transcript)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.9 if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


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
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    use_4bit = not args.no_4bit and use_cuda

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    model.eval()

    n_flagged = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for i, (fid, src) in enumerate(todo, 1):
            english, flags = "", ["empty"]
            for attempt in range(args.retries + 1):
                # Retry greedily first; if that output is bad, resample with a little
                # temperature rather than re-drawing the identical deterministic text.
                english = translate_one(
                    model, tokenizer, src,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0 if attempt == 0 else 0.3,
                )
                flags = check_translation(src, english)
                if not flags:
                    break
                print(f"  [{fid}] attempt {attempt + 1} flagged: {flags}")

            if flags:
                n_flagged += 1
            rec = {
                "file_id": fid,
                "arabic": src,
                "english": english,
                "flags": flags,
                "len_ratio": round(len(english) / max(len(src), 1), 3),
                "arabic_ratio": round(arabic_ratio(english), 3),
                "pers_src": src.count("<PERS>"),
                "pers_out": english.count("<PERS>"),
                "model": args.model,
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()  # resumable even if the job is killed mid-sweep
            status = "OK" if not flags else f"FLAGGED {flags}"
            print(f"[{i}/{len(todo)}] {fid}  {len(src)}->{len(english)} chars  {status}")

    print(f"\n=== Done. {len(todo) - n_flagged}/{len(todo)} clean, {n_flagged} flagged.")
    print(f"Translations: {args.out}")
    print("Inspect flagged ones before building the English datasets:")
    print(f"  python inspect_translations.py --pred {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
