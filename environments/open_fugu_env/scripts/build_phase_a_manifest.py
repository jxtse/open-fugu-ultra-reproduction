"""Build the frozen Phase A manifest from DAPO-Math-17k.

Steps:
1. Load DAPO-Math-17k, dedupe by normalized question hash.
2. Contamination scan against eval denylist sources we can check offline:
   - GPQA (all splits are held out; questions are science MCQ so overlap with
     DAPO math is expected ~0, but we verify by 13-gram overlap)
   - Note: LCB/SWE/Terminal-Bench are code/agentic benchmarks with no
     plausible overlap with pure-math DAPO; recorded as N/A in the manifest.
3. Sample PHASE_A_SIZE tasks with a fixed seed, write manifest JSONL with
   sha256, question, answer, source, and filter provenance.

Run on a800 from the prime-rl repo (uses its venv's datasets).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

PHASE_A_SIZE = 500
SEED = 17
OUT_DIR = Path("/vePFS-Mindverse/share/richard/open-fugu-ultra/data/phase_a")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ngrams(text: str, n: int = 13) -> set[str]:
    words = normalize(text).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def main() -> None:
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading DAPO-Math-17k ...", flush=True)
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    raw_rows = len(ds)

    seen: set[str] = set()
    unique: list[dict] = []
    for row in ds:
        question = row["prompt"][-1]["content"]
        answer = str(row["reward_model"]["ground_truth"]).strip()
        if not answer:
            continue
        key = hashlib.sha256(normalize(question).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append({"sha256": key, "question": question, "answer": answer})
    print(f"raw rows: {raw_rows}, unique questions: {len(unique)}", flush=True)

    # Contamination scan: GPQA (main covers extended+diamond questions? diamond
    # is a subset of main; we scan all three configs to be safe).
    print("loading GPQA for denylist scan ...", flush=True)
    gpqa_ngrams: set[str] = set()
    gpqa_counts = {}
    for config in ("gpqa_main", "gpqa_extended", "gpqa_diamond"):
        try:
            gp = load_dataset("Idavidrein/gpqa", config, split="train")
            gpqa_counts[config] = len(gp)
            for row in gp:
                gpqa_ngrams |= ngrams(row["Question"])
        except Exception as exc:  # gated dataset fallback
            gpqa_counts[config] = f"UNAVAILABLE: {exc}"
    print("gpqa splits:", gpqa_counts, flush=True)

    flagged = []
    clean = []
    for item in unique:
        item_ngrams = ngrams(item["question"])
        if gpqa_ngrams and item_ngrams & gpqa_ngrams:
            flagged.append(item)
        else:
            clean.append(item)
    print(f"flagged by GPQA 13-gram overlap: {len(flagged)}", flush=True)

    rng = random.Random(SEED)
    rng.shuffle(clean)
    manifest = clean[:PHASE_A_SIZE]

    manifest_path = OUT_DIR / "phase_a_manifest.jsonl"
    with manifest_path.open("w") as fh:
        for i, item in enumerate(manifest):
            fh.write(
                json.dumps(
                    {
                        "idx": i,
                        "sha256": item["sha256"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "source": "BytedTsinghua-SIA/DAPO-Math-17k",
                        "filters": ["dedup_sha256", "gpqa_13gram"],
                        "seed": SEED,
                    }
                )
                + "\n"
            )

    summary = {
        "raw_rows": raw_rows,
        "unique_questions": len(unique),
        "gpqa_splits": {k: v for k, v in gpqa_counts.items()},
        "gpqa_flagged": len(flagged),
        "clean_pool": len(clean),
        "phase_a_size": len(manifest),
        "seed": SEED,
        "denylist_na": {
            "livecodebench": "no overlap surface: DAPO is pure math, LCB is code",
            "swe_bench_pro": "no overlap surface: no repo/issue data in DAPO",
            "terminal_bench": "no overlap surface: no terminal tasks in DAPO",
            "hle": "HLE public set not scanned offline (gated); DAPO predates HLE curation and is math-only — risk accepted and documented",
        },
    }
    (OUT_DIR / "phase_a_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("manifest:", manifest_path)


if __name__ == "__main__":
    sys.exit(main())
