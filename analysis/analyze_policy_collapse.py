#!/usr/bin/env python3
"""Recompute the main policy-collapse diagnostics from Open Fugu traces."""

from __future__ import annotations

import argparse
import collections
import json
import math
import tarfile
from pathlib import Path

GENERIC = "Compute the requested result and return only it."
WINDOWS = [(1, 20), (21, 40), (41, 55), (56, 65), (66, 75), (76, 85), (86, 100)]


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_trace(record: dict) -> dict:
    # Eval output wraps one or more traces; training output is a trace directly.
    return record["traces"][0] if "traces" in record else record


def workflow(trace: dict):
    return trace.get("info", {}).get("workflow")


def is_correct(trace: dict) -> bool:
    metric = trace.get("metrics", {}).get("answer_correct", 0)
    return float(metric.get("value", 0) if isinstance(metric, dict) else metric) == 1.0


def entropy(counts: collections.Counter[str]) -> float:
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values()) if total else 0.0


def heldout(path: Path, label: str):
    traces = [get_trace(x) for x in iter_jsonl(path)]
    valid = [t for t in traces if workflow(t)]
    lengths = collections.Counter(len(workflow(t)["steps"]) for t in valid)
    workers = collections.Counter(s["worker"] for t in valid for s in workflow(t)["steps"])
    subtasks = [s["subtask"] for t in valid for s in workflow(t)["steps"]]
    exact = {
        json.dumps(workflow(t), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for t in valid
    }
    correct = sum(is_correct(t) for t in traces)
    conditional = sum(is_correct(t) for t in valid)
    print(f"[{label}]")
    print(f"total={len(traces)} valid={len(valid)} ({len(valid)/len(traces):.1%}) correct={correct} ({correct/len(traces):.1%})")
    print(f"correct_given_valid={conditional}/{len(valid)} ({conditional/len(valid):.1%})")
    print(f"workflow_lengths={dict(sorted(lengths.items()))} exact_templates={len(exact)} unique_subtasks={len(set(subtasks))}")
    print(f"workers={dict(workers)} entropy={entropy(workers):.2f} bits generic_subtasks={subtasks.count(GENERIC)}/{len(subtasks)}")
    print()


def training(archive: Path):
    by_step: dict[int, list[dict]] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            # Use all/ rather than effective/ so validity collapse is visible.
            if not member.isfile() or not member.name.endswith("/train/all/traces.jsonl"):
                continue
            step = int(member.name.split("/")[1].split("_")[1])
            f = tar.extractfile(member)
            assert f is not None
            by_step[step] = [json.loads(line) for line in f if line.strip()]

    print("[training windows: train/all]")
    print("steps | valid | one-step | multi-step | worker A/B/C | entropy")
    for lo, hi in WINDOWS:
        traces = [t for step in range(lo, hi + 1) for t in by_step.get(step, [])]
        valid = [t for t in traces if workflow(t)]
        one = [t for t in valid if len(workflow(t)["steps"]) == 1]
        multi = [t for t in valid if len(workflow(t)["steps"]) > 1]
        workers = collections.Counter(s["worker"] for t in valid for s in workflow(t)["steps"])
        n_workers = sum(workers.values()) or 1
        mix = "/".join(f"{100*workers.get(w,0)/n_workers:.1f}" for w in ("worker_a", "worker_b", "worker_c"))
        n = len(traces) or 1
        print(f"{lo:02d}-{hi:03d} | {len(valid)/n:6.1%} | {len(one)/n:8.1%} | {len(multi)/n:10.1%} | {mix} | {entropy(workers):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.root
    heldout(root / "runs/heldout-eval/base-v1/traces.jsonl", "base matched-v1")
    heldout(root / "runs/heldout-eval/step100-v1/traces.jsonl", "trained step100 matched-v1")
    training(root / "evidence/train_rollouts_steps1-100.tar.gz")


if __name__ == "__main__":
    main()
