"""Generate SFT cold-start data: legal workflow JSON per Phase A task.

For each manifest task, ask a frontier model to produce a workflow JSON that
follows the orchestrator schema; validate with parse_workflow; retry on
failure. Output JSONL: {prompt, system, target, sha256, attempts}.

Run locally on the Mac (frontier endpoint at 127.0.0.1:3000).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "open_fugu_env"
import importlib.util


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # register so pydantic forward refs resolve
    spec.loader.exec_module(module)
    return module


import httpx  # noqa: E402

ORCHESTRATOR_SYSTEM = _load("prompts").ORCHESTRATOR_SYSTEM
_workflow = _load("workflow")
WorkflowFormatError = _workflow.WorkflowFormatError
parse_workflow = _workflow.parse_workflow

GEN_SYSTEM = ORCHESTRATOR_SYSTEM + """

You are generating a training example. Produce a REALISTIC, high-quality
workflow for the given task: decompose it into 2-3 steps when the task
benefits from decomposition (compute -> verify/format), or 1 step when
trivial. Vary worker choices across steps sensibly."""

ALLOWED = {"worker_a", "worker_b", "worker_c"}


def generate_one(client: httpx.Client, base_url: str, model: str, question: str,
                 max_attempts: int = 4) -> tuple[str | None, int]:
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": GEN_SYSTEM},
                        {"role": "user", "content": question},
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.4 + 0.2 * (attempt - 1),
                },
                timeout=120,
            )
        except httpx.HTTPError:
            time.sleep(2 * attempt)
            continue
        if response.status_code != 200:
            time.sleep(2 * attempt)
            continue
        reply = response.json()["choices"][0]["message"]["content"] or ""
        try:
            workflow = parse_workflow(reply, ALLOWED)
        except WorkflowFormatError:
            continue
        return json.dumps(workflow.model_dump(mode="json"), ensure_ascii=False), attempt
    return None, max_attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000/v1")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in open(args.manifest)]
    if args.limit:
        tasks = tasks[args.start : args.start + args.limit]

    out_path = Path(args.out)
    done_hashes = set()
    if out_path.exists():
        for line in open(out_path):
            done_hashes.add(json.loads(line)["sha256"])
        print(f"resuming: {len(done_hashes)} already done", flush=True)

    ok = fail = 0
    with httpx.Client() as client, out_path.open("a") as fh:
        for i, task in enumerate(tasks):
            if task["sha256"] in done_hashes:
                continue
            target, attempts = generate_one(
                client, args.base_url, args.model, task["question"]
            )
            if target is None:
                fail += 1
                print(f"[{i}] FAILED after {attempts} attempts", flush=True)
                continue
            fh.write(
                json.dumps(
                    {
                        "sha256": task["sha256"],
                        "prompt": task["question"],
                        "system": ORCHESTRATOR_SYSTEM,
                        "target": target,
                        "attempts": attempts,
                        "gen_model": args.model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            ok += 1
            if ok % 20 == 0:
                print(f"progress: {ok} ok, {fail} failed", flush=True)
    print(f"DONE: {ok} generated, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
