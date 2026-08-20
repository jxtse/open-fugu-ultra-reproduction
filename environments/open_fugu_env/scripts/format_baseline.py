"""Format-compliance baseline for Qwen3.5-9B on real DAPO tasks.

Arms:
  A) thinking OFF (enable_thinking=false) — clean content, schema in system.
  B) thinking ON with reasoning parser — content should hold only final output.
  C) thinking OFF + schema reminder appended to the user turn.

Each arm runs N manifest tasks x R samples; a response passes if
parse_workflow accepts the content. Prints per-arm legal rates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import httpx

PKG = Path(__file__).resolve().parents[1] / "open_fugu_env"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ORCHESTRATOR_SYSTEM = _load("prompts").ORCHESTRATOR_SYSTEM
_workflow = _load("workflow")
parse_workflow = _workflow.parse_workflow
WorkflowFormatError = _workflow.WorkflowFormatError

ALLOWED = {"worker_a", "worker_b", "worker_c"}

REMINDER = (
    "\n\nRemember: reply with ONE JSON object only, schema "
    '{"steps":[{"id":1,"subtask":"...","worker":"worker_a|worker_b|worker_c",'
    '"access":[]}]} — no prose, no code fences.'
)


def run_arm(client: httpx.Client, base_url: str, model: str, tasks, samples: int,
            thinking: bool, reminder: bool, max_tokens: int) -> tuple[int, int, list[str]]:
    legal = total = 0
    errors: list[str] = []
    for task in tasks:
        prompt = task["question"] + (REMINDER if reminder else "")
        for _ in range(samples):
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": thinking},
            }
            response = client.post(f"{base_url}/chat/completions", json=body, timeout=300)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"] or ""
            total += 1
            try:
                parse_workflow(content, ALLOWED)
                legal += 1
            except WorkflowFormatError as exc:
                errors.append(f"{str(exc)[:60]} | head={content[:80]!r}")
    return legal, total, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in open(args.manifest)][: args.tasks]
    arms = [
        ("A think-off", False, False),
        ("B think-on+parser", True, False),
        ("C think-off+reminder", False, True),
    ]
    with httpx.Client(trust_env=False) as client:
        for name, thinking, reminder in arms:
            legal, total, errors = run_arm(
                client, args.base_url, args.model, tasks, args.samples,
                thinking, reminder, args.max_tokens,
            )
            print(f"[{name}] legal={legal}/{total} ({100 * legal / total:.0f}%)")
            for err in errors[:3]:
                print(f"    fail: {err}")


if __name__ == "__main__":
    main()
