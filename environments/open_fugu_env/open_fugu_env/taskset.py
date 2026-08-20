"""Open Fugu: a constrained multi-agent workflow environment."""

from __future__ import annotations

import json
import re
from typing import ClassVar

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.harnesses.null import NullHarnessConfig

from .grading import grade_answer
from .prompts import ORCHESTRATOR_SYSTEM, WORKER_SYSTEM
from .workflow import (
    WorkflowFormatError,
    execute_workflow,
    parse_workflow,
    retry_worker_step,
)


def extract_final_answer(reply: str) -> str:
    """Pull the final answer out of a worker reply (last 'Answer:' line wins)."""
    matches = re.findall(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", reply or "")
    if matches:
        return matches[-1].strip()
    return (reply or "").strip()


class OpenFuguData(vf.TaskData):
    answer: str


class OpenFuguTask(vf.Task[OpenFuguData]):
    pass


class OpenFuguConfig(vf.TasksetConfig):
    samples: int = Field(4, ge=1)
    source: str = Field(
        "smoke", description="Task source: 'smoke', 'dapo', or 'manifest'."
    )
    manifest_path: str = Field(
        "/vePFS-Mindverse/share/richard/open-fugu-ultra/data/phase_a/phase_a_manifest.jsonl",
        description="Frozen task manifest (JSONL with question/answer/sha256).",
    )
    seed: int = 17


class OpenFuguTaskset(vf.Taskset[OpenFuguTask, OpenFuguConfig]):
    SPECS: ClassVar[list[tuple[str, str]]] = [
        ("Compute 17 + 25. Return only the integer.", "42"),
        ("Compute 9 * 13. Return only the integer.", "117"),
        ("Reverse the string ORCHESTRATE. Return only the reversed string.", "ETARTSEHCRO"),
        ("Sort 8, 3, 11, 2 ascending. Return comma-separated integers.", "2,3,8,11"),
    ]

    def load(self) -> list[OpenFuguTask]:
        if self.config.source == "manifest":
            rows = self._load_manifest()
        elif self.config.source == "dapo":
            rows = self._load_dapo()
        else:
            rows = self.SPECS[: self.config.samples]
        # Keep the coordinator system instruction inside the user turn. Qwen3.5's
        # training renderer requires a real user query after system messages; the
        # eval relay tolerated the separate TaskData.system_prompt, but the GRPO
        # renderer path rejected it as "No user query found in messages".
        return [
            OpenFuguTask(
                OpenFuguData(
                    idx=i,
                    name=f"{self.config.source}-{i}",
                    prompt=f"{ORCHESTRATOR_SYSTEM}\n\nUser task:\n{question}",
                    system_prompt=None,
                    answer=answer,
                ),
                self.config.task,
            )
            for i, (question, answer) in enumerate(rows)
        ]

    def _load_manifest(self) -> list[tuple[str, str]]:
        """Frozen Phase A manifest — the only sanctioned training source."""
        rows: list[tuple[str, str]] = []
        with open(self.config.manifest_path) as fh:
            for line in fh:
                item = json.loads(line)
                rows.append((item["question"], item["answer"]))
        return rows[: self.config.samples]

    def _load_dapo(self) -> list[tuple[str, str]]:
        """Deduplicated DAPO-Math-17k sample (HF row count is inflated by repeats)."""
        import hashlib
        import random

        from datasets import load_dataset

        dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for row in dataset:
            question = row["prompt"][-1]["content"]
            answer = str(row["reward_model"]["ground_truth"]).strip()
            key = hashlib.sha256(question.encode()).hexdigest()
            if not answer or key in seen:
                continue
            seen.add(key)
            unique.append((question, answer))
        random.Random(self.config.seed).shuffle(unique)
        return unique[: self.config.samples]


class OpenFuguEnvConfig(vf.EnvConfig):
    """Orchestrator is the trainable policy; workers default to the frontier pool
    (override per run via --env.worker-X.model / --env.worker-X.client.base-url).
    Workers default to the null harness (single-shot reasoning): the forwarded
    frontier endpoints only accept plain chat, and Phase A tasks need no tools.
    Opt into bash per run for containerized coding tasks (Phase C).
    NOTE: nested TOML overrides of these fields do NOT reliably reach the
    training path (verified 2026-08-13: run kept bash despite rl.toml null
    override), so defaults here must be the safe configuration."""

    orchestrator: vf.AgentConfig = vf.AgentConfig(
        max_turns=1,
        # Thinking-on experimental arm. Qwen3.5 returns deliberation separately
        # from the final workflow content; the renderer's `think` parser masks it
        # correctly while preserving it as trainable rollout tokens.
        sampling=vf.SamplingConfig(chat_template_kwargs={"enable_thinking": True}),
    )
    worker_a: vf.AgentConfig = vf.AgentConfig(
        model="gpt-5.5",
        max_turns=1,
        harness=NullHarnessConfig(id="null"),
        client=vf.EvalClientConfig(
            base_url="http://127.0.0.1:13000/v1", api_key_var="OPENFUGU_WORKER_KEY"
        ),
    )
    worker_b: vf.AgentConfig = vf.AgentConfig(
        model="dev-anthropic-claude-opus-4-8",
        max_turns=1,
        harness=NullHarnessConfig(id="null"),
        client=vf.EvalClientConfig(
            base_url="http://127.0.0.1:13000/v1", api_key_var="OPENFUGU_WORKER_KEY"
        ),
    )
    worker_c: vf.AgentConfig = vf.AgentConfig(
        model="gemini-3.1-pro-preview",
        max_turns=1,
        harness=NullHarnessConfig(id="null"),
        client=vf.EvalClientConfig(
            base_url="http://127.0.0.1:18142/v1",
            api_key_var="OPENFUGU_WORKER_KEY",
        ),
        # The 18142 proxy intermittently returns malformed completions
        # (502 ProviderError); retry the rollout so infra flakes don't teach
        # the orchestrator to avoid worker_c for the wrong reason.
        retries=vf.RetryConfig(max_retries=2, include=["ProviderError"]),
    )


class OpenFuguEnv(vf.Env[OpenFuguEnvConfig]):
    ALLOWED_WORKERS = frozenset({"worker_a", "worker_b", "worker_c"})

    async def setup(self, agents: vf.Agents) -> None:
        agents.worker_a.trainable = False
        agents.worker_b.trainable = False
        agents.worker_c.trainable = False

    async def start(self) -> None:
        """Fail fast when the forwarded frontier endpoints are unavailable."""
        import httpx

        endpoints = {spec.client.base_url for spec in self._agent_specs.values() if spec.client}
        async with httpx.AsyncClient(timeout=10.0) as client:
            for base_url in endpoints:
                response = await client.get(f"{base_url.rstrip('/')}/models")
                response.raise_for_status()

    async def run(self, task: OpenFuguTask, agents: vf.Agents) -> None:
        orchestrator_trace = await agents.orchestrator.run(task)
        orchestrator_trace.info["workflow_valid"] = False
        orchestrator_trace.info["workflow_answer"] = ""
        orchestrator_trace.info["workflow_results"] = {}
        orchestrator_trace.info["workflow_error"] = None

        try:
            workflow = parse_workflow(
                orchestrator_trace.last_reply or "", set(self.ALLOWED_WORKERS)
            )
        except WorkflowFormatError as exc:
            orchestrator_trace.info["workflow_error"] = str(exc)
            orchestrator_trace.record_metric("answer_correct", 0.0)
            orchestrator_trace.record_reward("conductor_reward", 0.0)
            return

        async def call_worker(step, prompt: str) -> str:
            # Frontier workers are frozen one-shot inference services. Calling them
            # directly avoids nesting a second verifiers harness/interception graph
            # inside the trainable orchestrator rollout; that nested path can drop
            # the user turn and produce `No user query found in messages`.
            import os

            from openai import (
                APIConnectionError,
                APITimeoutError,
                AsyncOpenAI,
                InternalServerError,
                RateLimitError,
            )

            spec = getattr(self.config, step.worker)
            assert spec.client is not None and spec.model is not None
            client = AsyncOpenAI(
                base_url=spec.client.base_url,
                api_key=os.environ.get(spec.client.api_key_var, "dummy"),
                timeout=180.0,
                # Keep retry policy explicit at the workflow-step layer so logs
                # retain worker/step attribution and completed prior steps survive.
                max_retries=0,
            )
            try:
                async def request_completion():
                    return await client.chat.completions.create(
                        model=spec.model,
                        messages=[
                            {"role": "system", "content": WORKER_SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                    )

                completion = await retry_worker_step(
                    request_completion,
                    worker=step.worker,
                    step_id=step.id,
                    max_attempts=4,
                    retryable=(
                        APIConnectionError,
                        APITimeoutError,
                        InternalServerError,
                        RateLimitError,
                    ),
                    delays=(2.0, 8.0, 20.0),
                )
            finally:
                await client.close()
            message = completion.choices[0].message
            if not message.content:
                raise RuntimeError(
                    f"worker {step.worker} returned no text on step {step.id}"
                )
            return extract_final_answer(message.content)

        try:
            answer, results = await execute_workflow(
                workflow, task.data.prompt_text, call_worker
            )
        except (WorkflowFormatError, RuntimeError) as exc:
            orchestrator_trace.info["workflow_error"] = str(exc)
            orchestrator_trace.record_metric("answer_correct", 0.0)
            orchestrator_trace.record_reward("conductor_reward", 0.0)
            return

        correct = float(grade_answer(answer, task.data.answer))
        orchestrator_trace.info["workflow_valid"] = True
        orchestrator_trace.info["workflow_answer"] = answer
        orchestrator_trace.info["workflow_results"] = {
            str(key): value for key, value in results.items()
        }
        orchestrator_trace.info["workflow"] = workflow.model_dump(mode="json")
        orchestrator_trace.record_metric("answer_correct", correct)
        orchestrator_trace.record_reward(
            "conductor_reward", 1.0 if correct else 0.5
        )


__all__ = ["OpenFuguEnv", "OpenFuguTaskset"]
