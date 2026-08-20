"""Typed workflow schema, parser, and deterministic execution helpers."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from typing import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class WorkflowFormatError(ValueError):
    """The orchestrator output cannot be parsed as a valid workflow."""


class WorkerInfrastructureError(Exception):
    """A worker step exhausted retries because its inference service failed."""

    def __init__(self, *, worker: str, step_id: int, attempts: int, cause: Exception):
        self.worker = worker
        self.step_id = step_id
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"infrastructure failure: worker={worker} step={step_id} "
            f"attempts={attempts} cause={type(cause).__name__}: {cause}"
        )


async def retry_worker_step(
    operation: Callable[[], Awaitable[str]],
    *,
    worker: str,
    step_id: int,
    max_attempts: int,
    retryable: tuple[type[BaseException], ...],
    delays: Sequence[float],
) -> str:
    """Retry one worker step without repeating completed workflow steps."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except retryable as exc:
            if attempt == max_attempts:
                raise WorkerInfrastructureError(
                    worker=worker, step_id=step_id, attempts=attempt, cause=exc
                ) from exc
            delay = delays[min(attempt - 1, len(delays) - 1)] if delays else 0.0
            if delay > 0:
                await asyncio.sleep(delay)
    raise AssertionError("unreachable")


class WorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=1)
    subtask: str = Field(min_length=1)
    worker: str = Field(min_length=1)
    access: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def access_is_prior_and_unique(self) -> "WorkflowStep":
        if len(set(self.access)) != len(self.access):
            raise ValueError("access entries must be unique")
        if any(parent >= self.id for parent in self.access):
            raise ValueError("access may reference only prior steps")
        return self


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: list[WorkflowStep] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def step_ids_are_contiguous(self) -> "Workflow":
        ids = [step.id for step in self.steps]
        expected = list(range(1, len(self.steps) + 1))
        if ids != expected:
            raise ValueError(f"step ids must be contiguous in order: {expected}")
        return self


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise WorkflowFormatError("no JSON object found")
    return candidate[start : end + 1]


def parse_workflow(text: str, allowed_workers: set[str]) -> Workflow:
    try:
        raw = json.loads(_extract_json(text))
        workflow = Workflow.model_validate(raw)
    except (json.JSONDecodeError, ValidationError, WorkflowFormatError) as exc:
        raise WorkflowFormatError(str(exc)) from exc
    unknown = sorted({step.worker for step in workflow.steps} - allowed_workers)
    if unknown:
        raise WorkflowFormatError(f"unknown workers: {unknown}")
    return workflow


def build_worker_prompt(
    question: str,
    step: WorkflowStep,
    results: Mapping[int, str],
) -> str:
    context = [
        {"step": parent, "output": results[parent]}
        for parent in step.access
        if parent in results
    ]
    if len(context) != len(step.access):
        missing = sorted(set(step.access) - results.keys())
        raise WorkflowFormatError(f"step {step.id} missing dependencies: {missing}")
    return (
        "Solve the assigned subtask. Return only the useful result for the next "
        "workflow step.\n\n"
        f"Original task:\n{question}\n\n"
        f"Assigned subtask:\n{step.subtask}\n\n"
        "Accessible prior outputs (and no others):\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )


WorkerCall = Callable[[WorkflowStep, str], Awaitable[str]]


async def execute_workflow(
    workflow: Workflow,
    question: str,
    call_worker: WorkerCall,
) -> tuple[str, dict[int, str]]:
    results: dict[int, str] = {}
    for step in workflow.steps:
        prompt = build_worker_prompt(question, step, results)
        results[step.id] = await call_worker(step, prompt)
    return results[workflow.steps[-1].id], results
