from __future__ import annotations

import asyncio
import json

from open_fugu_env.workflow import (
    WorkflowFormatError,
    build_worker_prompt,
    execute_workflow,
    parse_workflow,
)

ALLOWED = {"worker_a", "worker_b", "worker_c"}


def valid_payload() -> str:
    return json.dumps(
        {
            "steps": [
                {"id": 1, "subtask": "derive", "worker": "worker_a", "access": []},
                {"id": 2, "subtask": "verify", "worker": "worker_b", "access": [1]},
            ]
        }
    )


def test_valid_workflow_and_execution() -> None:
    workflow = parse_workflow(valid_payload(), ALLOWED)
    seen: list[tuple[int, str]] = []

    async def worker(step, prompt):
        seen.append((step.id, prompt))
        return f"result-{step.id}"

    answer, results = asyncio.run(execute_workflow(workflow, "question", worker))
    assert answer == "result-2"
    assert results == {1: "result-1", 2: "result-2"}
    assert "result-1" not in seen[0][1]
    assert "result-1" in seen[1][1]


def test_access_list_isolation() -> None:
    workflow = parse_workflow(
        json.dumps(
            {
                "steps": [
                    {"id": 1, "subtask": "one", "worker": "worker_a", "access": []},
                    {"id": 2, "subtask": "two", "worker": "worker_b", "access": []},
                    {"id": 3, "subtask": "three", "worker": "worker_c", "access": [1]},
                ]
            }
        ),
        ALLOWED,
    )
    prompt = build_worker_prompt("q", workflow.steps[2], {1: "VISIBLE", 2: "SECRET"})
    assert "VISIBLE" in prompt
    assert "SECRET" not in prompt


def test_invalid_workflows_are_rejected() -> None:
    bad = [
        "not json",
        json.dumps({"steps": []}),
        json.dumps({"steps": [{"id": 2, "subtask": "x", "worker": "worker_a", "access": []}]}),
        json.dumps({"steps": [{"id": 1, "subtask": "x", "worker": "missing", "access": []}]}),
        json.dumps({"steps": [{"id": 1, "subtask": "x", "worker": "worker_a", "access": [1]}]}),
        json.dumps({"steps": [{"id": i, "subtask": "x", "worker": "worker_a", "access": []} for i in range(1, 7)]}),
    ]
    for payload in bad:
        try:
            parse_workflow(payload, ALLOWED)
        except WorkflowFormatError:
            pass
        else:
            raise AssertionError(f"accepted invalid workflow: {payload}")

def test_retry_worker_step_retries_only_failing_step() -> None:
    from open_fugu_env.workflow import retry_worker_step

    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("infra")
        return "ok"

    result = asyncio.run(
        retry_worker_step(
            flaky,
            worker="worker_c",
            step_id=2,
            max_attempts=3,
            retryable=(TimeoutError,),
            delays=(0.0, 0.0),
        )
    )
    assert result == "ok"
    assert calls == 3


def test_retry_worker_step_raises_attributed_error_after_budget() -> None:
    from open_fugu_env.workflow import WorkerInfrastructureError, retry_worker_step

    calls = 0

    async def broken() -> str:
        nonlocal calls
        calls += 1
        raise ConnectionError("down")

    try:
        asyncio.run(
            retry_worker_step(
                broken,
                worker="worker_b",
                step_id=1,
                max_attempts=2,
                retryable=(ConnectionError,),
                delays=(0.0,),
            )
        )
    except WorkerInfrastructureError as exc:
        assert exc.worker == "worker_b"
        assert exc.step_id == 1
        assert exc.attempts == 2
        assert "worker_b" in str(exc)
    else:
        raise AssertionError("expected attributed infrastructure error")
    assert calls == 2


def test_retry_worker_step_does_not_retry_semantic_errors() -> None:
    from open_fugu_env.workflow import retry_worker_step

    calls = 0

    async def bad_result() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("worker returned no text")

    try:
        asyncio.run(
            retry_worker_step(
                bad_result,
                worker="worker_a",
                step_id=1,
                max_attempts=4,
                retryable=(TimeoutError,),
                delays=(0.0, 0.0, 0.0),
            )
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected semantic error")
    assert calls == 1


def test_worker_infrastructure_error_is_not_a_semantic_runtime_error() -> None:
    from open_fugu_env.workflow import WorkerInfrastructureError

    assert not issubclass(WorkerInfrastructureError, RuntimeError)
