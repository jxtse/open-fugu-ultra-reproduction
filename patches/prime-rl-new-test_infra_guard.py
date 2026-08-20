from __future__ import annotations

import pytest

from prime_rl.orchestrator.infra_guard import InfraCircuitBreaker, is_infra_error_message


def test_worker_infrastructure_marker_is_infra() -> None:
    msg = (
        "OpenFuguEnv.run(): WorkerInfrastructureError: infrastructure failure: "
        "worker=worker_c step=2 attempts=4 cause=APITimeoutError: Request timed out."
    )
    assert is_infra_error_message(msg)


def test_connection_and_timeout_errors_are_infra() -> None:
    assert is_infra_error_message("OpenFuguEnv.run(): APIConnectionError: Connection error.")
    assert is_infra_error_message("OpenFuguEnv.run(): APITimeoutError: Request timed out.")


def test_semantic_errors_are_not_infra() -> None:
    assert not is_infra_error_message("worker worker_a returned no text on step 1")
    assert not is_infra_error_message("no JSON object found")
    assert not is_infra_error_message("Expecting value: line 1 column 12 (char 11)")
    assert not is_infra_error_message(None)


def test_breaker_does_not_trip_below_min_events() -> None:
    breaker = InfraCircuitBreaker(window=10, max_rate=0.5, min_events=5)
    for _ in range(4):
        breaker.record(infra_dropped=True)
    assert not breaker.tripped


def test_breaker_trips_on_high_infra_drop_rate() -> None:
    breaker = InfraCircuitBreaker(window=10, max_rate=0.5, min_events=5)
    for _ in range(3):
        breaker.record(infra_dropped=False)
    for _ in range(4):
        breaker.record(infra_dropped=True)
    assert breaker.tripped
    with pytest.raises(RuntimeError, match="circuit breaker"):
        breaker.raise_if_tripped()


def test_breaker_recovers_when_window_slides_past_outage() -> None:
    breaker = InfraCircuitBreaker(window=6, max_rate=0.5, min_events=4)
    for _ in range(4):
        breaker.record(infra_dropped=True)
    assert breaker.tripped
    # After the outage ends, healthy groups push the failures out of the window.
    for _ in range(6):
        breaker.record(infra_dropped=False)
    assert not breaker.tripped


def test_group_with_infra_loss_is_dropped_entirely() -> None:
    from prime_rl.orchestrator.infra_guard import group_infra_verdict

    verdict = group_infra_verdict(
        error_messages=[
            "OpenFuguEnv.run(): WorkerInfrastructureError: infrastructure failure: "
            "worker=worker_c step=2 attempts=4 cause=APITimeoutError: Request timed out.",
        ]
    )
    assert verdict.drop_group
    assert verdict.infra_errored == 1


def test_group_with_only_semantic_errors_keeps_survivors() -> None:
    from prime_rl.orchestrator.infra_guard import group_infra_verdict

    verdict = group_infra_verdict(
        error_messages=["worker worker_a returned no text on step 1", None]
    )
    assert not verdict.drop_group
    assert verdict.infra_errored == 0


def test_clean_group_is_not_dropped() -> None:
    from prime_rl.orchestrator.infra_guard import group_infra_verdict

    verdict = group_infra_verdict(error_messages=[])
    assert not verdict.drop_group
    assert verdict.infra_errored == 0


def _make_rollout(*, error: Exception | None = None, reward: float | None = None):
    import verifiers.v1 as vf

    from prime_rl.orchestrator.types import Rollout

    rollout = Rollout(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=1, prompt=None)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
    )
    rollout.env_name = "test-env"
    rollout.ok = True
    if error is not None:
        rollout.record_error(error)
        rollout.ok = False
    if reward is not None:
        rollout.record_reward("conductor_reward", reward)
    return rollout


def _make_sink():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from prime_rl.orchestrator.train_sink import TrainSink

    env = SimpleNamespace(
        config=SimpleNamespace(group_size=2),
        algorithm=SimpleNamespace(
            finalize_rollout=AsyncMock(),
            finalize_group=AsyncMock(),
        ),
        sampling_args={"temperature": 1.0},
    )
    train_envs = MagicMock()
    train_envs.get.return_value = env
    return TrainSink(
        config=MagicMock(),
        tokenizer=MagicMock(),
        train_envs=train_envs,
        mm_token_type_ids_mapping=None,
        batch_size=4,
        token_batch_size=None,
        pre_filters=[],
        post_filters=[],
    )


def test_train_sink_drops_whole_group_on_infra_error() -> None:
    import asyncio
    import uuid

    class FakeInfraError(Exception):
        pass

    sink = _make_sink()
    gid = uuid.uuid4()
    good = _make_rollout(reward=1.0)
    bad = _make_rollout(
        error=FakeInfraError(
            "OpenFuguEnv.run(): WorkerInfrastructureError: infrastructure failure: "
            "worker=worker_c step=2 attempts=4 cause=APITimeoutError: Request timed out."
        )
    )
    for r in (good, bad):
        r.group_id = gid
    sink.pending_groups[gid] = [good, bad]
    sink.pending_group_episodes[gid] = 2

    asyncio.run(sink.process_group(gid))

    assert sink.pending_batch == []
    assert sink.infra_dropped_groups == 1
    assert sink.infra_dropped_rollouts == 2


def test_train_sink_keeps_survivors_on_semantic_error() -> None:
    import asyncio
    import uuid

    sink = _make_sink()
    gid = uuid.uuid4()
    good = _make_rollout(reward=1.0)
    good.agent.trainable = True
    bad = _make_rollout(error=ValueError("worker worker_a returned no text on step 1"))
    for r in (good, bad):
        r.group_id = gid
    sink.pending_groups[gid] = [good, bad]
    sink.pending_group_episodes[gid] = 2

    asyncio.run(sink.process_group(gid))

    assert sink.infra_dropped_groups == 0
    assert good in sink.pending_batch
