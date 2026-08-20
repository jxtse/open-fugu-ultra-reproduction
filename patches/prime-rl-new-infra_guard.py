"""Infrastructure-failure guard for the training rollout stream.

Two concerns, both deliberately small and dependency-free:

- ``is_infra_error_message``: classify a rollout error message as an
  infrastructure failure (endpoint/tunnel/provider transport problem) versus a
  semantic env failure (bad workflow JSON, empty worker text, wrong answer).
  Infra failures must never become reward-0 training signal, and a group that
  lost a member to infra failure must not train on the survivor subset (the
  loss is action-conditioned, so the survivor baseline is biased).

- ``InfraCircuitBreaker``: sliding-window trip switch at group granularity.
  Retries absorb transient flakes; correlated outages (dead tunnel, provider
  down) would otherwise burn unbounded refill sampling while the run drifts —
  stop the run instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# Substrings that identify transport/provider infrastructure failures. Keep in
# sync with the retryable set in open_fugu_env (APIConnectionError,
# APITimeoutError, InternalServerError, RateLimitError) plus the attributed
# marker raised after retry exhaustion.
_INFRA_MARKERS: tuple[str, ...] = (
    "WorkerInfrastructureError",
    "infrastructure failure:",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "RateLimitError",
    "Connection error",
    "Request timed out",
)


def is_infra_error_message(message: str | None) -> bool:
    """True when the error text identifies an infrastructure failure."""
    if not message:
        return False
    return any(marker in message for marker in _INFRA_MARKERS)


@dataclass(frozen=True)
class GroupInfraVerdict:
    """Outcome of screening one finalized group's error messages."""

    infra_errored: int
    drop_group: bool


def group_infra_verdict(*, error_messages: list[str | None]) -> GroupInfraVerdict:
    """Screen a finalized group for infrastructure losses.

    Any infra-classified error message means the group is incomplete for a
    reason correlated with the orchestrator's own action (which worker the
    policy picked), so training on the survivor subset would bias the group
    baseline. Verdict: drop the whole group. Semantic errors (bad JSON, empty
    worker text) stay group-local — those rollouts are already dropped
    individually and the survivors remain a fair comparison set.
    """
    infra = sum(1 for message in error_messages if is_infra_error_message(message))
    return GroupInfraVerdict(infra_errored=infra, drop_group=infra > 0)


class InfraCircuitBreaker:
    """Sliding-window breaker over group finalizations.

    ``record(infra_dropped=...)`` once per finalized group; ``tripped`` turns
    True when, across the last ``window`` groups (and at least ``min_events``
    of them), the fraction dropped for infrastructure reasons exceeds
    ``max_rate``. The window keeps sliding after a trip, so a breaker consulted
    (rather than raised) can also recover once healthy groups flush the outage
    out of the window.
    """

    def __init__(self, *, window: int = 40, max_rate: float = 0.05, min_events: int = 20) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 0.0 < max_rate <= 1.0:
            raise ValueError("max_rate must be in (0, 1]")
        if min_events < 1:
            raise ValueError("min_events must be >= 1")
        self.window = window
        self.max_rate = max_rate
        self.min_events = min_events
        self._events: deque[bool] = deque(maxlen=window)

    def record(self, *, infra_dropped: bool) -> None:
        self._events.append(infra_dropped)

    @property
    def infra_rate(self) -> float:
        if not self._events:
            return 0.0
        return sum(self._events) / len(self._events)

    @property
    def tripped(self) -> bool:
        if len(self._events) < self.min_events:
            return False
        return self.infra_rate > self.max_rate

    def raise_if_tripped(self) -> None:
        if self.tripped:
            raise RuntimeError(
                f"infrastructure circuit breaker tripped: {sum(self._events)}/{len(self._events)} "
                f"recent groups dropped for infrastructure failures "
                f"(rate {self.infra_rate:.1%} > limit {self.max_rate:.1%}). "
                "Check worker endpoints/tunnels before resuming; refill sampling "
                "during an outage wastes budget and skews the rollout stream."
            )
