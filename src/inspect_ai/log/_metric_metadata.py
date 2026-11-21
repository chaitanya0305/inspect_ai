"""Sample metadata for metrics computed from logged samples."""

from collections.abc import Sequence
from datetime import datetime
from logging import getLogger
from typing import Any, NamedTuple

from inspect_ai._util.logger import warn_once
from inspect_ai.event._event import Event
from inspect_ai.event._sample_init import SampleInitEvent
from inspect_ai.event._score import ScoreEvent

logger = getLogger(__name__)


class SampleScoringRecord(NamedTuple):
    """What a logged sample's events record about how its scores were produced."""

    sample_init: SampleInitEvent | None
    """The sample's first `SampleInitEvent` (it holds the dataset metadata)."""

    run_scorers: frozenset[str]
    """Score names a scorer produced during the run, before the sample completed."""


def sample_scoring_record(
    events: Sequence[Event], completed_at: str | None
) -> SampleScoringRecord | None:
    """Read a sample's scoring record from its events (None if it has none).

    A score name counts as produced during the run when its last final
    `ScoreEvent` names it, carries `scorer_args` (set for scorers, not for
    scores a solver set in `state.scores`), and is no later than the sample's
    `completed_at`. Scores added afterwards by `inspect score` are later.
    """
    if not events:
        return None
    completed = datetime.fromisoformat(completed_at) if completed_at else None
    sample_init: SampleInitEvent | None = None
    last_scores: dict[str, ScoreEvent] = {}
    for event in events:
        if isinstance(event, SampleInitEvent) and sample_init is None:
            sample_init = event
        elif (
            isinstance(event, ScoreEvent)
            and not event.intermediate
            and event.scorer is not None
        ):
            last_scores[event.scorer] = event
    run_scorers = frozenset(
        name
        for name, event in last_scores.items()
        if event.scorer_args is not None
        and completed is not None
        and event.timestamp <= completed
    )
    return SampleScoringRecord(sample_init=sample_init, run_scorers=run_scorers)


def metric_sample_metadata(
    score_name: str,
    record: SampleScoringRecord | None,
    final_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Sample metadata for the metrics of one of a logged sample's scores.

    Metrics get the metadata the score's producer saw. During a run, a
    scorer's score goes to the metrics with the dataset metadata, recorded
    by the sample's `SampleInitEvent` (as the shard merge in
    `log/_shards/_write.py` reads it). A score a solver set, or one that
    `inspect score` added later, goes with the final `TaskState.metadata`,
    which the log records as `EvalSample.metadata`.

    A sample logged without events records neither, so its scores fall back
    to the final metadata, with a warning: the results differ from the run's
    if a solver changed metadata that a metric reads.

    Args:
        score_name: Name of the score in `EvalSample.scores`.
        record: The sample's `sample_scoring_record()`.
        final_metadata: The sample's final metadata (`EvalSample.metadata`).
    """
    if record is None:
        warn_once(
            logger,
            "Samples logged without events don't record the dataset metadata, "
            "so their metrics use the final sample metadata. Results differ "
            "from the run's if a solver changed metadata that a metric reads "
            "(such as a grouped() or cluster key).",
        )
        return final_metadata
    if record.sample_init is not None and score_name in record.run_scorers:
        return record.sample_init.sample.metadata
    return final_metadata
