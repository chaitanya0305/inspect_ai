"""Sample metadata for metrics computed from logged samples."""

from collections.abc import Sequence
from logging import getLogger
from typing import Any

from inspect_ai.event._event import Event
from inspect_ai.event._sample_init import SampleInitEvent

logger = getLogger(__name__)

_MAX_SAMPLES_NAMED = 5


def find_sample_init(events: Sequence[Event]) -> SampleInitEvent | None:
    """The sample's first `SampleInitEvent`, or None if it has none."""
    return next((e for e in events if isinstance(e, SampleInitEvent)), None)


class MetricSampleMetadata:
    """Chooses the sample metadata that a logged score's metrics receive.

    A fresh run gives the metrics of a declared scorer's score the dataset
    metadata, and the metrics of a solver-set score the final
    `TaskState.metadata`. A log records the final metadata as
    `EvalSample.metadata` and the dataset metadata in the sample's first
    `SampleInitEvent` (the event the shard merge in `log/_shards/_write.py`
    also reads). Metrics recomputed from a log use the same inputs.

    A sample logged without a `SampleInitEvent` (events excluded, or a sample
    whose record could not be serialized) has no dataset metadata, so its
    declared scorers' metrics fall back to the final metadata. That gives
    different results from the run when a solver changed metadata a metric
    reads (e.g. a `grouped()` or `cluster` key). `warn()` names such samples.
    """

    def __init__(self) -> None:
        self._missing: dict[tuple[str, int | None], None] = {}

    def metadata(
        self,
        *,
        declared_scorer: bool,
        sample_id: int | str,
        epoch: int | None,
        sample_init: SampleInitEvent | None,
        final_metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Sample metadata for the metrics of one of a sample's scores.

        Args:
            declared_scorer: Whether a scorer the task declares produced the
                score (False for a score a solver set).
            sample_id: Id of the sample.
            epoch: Epoch of the sample.
            sample_init: The sample's `SampleInitEvent` (`find_sample_init()`).
            final_metadata: The sample's final metadata (`EvalSample.metadata`).
        """
        if not declared_scorer:
            return final_metadata
        if sample_init is None:
            self._missing[(str(sample_id), epoch)] = None
            return final_metadata
        return sample_init.sample.metadata

    def warn(self) -> None:
        """Log one warning naming the samples that fell back to final metadata."""
        if not self._missing:
            return
        named = ", ".join(
            f"{sample_id} (epoch {epoch})" if epoch is not None else sample_id
            for sample_id, epoch in list(self._missing)[:_MAX_SAMPLES_NAMED]
        )
        more = len(self._missing) - _MAX_SAMPLES_NAMED
        if more > 0:
            named = f"{named} and {more} more"
        logger.warning(
            f"{len(self._missing)} sample(s) have no sample_init event: {named}. "
            "Their scorers' metrics use the sample's final metadata instead of "
            "the dataset metadata the run used, so the results differ from the "
            "run's if a solver changed metadata that a metric reads (such as a "
            "grouped() or cluster key)."
        )
        self._missing.clear()
