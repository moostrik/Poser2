"""Single-pose filter pipeline for chaining multiple filters."""

from modules.pose.Pose import Pose
from modules.pose.filters.FilterBase import FilterBase


class FilterPipeline(FilterBase):
    """Chains multiple filters into a sequential pipeline for a single pose.

    Processes a pose through a series of filters in order, where each filter's
    output becomes the next filter's input.
    """

    def __init__(self, filters: list[FilterBase]) -> None:
        """Initialize pipeline with a list of filters.

        Args:
            filters: List of filter instances to apply in sequence.
                    Must not be empty.

        Example:
            pipeline = PoseFilterPipeline([
                PoseValidator(config),
                PoseConfidenceFilter(config),
                PoseStickyFiller(config),
                PoseSmoother(config),
            ])

            filtered_pose = pipeline.process(pose)
        """
        if not filters:
            raise ValueError("PoseFilterPipeline: filters list must not be empty.")

        self._filters: list[FilterBase] = filters

    def process(self, pose: Pose) -> Pose:
        """Process pose through all filters in sequence."""
        current_pose: Pose = pose

        for filter_instance in self._filters:
            current_pose = filter_instance.process(current_pose)

        return current_pose

    def reset(self) -> None:
        """Reset all filters in the pipeline."""
        for filter_instance in self._filters:
            filter_instance.reset()