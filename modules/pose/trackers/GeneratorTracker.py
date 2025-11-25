"""Tracks and generates multiple poses independently."""

from traceback import print_exc
from typing import Callable, Generic, TypeVar

from modules.pose.Frame import FrameDict
from modules.pose.nodes.Nodes import GeneratorNode
from .TrackerBase import TrackerBase


TInput = TypeVar('TInput')


class GeneratorTracker(TrackerBase, Generic[TInput]):
    """Tracks multiple inputs, maintaining a separate generator for each."""

    def __init__(self, num_tracks: int, generator_factory: Callable[[], GeneratorNode[TInput]]) -> None:
        """Initialize tracker with generators for fixed number of inputs."""
        super().__init__()  # Initialize PoseDictCallbackMixin

        self._generators: dict[int, GeneratorNode[TInput]] = {
            input_id: generator_factory()
            for input_id in range(num_tracks)
        }

    def submit(self, input_data_dict: dict[int, TInput]) -> None:
        """Set input data for generators."""

        for input_id, generator in self._generators.items():
            if input_id in input_data_dict:
                generator.submit(input_data_dict[input_id])
            else:
                generator.submit(None)



    def update(self) -> FrameDict:
        """Generate poses from all ready generators."""

        generated_poses: FrameDict = {}

        for input_id, generator in self._generators.items():
            try:
                if generator.is_ready():
                    generated_poses[input_id] = generator.update()
            except Exception as e:
                print(f"GeneratorTracker: Error generating pose {input_id}: {e}")
                print_exc()

        self._notify_poses_callbacks(generated_poses)

        return generated_poses

    def is_ready(self) -> bool:
        """Return True if all generators are ready to generate poses."""
        return all(generator.is_ready() for generator in self._generators.values())

    def reset(self) -> None:
        """Reset all generators."""
        for generator in self._generators.values():
            generator.reset()

    def reset_at(self, id: int) -> None:
        """Reset generator for a specific input ID."""
        if id in self._generators:
            self._generators[id].reset()