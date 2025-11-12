"""Tracks and generates multiple poses independently."""

from typing import Callable, Generic, TypeVar

from modules.pose.Pose import Pose, PoseDict
from modules.pose.nodes.Nodes import GeneratorNode


TInput = TypeVar('TInput')


class GeneratorTracker(Generic[TInput]):
    """Tracks multiple inputs, maintaining a separate generator for each.

    Each input ID gets its own GeneratorNode instance which maintains
    independent state. Generators are created upfront for a fixed number of IDs.

    Type parameter TInput specifies what type of data the generators consume
    (e.g., Tracklet, np.ndarray, etc.)
    """

    def __init__(self, num_inputs: int, generator_factory: Callable[[], GeneratorNode[TInput]]) -> None:
        """Initialize tracker with generators for fixed number of inputs.

        Args:
            num_inputs: Number of inputs to track.
            generator_factory: Factory function that creates GeneratorNode instances.
        """
        self._num_inputs = num_inputs
        self._generator_factory = generator_factory

        # Create one generator instance per input ID
        self._generators: dict[int, GeneratorNode[TInput]] = {
            input_id: generator_factory() for input_id in range(num_inputs)
        }

    def set(self, inputs: dict[int, TInput]) -> None:
        """Set input data for all generators.

        Args:
            inputs: Dictionary of id -> input_data

        Raises:
            ValueError: If input ID exceeds configured range
        """
        for input_id, input_data in inputs.items():
            if input_id not in self._generators:
                raise ValueError(f"Input ID {input_id} exceeds configured range (0-{self._num_inputs-1})")

            self._generators[input_id].set(input_data)

    def generate(self) -> PoseDict:
        """Generate poses from all ready generators.

        Returns:
            Dictionary of id -> Pose for all ready generators
        """
        poses: PoseDict = {}

        for input_id, generator in self._generators.items():
            if generator.is_ready():
                poses[input_id] = generator.generate()

        return poses

    def reset(self) -> None:
        """Reset all generators and clear state."""
        for generator in self._generators.values():
            generator.reset()

    def reset_generator(self, input_id: int) -> None:
        """Reset generator for a specific input ID."""
        if input_id in self._generators:
            self._generators[input_id].reset()