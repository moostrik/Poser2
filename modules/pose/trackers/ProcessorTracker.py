"""Tracks and processes multiple poses independently using stored context."""

from threading import Lock
from traceback import print_exc
from typing import Callable, Generic, TypeVar

from modules.pose.Pose import Pose, PoseDict
from modules.pose.nodes.Nodes import ProcessorNode
from .TrackerBase import TrackerBase


TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')

TOutputDict = dict[int, TOutput]
Output_Callback = Callable[[TOutputDict], None]

class ProcessorTracker(TrackerBase, Generic[TInput, TOutput]):
    """Tracks multiple poses, maintaining a separate processor for each."""

    def __init__(self, num_tracks: int, processor_factory: Callable[[], ProcessorNode[TInput, TOutput]]) -> None:
        """Initialize tracker with processors for fixed number of poses. """
        super().__init__() # Initialize PoseDictCallbackMixin

        self._processors: dict[int, ProcessorNode[TInput, TOutput]] = {
            id: processor_factory()
            for id in range(num_tracks)
        }

        self._output_callbacks: set[Output_Callback] = set()
        self._callback_lock = Lock()

    def set(self, input_data_dict: dict[int, TInput]) -> None:
        """Set input data for processor(s)."""

        for id, input_data in input_data_dict.items():
            self._processors[id].set(input_data)

    def process(self, poses: PoseDict) -> dict[int, TOutput]:
        """Process poses to produce derived outputs. """

        output_data_dict: dict[int, TOutput] = {}

        for id, pose in poses.items():
            try:
                processor = self._processors[id]
                if processor.is_ready():
                    output_data_dict[id] = processor.process(pose)
            except Exception as e:
                print(f"ProcessorTracker: Error processing pose {id}: {e}")
                print_exc()

        self._notify_output_callbacks(output_data_dict)
        self._notify_pose_dict_callbacks(poses)

        return output_data_dict

    def reset(self) -> None:
        """Reset all processors."""
        for processor in self._processors.values():
            processor.reset()

    def reset_at(self, id: int) -> None:
        """Reset processor for a specific pose ID."""
        if id in self._processors:
            self._processors[id].reset()

    def _notify_output_callbacks(self, output: TOutputDict) -> None:
        """Emit callbacks with output of type TOutput."""
        with self._callback_lock:
            for callback in self._output_callbacks:
                try:
                    callback(output)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_output_callback(self, callback: Output_Callback) -> None:
        """Register output callback."""
        with self._callback_lock:
            self._output_callbacks.add(callback)

    def remove_output_callback(self, callback: Output_Callback) -> None:
        """Unregister output callback."""
        with self._callback_lock:
            self._output_callbacks.discard(callback)