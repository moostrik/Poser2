"""Rolling GPU buffer for temporal feature accumulation.

Accumulates BaseScalarFeature values from pose frames into a rolling temporal
buffer on GPU, with separate validity mask derived from feature scores.
"""

from dataclasses import dataclass
import threading

import numpy as np
import torch

from modules.ConfigBase import ConfigBase, config_field
from modules.pose.callback.mixins import TypedCallbackMixin
from modules.pose.Frame import FrameDict, FrameField
from modules.pose.features import BaseScalarFeature


# Type alias for buffer output: (values, mask) tensors
BufferOutput = tuple[torch.Tensor, torch.Tensor]


@dataclass
class TensorWindowConfig(ConfigBase):
    """Configuration for TensorWindow.

    The feature_length is derived from frame_field.get_length().
    """
    frame_field: FrameField = config_field(FrameField.angle_vel, fixed=True, description="Frame field to extract (must be a BaseScalarFeature)")
    window_size: int = config_field(100, fixed=True, min=16, max=1024, description="Temporal window size (number of frames to keep)")
    num_tracks: int = config_field(3,fixed=True, min=1, max=16, description="Maximum number of concurrent tracks")
    fill_value: float = config_field(0.0, min=-1000.0, max=1000.0, description="Fill value for missing/invalid data" )


class TensorWindow(TypedCallbackMixin[BufferOutput]):
    """GPU-resident circular buffer with async callback dispatch.

    Maintains a sliding window of feature values per track on GPU using a
    circular buffer with CPU write index. Each frame writes to the next
    column, avoiding expensive GPU roll/clone operations.
    Missing tracks or invalid feature elements get fill_value with mask=0.

    Architecture:
    - submit() copies numpy data to pinned staging (CPU-only, ~5-20μs)
    - Background thread handles GPU transfer, reordering, and callbacks
    - Pose processing thread never blocks on GPU operations

    Internal buffer shape: (num_tracks, window_size, feature_length)
    Callback output shape: same, but time-ordered and contiguous
    - Index 0 along window_size axis is the oldest frame
    - Index -1 (window_size-1) is the most recent frame
    - Mask is float32 derived from feature scores (0.0 = invalid, >0 = valid)

    Usage:
        config = RollingFeatureBufferConfig(
            frame_field=FrameField.angle_vel,
            window_size=64,
            num_tracks=4
        )
        buffer = RollingFeatureBuffer(config)
        buffer.add_callback(my_callback)  # Receives (values, mask) tuple
        buffer.start()  # Start async worker thread

        # On each frame:
        buffer.submit(poses)  # Returns immediately after CPU copy

        # Clean shutdown:
        buffer.stop()
    """

    def __init__(self, config: TensorWindowConfig) -> None:
        super().__init__()

        self._config = config
        self._frame_field = config.frame_field
        self._window_size = config.window_size
        self._num_tracks = config.num_tracks
        self._fill_value = config.fill_value

        # Derive feature length from FrameField
        self._feature_length = self._frame_field.get_length()

        # Preallocate GPU buffers
        # Shape: (num_tracks, window_size, feature_length)
        self._values = torch.full(
            (self._num_tracks, self._window_size, self._feature_length),
            fill_value=self._fill_value,
            dtype=torch.float32,
            device='cuda'
        )

        self._mask = torch.zeros(
            (self._num_tracks, self._window_size, self._feature_length),
            dtype=torch.float32,
            device='cuda'
        )

        # Circular buffer write index (CPU-side, modified by async thread)
        self._write_idx: int = 0

        # Pinned staging buffers for ALL tracks (written in submit, read by thread)
        self._staging_values = torch.empty(
            (self._num_tracks, self._feature_length),
            dtype=torch.float32,
            pin_memory=True
        )
        self._staging_mask = torch.empty(
            (self._num_tracks, self._feature_length),
            dtype=torch.float32,
            pin_memory=True
        )
        self._staging_valid = torch.zeros(
            (self._num_tracks,),
            dtype=torch.bool,
            pin_memory=True
        )

        # Track previous frame state (for detecting pose loss in worker)
        self._prev_valid = torch.zeros(
            (self._num_tracks,),
            dtype=torch.bool
        )

        # Reset flag (set by reset(), consumed by worker)
        self._reset_all_flag = False

        # Async notification thread
        self._notify_event = threading.Event()
        self._shutdown_flag = False
        self._started = False
        self._notify_thread: threading.Thread | None = None

        # Track frame count for debugging
        self._frame_count = 0

    def start(self) -> None:
        """Start the async notification thread.

        Must be called before submit() to enable async processing.
        Safe to call multiple times (subsequent calls are no-op).
        """
        if self._started:
            return

        self._shutdown_flag = False
        self._notify_thread = threading.Thread(
            target=self._process,
            name="FeatureBufferNotify",
            daemon=False
        )
        self._notify_thread.start()
        self._started = True

    def stop(self) -> None:
        """Stop the async notification thread.

        Waits for thread to finish current work before returning.
        Safe to call multiple times or before start().
        """
        if not self._started:
            return

        self._shutdown_flag = True
        self._notify_event.set()

        if self._notify_thread is not None:
            self._notify_thread.join(timeout=1.0)
            if self._notify_thread.is_alive():
                print(f"WARNING: {self._notify_thread.name} did not exit cleanly within timeout")

        self._started = False

    def submit(self, poses: FrameDict) -> None:
        """Submit new frame data to staging buffer and signal async thread.

        ONLY performs CPU-side copies to pinned staging memory (~5-20μs).
        Background thread handles GPU transfer, reordering, and callbacks.

        Note: Call start() before first submit() to enable async processing.

        Args:
            poses: Dictionary of track_id -> Frame
        """
        if not self._started:
            raise RuntimeError("RollingFeatureBuffer not started. Call start() first.")

        # Copy all track data to pinned staging (CPU-side only)
        for track_id in range(self._num_tracks):
            if track_id in poses:
                frame = poses[track_id]
                feature: BaseScalarFeature = frame.get_feature(self._frame_field)

                # Direct copy to pinned staging memory (NaN replaced with fill_value)
                np.copyto(self._staging_values[track_id].numpy(), np.nan_to_num(feature.values, nan=self._fill_value))
                np.copyto(self._staging_mask[track_id].numpy(), feature.scores)
                self._staging_valid[track_id] = True
            else:
                self._staging_valid[track_id] = False

        # Signal async thread - done, return immediately
        self._notify_event.set()

    def reset(self) -> None:
        """Reset all buffer data to initial state.

        Sets flag for worker thread to perform reset on next cycle.
        Non-blocking - returns immediately.
        """
        if not self._started:
            return

        # Signal worker to reset
        self._reset_all_flag = True
        self._notify_event.set()

    def _process(self) -> None:
        """Background thread: GPU transfer, reorder, and notify callbacks.

        Runs continuously until shutdown_flag is set.
        Wakes on _notify_event signal from submit().
        """
        while not self._shutdown_flag:
            # Wait for signal from submit()
            self._notify_event.wait()
            self._notify_event.clear()

            if self._shutdown_flag:
                break

            try:
                with torch.no_grad():
                    # Handle reset request first (before processing new frame)
                    if self._reset_all_flag:
                        self._values.fill_(self._fill_value)
                        self._mask.zero_()
                        self._prev_valid.zero_()
                        self._write_idx = 0
                        self._frame_count = 0
                        self._reset_all_flag = False
                        continue  # Skip frame processing, wait for next submit

                    idx = self._write_idx

                    # Transfer from pinned staging to GPU
                    for track_id in range(self._num_tracks):
                        is_valid = self._staging_valid[track_id]
                        was_valid = self._prev_valid[track_id]

                        if is_valid:
                            # Direct GPU transfer (NaN already handled in submit)
                            self._values[track_id, idx, :].copy_(self._staging_values[track_id], non_blocking=True)
                            self._mask[track_id, idx, :].copy_(self._staging_mask[track_id], non_blocking=True)
                        elif was_valid:
                            # Pose just lost - reset entire track buffer (already all fill_value, no per-frame writes needed)
                            self._values[track_id].fill_(self._fill_value)
                            self._mask[track_id].zero_()
                        # else: track remains lost - buffer already full of fill_value from reset, no-op

                        # Update previous state
                        self._prev_valid[track_id] = is_valid

                    # Advance write index (circular)
                    self._write_idx = (idx + 1) % self._window_size
                    self._frame_count += 1

                    # Sync to ensure GPU writes complete
                    torch.cuda.synchronize()

                    # Build time-ordered output using torch.roll
                    # After write, write_idx points to oldest data (next to be overwritten)
                    # Roll to put oldest at index 0
                    write_idx = self._write_idx
                    if write_idx == 0:
                        # Buffer already in order - clone to create snapshot
                        ordered_values = self._values.clone()
                        ordered_mask = self._mask.clone()
                    else:
                        # Roll: shift by -write_idx to put oldest at position 0
                        # Equivalent to cat([write_idx:], [:write_idx]) but single op
                        ordered_values = torch.roll(self._values, shifts=-write_idx, dims=1).contiguous()
                        ordered_mask = torch.roll(self._mask, shifts=-write_idx, dims=1).contiguous()

                    # Notify callbacks with time-ordered contiguous tensors (new allocation each frame)
                    self._notify_callbacks((ordered_values, ordered_mask))

            except Exception as e:
                print(f"RollingFeatureBuffer notify worker error: {e}")

    def __repr__(self) -> str:
        return (
            f"RollingFeatureBuffer("
            f"field={self._frame_field.name}, "
            f"tracks={self._num_tracks}, "
            f"window={self._window_size}, "
            f"features={self._feature_length}, "
            f"frames={self._frame_count})"
        )
