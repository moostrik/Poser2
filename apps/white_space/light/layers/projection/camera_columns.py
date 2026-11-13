"""A camera's columns: each uploaded image's band of rows reduced to one row on the GPU, read back without waiting.

`submit` queues, on this one's own CUDA stream after the image's `ready` event, the band's luminance reduced over its
rows (`SampleMethod`) and the row's copy into pinned memory, with an event recorded after it. `newest` hands back
the newest row whose event has passed, so a reader on the light tick never waits on the GPU; a row is about a tick
old when it is read."""

from enum import IntEnum, auto

import numpy as np
import torch

from modules.inference import GpuImage

PENDING: int = 3        # rows in flight per camera; beyond that an image is skipped (pinned memory from torch's host cache)


class SampleMethod(IntEnum):
    MAX = auto()
    MIN = auto()
    AVG = auto()


class CameraColumns:
    """See the module docstring. Called from one thread, the light tick's."""

    def __init__(self) -> None:
        self._stream: torch.cuda.Stream | None = None
        self._pending: dict[int, list[tuple[torch.Tensor, torch.cuda.Event]]] = {}      # per camera, oldest first
        self._newest: dict[int, np.ndarray] = {}
        self._submitted: dict[int, float] = {}          # each camera's newest image stamp submitted

    def submit(self, image: GpuImage, band_top: float, band_bottom: float, method: SampleMethod) -> None:
        """The image's rows `band_top` to `band_bottom` (shares of its height) reduced to one row by `method`, queued;
        an image already submitted, or an empty band, is skipped."""
        if self._submitted.get(image.cam_id) == image.timestamp:
            return
        height: int = image.image.shape[-2]
        row_top: int = max(0, int(band_top * height))
        row_bottom: int = min(height, int(band_bottom * height))
        if row_bottom <= row_top:
            return
        self._submitted[image.cam_id] = image.timestamp
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        pending: list[tuple[torch.Tensor, torch.cuda.Event]] = self._pending.setdefault(image.cam_id, [])
        self._collect(image.cam_id)
        if len(pending) >= PENDING:
            return                      # the GPU is behind: this image is skipped, the next one is taken
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(image.ready)
            band: torch.Tensor = image.image[:, row_top:row_bottom].float().mean(dim=0)     # the luminance, rows × columns
            if method == SampleMethod.MAX:
                row: torch.Tensor = band.amax(dim=0)
            elif method == SampleMethod.MIN:
                row = band.amin(dim=0)
            else:
                row = band.mean(dim=0)
            pinned: torch.Tensor = torch.empty(row.shape, dtype=torch.float32, pin_memory=True)
            pinned.copy_(row, non_blocking=True)
            done: torch.cuda.Event = torch.cuda.Event()
            done.record(self._stream)
        image.image.record_stream(self._stream)     # the uploader's image stays alive until this stream is done with it
        pending.append((pinned, done))

    def newest(self, cam_id: int) -> np.ndarray | None:
        """The newest row of the camera whose copy is done, float32 [0, 1] per column; None before the first."""
        self._collect(cam_id)
        return self._newest.get(cam_id)

    def _collect(self, cam_id: int) -> None:
        pending: list[tuple[torch.Tensor, torch.cuda.Event]] | None = self._pending.get(cam_id)
        while pending and pending[0][1].query():
            pinned, _ = pending.pop(0)
            self._newest[cam_id] = pinned.numpy()
