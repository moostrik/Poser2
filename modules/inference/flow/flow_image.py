from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch


@dataclass(frozen=True)
class FlowImage:
    """GPU-resident optical flow result for a single tracked pose.

    Produced by FlowBatchExtractor after RAFT inference.
    Format: (2, H, W) float32 where [0] = x-displacement, [1] = y-displacement.

    Attributes:
        track_id: Tracklet identifier
        flow: Optical flow field on GPU (2, H, W) float32
    """
    track_id: int
    flow: torch.Tensor


FlowImageDict: TypeAlias = dict[int, FlowImage]
FlowImageCallback: TypeAlias = Callable[[FlowImageDict], None]
