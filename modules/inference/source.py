from typing import TypeAlias

import torch


ImageDict: TypeAlias = dict[int, torch.Tensor]
TimestampDict: TypeAlias = dict[int, float]     # each camera's image's capture time (`CameraFrame.timestamp`)
