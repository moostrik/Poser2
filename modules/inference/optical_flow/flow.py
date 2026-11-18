from typing import Callable, TypeAlias

import torch

FlowDict: TypeAlias = dict[int, torch.Tensor]       # per track id: the flow (2, H, W), x and y in the model's pixels
FlowCallback: TypeAlias = Callable[[FlowDict], None]
