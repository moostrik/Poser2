from typing import TypeAlias

import torch


ImageDict: TypeAlias = dict[int, torch.Tensor]
