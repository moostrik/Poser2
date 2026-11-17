from dataclasses import dataclass

import torch


@dataclass
class RecurrentState:
    """RVM's recurrent states of one player, (1, C, H/2..H/16, W/2..W/16) each: its memory of the earlier frames."""
    r1: torch.Tensor
    r2: torch.Tensor
    r3: torch.Tensor
    r4: torch.Tensor
