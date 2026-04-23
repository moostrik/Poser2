from __future__ import annotations

import numpy as np
from .base.BaseFeature import POSITIVE_RANGE
from .base.SingleValue import SingleValue


class Age(SingleValue):
    """Non-negative elapsed time since first detection."""

    @classmethod
    def range(cls) -> tuple[float, float]:
        return POSITIVE_RANGE
