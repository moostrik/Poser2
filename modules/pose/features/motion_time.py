from __future__ import annotations

import numpy as np
from .base.base_feature import POSITIVE_RANGE
from .base.single_value import SingleValue


class MotionTime(SingleValue):
    """Non-negative accumulated motion value."""

    @classmethod
    def range(cls) -> tuple[float, float]:
        return POSITIVE_RANGE
