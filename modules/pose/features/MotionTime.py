from __future__ import annotations

import numpy as np
from modules.pose.features.base.BaseFeature import POSITIVE_RANGE
from modules.pose.features.base.SingleValue import SingleValue


class MotionTime(SingleValue):
    """Non-negative accumulated motion value."""

    @classmethod
    def range(cls) -> tuple[float, float]:
        return POSITIVE_RANGE
