"""Posture similarity — the one kernel that scores how alike two poses are.

``WindowSimilarity`` applies it to every pair at every window frame (as arrays); anything comparing two poses
on their own applies it to two ``Angles`` (``posture_similarity``). Per joint a Gaussian on the wrapped angle
difference, 1/e at one tolerance; the selected joints both poses have (``WindowSimilaritySettings.joints``)
aggregated (the harmonic mean is strict: one poor joint pulls it down); remapped into [0, 1]; times the
fraction of the selected joints compared, so a mostly-occluded pair cannot read as fully alike. Both callers
share this code, so their numbers cannot drift apart.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING, cast

import numpy as np

from modules.settings import BaseSettings, Field
from ..features import Angles, AngleLandmark, AggregationMethod, NormalizedScalarFeature

if TYPE_CHECKING:
    from .window_similarity import WindowSimilaritySettings


class JointSelectSettings(BaseSettings):
    """Which joints the posture similarity compares; an unchecked joint is neither compared nor counted.
    The fields are the ``AngleLandmark`` members, in their order."""
    left_shoulder:  Field[bool] = Field(True, description="Compare the left shoulder")
    right_shoulder: Field[bool] = Field(True, description="Compare the right shoulder")
    left_elbow:     Field[bool] = Field(True, description="Compare the left elbow")
    right_elbow:    Field[bool] = Field(True, description="Compare the right elbow")
    left_hip:       Field[bool] = Field(True, description="Compare the left hip")
    right_hip:      Field[bool] = Field(True, description="Compare the right hip")
    left_knee:      Field[bool] = Field(True, description="Compare the left knee")
    right_knee:     Field[bool] = Field(True, description="Compare the right knee")
    head:           Field[bool] = Field(True, description="Compare the head")


def joint_mask(joints: JointSelectSettings) -> np.ndarray:
    """The joints selected, as a bool ``(F,)`` in ``AngleLandmark`` order."""
    return np.array([bool(getattr(joints, landmark.name)) for landmark in AngleLandmark], dtype=bool)


class _JointAggregator(NormalizedScalarFeature):
    """Per-joint similarities as a feature, so NormalizedScalarFeature's aggregation methods (mean, harmonic
    mean, …) apply to them. Configured once with the joint count (class-level, idempotent)."""
    _joint_enum: type[IntEnum] | None = None

    @classmethod
    def enum(cls) -> type[IntEnum]:
        if cls._joint_enum is None:
            raise RuntimeError("_JointAggregator not configured")
        return cls._joint_enum

    @classmethod
    def configure(cls, num_joints: int) -> None:
        """Configure the aggregator with the number of joints."""
        if cls._joint_enum is None:
            cls._joint_enum = cast(type[IntEnum], IntEnum("JointIndex", {f"J{i}": i for i in range(num_joints)}))


def joint_similarity(diff: np.ndarray, tolerance: float) -> np.ndarray:
    """Per joint ``exp(-(Δ / tolerance)²)`` on the angle differences ``diff`` (radians, any shape), wrapped to
    the shortest way round; NaN where a joint is missing stays NaN. ``tolerance`` in radians: 1/e at one."""
    wrapped = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    return np.exp(-np.square(wrapped / tolerance))


def aggregate_joints(joint_sims: np.ndarray, method: AggregationMethod,
                     remap_low: float, remap_high: float, mask: np.ndarray | None = None) -> tuple[float, float]:
    """One pair's ``(value, coverage)`` from its per-joint similarities (F,), NaN for joints not compared: the
    aggregate over the joints present (``method``), remapped ``[remap_low, remap_high] → [0, 1]`` when the
    range is positive, times the fraction of joints compared. A joint ``mask`` (bool (F,)) leaves out is
    neither compared nor counted, so the coverage is over the selected joints. ``(NaN, 0.0)`` when no joint
    was compared."""
    joint_sims = np.asarray(joint_sims, dtype=np.float32)
    if mask is not None:
        joint_sims = np.where(mask, joint_sims, np.nan)
    valid = ~np.isnan(joint_sims)
    included = joint_sims.size if mask is None else int(np.sum(mask))
    if included == 0 or not valid.any():
        return math.nan, 0.0
    coverage = float(valid.sum() / included)
    _JointAggregator.configure(int(joint_sims.size))
    feature = _JointAggregator(values=joint_sims.copy(), scores=valid.astype(np.float32))
    value = feature.aggregate(method=method, min_confidence=0.0)
    if math.isnan(value):
        return math.nan, 0.0
    if remap_high > remap_low:
        value = float(np.clip((value - remap_low) / (remap_high - remap_low), 0.0, 1.0))
    return value * coverage, coverage


def posture_similarity(a: Angles, b: Angles, config: WindowSimilaritySettings) -> float:
    """How alike two poses are, scored exactly as ``WindowSimilarity`` scores a pair at ``window_length`` 1:
    the kernel on ``Angles.subtract`` with the config's ``angle_tolerance`` (°), ``joints``, ``method`` and
    remap. A joint with score 0 on either side is missing. NaN when the poses share no selected joint."""
    diff = a.subtract(b)
    values = np.where(diff.scores > 0.0, diff.values, np.nan)
    sims = joint_similarity(values, math.radians(config.angle_tolerance))
    value, _ = aggregate_joints(sims, config.method, config.remap_low, config.remap_high, joint_mask(config.joints))
    return value
