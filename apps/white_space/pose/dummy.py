"""Dummy — a figure with set joints standing in the room, entering the pose pipeline as a pose.

The dummy stands in for a person while the pose instrument is judged (``docs/POSE_INSTRUMENT.md``,
*The dummy*): a standing figure of the pipeline's 17 landmarks whose joints are turned by the
``PI.dummy`` settings, in degrees from neutral, and whose frame enters the LERP stage before the
filters, so it is extracted, drawn, heard in Max and lit exactly as a person is. Only the joints
and the torso are set: the leg deviation and the body bend are the pipeline's, derived from the
figure as for a person. Its poses are named in ``data/poses.json``, seeded with the design's pose
results; a change of any measure morphs over ``morph`` seconds, the shortest way round for the
azimuth and the arms.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pytweening

from modules.pose.features import Angles, AngleLandmark, Azimuth, BBox, Points2D, PointLandmark
from modules.pose.frame import Frame, FrameDict, FrameDictCallbackMixin
from modules.pose.nodes import AngleExtractor, AngleExtractorSettings
from modules.settings import BaseSettings, Field, Widget
from modules.utils import Rect

logger = logging.getLogger(__name__)


class DummySettings(BaseSettings):
    """The dummy: where it stands and how its joints are turned, in degrees from neutral."""
    enabled:        Field[bool]      = Field(False, description="Put the dummy in the pose pipeline")
    azimuth:        Field[float]     = Field(180.0, min=0.0,   max=360.0, step=1.0, description="Where the dummy stands (deg)")
    torso:          Field[float]     = Field(0.0,   min=-90.0, max=90.0,  step=1.0, description="Upper body from upright, positive to image right (deg)")
    left_shoulder:  Field[float]     = Field(0.0,   min=0.0,   max=360.0, step=1.0, description="Arm about the shoulder: 0 hanging, 90 level, 180 up (deg)", newline=True)
    right_shoulder: Field[float]     = Field(0.0,   min=0.0,   max=360.0, step=1.0, description="Arm about the shoulder: 0 hanging, 90 level, 180 up (deg)")
    left_elbow:     Field[float]     = Field(0.0,   min=0.0,   max=360.0, step=1.0, description="Forearm about the elbow: 0 straight, 180 folded (deg)")
    right_elbow:    Field[float]     = Field(0.0,   min=0.0,   max=360.0, step=1.0, description="Forearm about the elbow: 0 straight, 180 folded (deg)")
    left_hip:       Field[float]     = Field(0.0,   min=0.0,   max=120.0, step=1.0, description="Leg about the hip: 0 standing, 90 raised level (deg)", newline=True)
    right_hip:      Field[float]     = Field(0.0,   min=0.0,   max=120.0, step=1.0, description="Leg about the hip: 0 standing, 90 raised level (deg)")
    left_knee:      Field[float]     = Field(0.0,   min=0.0,   max=150.0, step=1.0, description="Shin about the knee: 0 straight, 150 folded (deg)")
    right_knee:     Field[float]     = Field(0.0,   min=0.0,   max=150.0, step=1.0, description="Shin about the knee: 0 straight, 150 folded (deg)")
    morph:          Field[float]     = Field(0.5,   min=0.0,   max=10.0,  step=0.05, description="A change morphs over this time (s); 0 at once", newline=True)
    poses:          Field[list[str]] = Field([""], access=Field.READ, visible=False, description="Saved pose names")
    pose:           Field[str]       = Field("", widget=Widget.text_select, options=poses, description="Load a saved pose")
    name:           Field[str]       = Field("", widget=Widget.input, description="Name to save the pose under")
    save:           Field[bool]      = Field(False, widget=Widget.button, description="Save the pose under the name")


@dataclass(frozen=True)
class Measures:
    """The dummy's measures, in the settings' units: degrees."""
    azimuth:        float = 180.0
    torso:          float = 0.0
    left_shoulder:  float = 0.0
    right_shoulder: float = 0.0
    left_elbow:     float = 0.0
    right_elbow:    float = 0.0
    left_hip:       float = 0.0
    right_hip:      float = 0.0
    left_knee:      float = 0.0
    right_knee:     float = 0.0


MEASURES: tuple[str, ...] = tuple(f.name for f in fields(Measures))
CIRCULAR: frozenset[str] = frozenset(('azimuth', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'))


def dummy_id(num_players: int) -> int:
    """The dummy's track id: its own, between the live players (0 … num_players − 1) and the
    ghosts, which start one above it."""
    return num_players

_P = PointLandmark
# A standing figure in square-pixel units, centred on x = 0.5, its sides named as the pipeline
# names people's: the left landmarks on image-right. A person fills the crop as this does after
# the aspect squeeze: four fifths of its height, the shoulders a quarter of its width.
_FIGURE: dict[PointLandmark, tuple[float, float]] = {
    _P.nose:           (0.50, 0.10),
    _P.left_eye:       (0.53, 0.08),
    _P.right_eye:      (0.47, 0.08),
    _P.left_ear:       (0.57, 0.10),
    _P.right_ear:      (0.43, 0.10),
    _P.left_shoulder:  (0.64, 0.30),
    _P.right_shoulder: (0.36, 0.30),
    _P.left_elbow:     (0.67, 0.52),
    _P.right_elbow:    (0.33, 0.52),
    _P.left_wrist:     (0.69, 0.72),
    _P.right_wrist:    (0.31, 0.72),
    _P.left_hip:       (0.59, 0.72),
    _P.right_hip:      (0.41, 0.72),
    _P.left_knee:      (0.59, 0.97),
    _P.right_knee:     (0.41, 0.97),
    _P.left_ankle:     (0.59, 1.20),
    _P.right_ankle:    (0.41, 1.20),
}
# The upper body: what turns with the torso about the hip midpoint.
_UPPER: list[PointLandmark] = [
    _P.nose, _P.left_eye, _P.right_eye, _P.left_ear, _P.right_ear,
    _P.left_shoulder, _P.right_shoulder, _P.left_elbow, _P.right_elbow, _P.left_wrist, _P.right_wrist,
]
# Each joint: the point it turns about, the chain that turns with it, and whether it is on the left.
_JOINTS: dict[str, tuple[PointLandmark, list[PointLandmark], bool]] = {
    'left_shoulder':  (_P.left_shoulder,  [_P.left_elbow, _P.left_wrist],   True),
    'right_shoulder': (_P.right_shoulder, [_P.right_elbow, _P.right_wrist], False),
    'left_elbow':     (_P.left_elbow,     [_P.left_wrist],                  True),
    'right_elbow':    (_P.right_elbow,    [_P.right_wrist],                 False),
    'left_hip':       (_P.left_hip,       [_P.left_knee, _P.left_ankle],    True),
    'right_hip':      (_P.right_hip,      [_P.right_knee, _P.right_ankle],  False),
    'left_knee':      (_P.left_knee,      [_P.left_ankle],                  True),
    'right_knee':     (_P.right_knee,     [_P.right_ankle],                 False),
}
# The joints in the order they are turned: a joint's chain carries the joints after it.
_LEVELS: tuple[tuple[str, ...], ...] = (
    ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'),
    ('left_elbow', 'right_elbow', 'left_knee', 'right_knee'),
)


class Dummy(FrameDictCallbackMixin):
    """The dummy in the pose pipeline; see the module docstring."""

    def __init__(self, settings: DummySettings, angle_extractor: AngleExtractorSettings, track_id: int,
                 poses_path: Path) -> None:
        super().__init__()
        self._settings = settings
        self._extractor = AngleExtractor(angle_extractor)
        self._aspect_ratio = float(angle_extractor.aspect_ratio)
        self._track_id = track_id
        self._poses_path = Path(poses_path)
        self._poses: dict[str, dict[str, float]] = self._read_poses()
        settings.poses = list(self._poses) or [""]
        self._last: float | None = None
        self._target = self._current = self._start = self._from_settings()
        self._route: dict[str, float] = {n: 0.0 for n in MEASURES}
        self._elapsed = 0.0
        settings.bind(DummySettings.pose, self._on_pose)
        settings.bind(DummySettings.save, self._on_save)

    # -- The pipeline step ----------------------------------------------------------

    def process(self, frames: FrameDict) -> None:
        """The merge step before the LERP filters: the frames as they are, plus the dummy's."""
        now = time.monotonic()
        dt = 0.0 if self._last is None else now - self._last
        self._last = now
        if not self._settings.enabled:
            self._notify_frames_callbacks(frames)
            return
        m = self.update(dt)
        points = self.points(m, self._extractor, self._aspect_ratio)
        low, high = points.values.min(axis=0), points.values.max(axis=0)
        frame = Frame(self._track_id, 0, time_stamp=time.time(), features={
            Points2D: points,
            Azimuth: Azimuth.from_value(math.radians(m.azimuth)),
            BBox: BBox.from_rect(Rect(float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1]))),
        })
        self._notify_frames_callbacks({**frames, self._track_id: self._extractor.process(frame)})

    # -- The morph --------------------------------------------------------------------

    def update(self, dt: float) -> Measures:
        """The measures this tick: a change of the settings starts a morph from the measures being
        drawn, eased over ``morph`` seconds; circular measures take the shortest way round."""
        target = self._from_settings()
        if target != self._target:
            self._start, self._target, self._elapsed = self._current, target, 0.0
            self._route = {n: self._route_of(getattr(self._start, n), getattr(target, n), n in CIRCULAR) for n in MEASURES}
        else:
            self._elapsed += dt
        morph = self._settings.morph
        t = 1.0 if morph <= 0.0 else min(1.0, self._elapsed / morph)
        eased = pytweening.easeInOutSine(t)
        self._current = Measures(**{n: getattr(self._start, n) + self._route[n] * eased for n in MEASURES})
        return self._current

    @staticmethod
    def _route_of(start: float, target: float, circular: bool) -> float:
        """The way from ``start`` to ``target``: the shortest round the circle, with a tie
        (exactly opposite) going up through the front, the increasing way."""
        if not circular:
            return target - start
        d = (target - start) % 360.0
        return d - 360.0 if d > 180.0 else d

    def _from_settings(self) -> Measures:
        S = self._settings
        return Measures(**{n: float(getattr(S, n)) for n in MEASURES})

    # -- The figure -------------------------------------------------------------------

    @staticmethod
    def points(m: Measures, extractor: AngleExtractor, aspect_ratio: float) -> Points2D:
        """The figure with its joints turned to ``m``: the upper body about the hip midpoint by
        the torso, then each joint's chain by the difference between the wanted angle and what
        the extractor reads, so the pipeline reads back exactly the set degrees whatever the
        figure's own geometry. The dummy's degrees turn a joint outward, which the pipeline reads
        as the negative angle. Levels in order, a joint's chain carrying the joints after it."""
        xy = np.array([_FIGURE[lm] for lm in PointLandmark], dtype=np.float64)
        pivot = (xy[_P.left_hip] + xy[_P.right_hip]) / 2.0
        Dummy._turn(xy, pivot, _UPPER, math.radians(m.torso))
        for level in _LEVELS:
            read = extractor.process(Frame(0, 0, features={Points2D: Dummy._points2d(xy, aspect_ratio)}))[Angles].values
            for name in level:
                joint, chain, left = _JOINTS[name]
                delta = -math.radians(getattr(m, name)) - float(read[AngleLandmark[name]])
                Dummy._turn(xy, xy[joint].copy(), chain, delta if left else -delta)
        return Dummy._points2d(xy, aspect_ratio)

    @staticmethod
    def _turn(xy: np.ndarray, pivot: np.ndarray, landmarks: list[PointLandmark], angle: float) -> None:
        """Turn ``landmarks`` about ``pivot`` by ``angle`` (rad), counter-clockwise in image coordinates."""
        c, s = math.cos(angle), math.sin(angle)
        d = xy[landmarks] - pivot
        xy[landmarks] = pivot + d @ np.array([[c, s], [-s, c]])

    @staticmethod
    def _points2d(xy: np.ndarray, aspect_ratio: float) -> Points2D:
        """The figure as the pipeline's keypoints: y squeezed by the crop's aspect ratio."""
        values = (xy * np.array([1.0, aspect_ratio])).astype(np.float32)
        return Points2D(values, np.ones(len(PointLandmark), dtype=np.float32))

    # -- The poses --------------------------------------------------------------------

    def _read_poses(self) -> dict[str, dict[str, float]]:
        try:
            if self._poses_path.exists():
                return json.loads(self._poses_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            logger.error("Dummy poses at %s not read: %s", self._poses_path, e)
        return {}

    def _write_poses(self) -> None:
        tmp = self._poses_path.with_suffix('.tmp')
        try:
            tmp.write_text(json.dumps(self._poses, indent=2), encoding='utf-8')
            os.replace(tmp, self._poses_path)
        except OSError as e:
            logger.error("Dummy poses at %s not written: %s", self._poses_path, e)

    def _on_pose(self, name: str) -> None:
        values = self._poses.get(name)
        if values is None:
            return
        for n in MEASURES:
            if n in values:
                setattr(self._settings, n, float(values[n]))

    def _on_save(self, _=None) -> None:
        name = self._settings.name.strip()
        if not name:
            logger.warning("Dummy pose not saved: no name")
            return
        self._poses[name] = {n: float(getattr(self._settings, n)) for n in MEASURES}
        self._write_poses()
        self._settings.poses = list(self._poses)
        self._settings.pose = name
