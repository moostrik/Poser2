"""Dummy — a figure with set joints standing in the room, entering the pose pipeline as a pose.

The dummy stands in for a person while the pose instrument is judged (``docs/POSE_INSTRUMENT.md``,
*The dummy*): a standing figure of the pipeline's 17 landmarks whose joints are set by the
``PI.dummy`` settings, each joint's degrees the angle the angle extractor reads at it when the
figure is upright (the shoulder 0 hanging, 90 across, 180 up, 270 out; the elbow 180 straight,
0 folded; the hip 180 standing; the knee 180 straight), and whose frame enters the LERP stage
before the filters, so it is extracted, drawn, heard in Max and lit exactly as a person is.
The torso leans the upper body over standing legs, as a person leans: the arms still read as set,
the hips read off by the lean. Only the joints and the torso are set: the leg deviation and the
body bend are the pipeline's, derived from the figure as for a person, and the pipeline's readings
of its poses are what the angle calibrator is read against. Its poses are named in
``data/poses.json``, seeded with the rows of the design's pose results; a change of any measure
morphs over ``morph`` seconds, the shortest way round for the azimuth and the joints.
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

from modules.pose.features import Azimuth, BBox, Points2D, PointLandmark
from modules.pose.frame import Frame, FrameDict, FrameDictCallbackMixin
from modules.pose.nodes import AngleCalibrator, AngleCalibratorSettings, AngleExtractor, AngleExtractorSettings
from modules.settings import BaseSettings, Field, Widget
from modules.utils import Rect

logger = logging.getLogger(__name__)


class DummySettings(BaseSettings):
    """The dummy: where it stands and how its joints are turned, in degrees from neutral."""
    enabled:        Field[bool]      = Field(False, description="Put the dummy in the pose pipeline")
    azimuth:        Field[float]     = Field(180.0, min=0.0,   max=360.0, step=1.0, description="Where the dummy stands (deg)")
    torso:          Field[float]     = Field(0.0,   min=-90.0, max=90.0,  step=1.0, description="Upper body leaned over standing legs, positive to image right (deg)")
    left_shoulder:  Field[float]     = Field(0.0,   min=0.0,   max=360.0, step=1.0, description="Shoulder angle as the extractor reads it: 0 hanging, 90 across, 180 up, 270 out (deg)", newline=True)
    right_shoulder: Field[float]     = Field(0.0,   min=0.0,   max=360.0, step=1.0, description="Shoulder angle as the extractor reads it: 0 hanging, 90 across, 180 up, 270 out (deg)")
    left_elbow:     Field[float]     = Field(180.0, min=0.0,   max=360.0, step=1.0, description="Elbow angle as the extractor reads it: 180 straight, 0 folded (deg)")
    right_elbow:    Field[float]     = Field(180.0, min=0.0,   max=360.0, step=1.0, description="Elbow angle as the extractor reads it: 180 straight, 0 folded (deg)")
    left_hip:       Field[float]     = Field(180.0, min=0.0,   max=360.0, step=1.0, description="Hip angle as the extractor reads it: 180 standing, 90 leg out level (deg)", newline=True)
    right_hip:      Field[float]     = Field(180.0, min=0.0,   max=360.0, step=1.0, description="Hip angle as the extractor reads it: 180 standing, 90 leg out level (deg)")
    left_knee:      Field[float]     = Field(180.0, min=0.0,   max=360.0, step=1.0, description="Knee angle as the extractor reads it: 180 straight, 90 bent (deg)")
    right_knee:     Field[float]     = Field(180.0, min=0.0,   max=360.0, step=1.0, description="Knee angle as the extractor reads it: 180 straight, 90 bent (deg)")
    morph:          Field[float]     = Field(0.5,   min=0.0,   max=10.0,  step=0.05, description="A change morphs over this time (s); 0 at once", newline=True)
    poses:          Field[list[str]] = Field([""], access=Field.READ, visible=False, description="Saved pose names")
    pose:           Field[str]       = Field("", widget=Widget.text_select, options=poses, description="Load a saved pose")
    name:           Field[str]       = Field("", widget=Widget.input, description="Name to save the pose under")
    save:           Field[bool]      = Field(False, widget=Widget.button, description="Save the pose under the name")


@dataclass(frozen=True)
class Measures:
    """The dummy's measures, in the settings' units: degrees, each joint's the angle the extractor
    reads at it. The defaults are standing with the arms hanging: 0 at the shoulders, 180 (straight)
    at the elbows, hips and knees."""
    azimuth:        float = 180.0
    torso:          float = 0.0
    left_shoulder:  float = 0.0
    right_shoulder: float = 0.0
    left_elbow:     float = 180.0
    right_elbow:    float = 180.0
    left_hip:       float = 180.0
    right_hip:      float = 180.0
    left_knee:      float = 180.0
    right_knee:     float = 180.0


MEASURES: tuple[str, ...] = tuple(f.name for f in fields(Measures))
NEUTRAL = 'neutral'                                               # the saved pose the dummy starts in
CIRCULAR: frozenset[str] = frozenset(MEASURES) - {'torso'}


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
# The point the extractor measures each joint's angle from (its first keypoint), joints nearer the
# torso first, so a joint's chain carries the joints after it. The legs are aimed against the
# upright torso line, before the torso leans; the arms against the leaning one.
_LEGS: tuple[tuple[str, PointLandmark], ...] = (
    ('left_hip',       _P.left_shoulder), ('right_hip',      _P.right_shoulder),
    ('left_knee',      _P.left_hip),      ('right_knee',     _P.right_hip),
)
_ARMS: tuple[tuple[str, PointLandmark], ...] = (
    ('left_shoulder',  _P.left_hip),      ('right_shoulder', _P.right_hip),
    ('left_elbow',     _P.left_shoulder), ('right_elbow',    _P.right_shoulder),
)


class Dummy(FrameDictCallbackMixin):
    """The dummy in the pose pipeline; see the module docstring."""

    def __init__(self, settings: DummySettings, angle_extractor: AngleExtractorSettings,
                 angle_calibrator: AngleCalibratorSettings, track_id: int, poses_path: Path) -> None:
        super().__init__()
        self._settings = settings
        self._extractor = AngleExtractor(angle_extractor)
        self._calibrator = AngleCalibrator(angle_calibrator)      # the dummy enters after the stages that run these
        self._aspect_ratio = float(angle_extractor.aspect_ratio)
        self._track_id = track_id
        self._poses_path = Path(poses_path)
        self._poses: dict[str, dict[str, float]] = self._read_poses()
        settings.poses = list(self._poses) or [""]
        if NEUTRAL in self._poses:                                # start in neutral, not where the preset left it
            settings.pose = NEUTRAL
            self._on_pose(NEUTRAL)
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
        points = self.points(m, self._aspect_ratio)
        low, high = points.values.min(axis=0), points.values.max(axis=0)
        tight = Rect(float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1]))
        box = Rect(0.0, 0.0, self._aspect_ratio, 1.0).aspect_fill(tight)     # around the pose, at the crop's aspect
        frame = Frame(self._track_id, 0, time_stamp=time.time(), features={
            Points2D: points,
            Azimuth: Azimuth.from_value(math.radians(m.azimuth)),
            BBox: BBox.from_rect(box),
        })
        frame = self._calibrator.process(self._extractor.process(frame))
        self._notify_frames_callbacks({**frames, self._track_id: frame})

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
    def points(m: Measures, aspect_ratio: float) -> Points2D:
        """The figure with its joints at the angles of ``m``, as the angle extractor measures
        them when the figure is upright: each joint's degrees are the signed angle from the
        segment above it (the torso line for the shoulder and the hip, the upper arm for the
        elbow, the thigh for the knee) to the limb below, the right side mirrored as the extractor
        mirrors it, so what is set is what the extractor reads. Each limb's chain is turned about
        its joint from where it points to where it should, the joints nearer the torso first: the
        legs against the upright torso line, then the upper body is leaned about the hip midpoint
        by the torso over the standing legs, then the arms against the leaning torso line. So a
        lean leaves the arms reading as set and moves the hips' reading by the lean, as a
        person's."""
        xy = np.array([_FIGURE[lm] for lm in PointLandmark], dtype=np.float64)
        Dummy._aim_joints(xy, m, _LEGS)
        pivot = (xy[_P.left_hip] + xy[_P.right_hip]) / 2.0
        Dummy._turn(xy, pivot, _UPPER, math.radians(m.torso))
        Dummy._aim_joints(xy, m, _ARMS)
        return Dummy._points2d(xy, aspect_ratio)

    @staticmethod
    def _aim_joints(xy: np.ndarray, m: Measures, joints: tuple[tuple[str, PointLandmark], ...]) -> None:
        for name, proximal in joints:
            joint, chain, left = _JOINTS[name]
            angle = math.radians(getattr(m, name)) * (1.0 if left else -1.0)
            above = Dummy._unit(xy[proximal] - xy[joint])              # the extractor's first vector
            c, s = math.cos(angle), math.sin(angle)
            Dummy._aim(xy, joint, chain, np.array([c * above[0] - s * above[1], s * above[0] + c * above[1]]))

    @staticmethod
    def _aim(xy: np.ndarray, joint: PointLandmark, chain: list[PointLandmark], direction: np.ndarray) -> None:
        """Turn the chain about the joint so its first link points along ``direction``."""
        current = Dummy._unit(xy[chain[0]] - xy[joint])
        angle = math.atan2(current[0] * direction[1] - current[1] * direction[0], float(current @ direction))
        Dummy._turn(xy, xy[joint].copy(), chain, angle)

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        return v / float(np.linalg.norm(v))

    @staticmethod
    def _turn(xy: np.ndarray, pivot: np.ndarray, landmarks: list[PointLandmark], angle: float) -> None:
        """Turn ``landmarks`` about ``pivot`` by ``angle`` (rad) in image coordinates (y down)."""
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
