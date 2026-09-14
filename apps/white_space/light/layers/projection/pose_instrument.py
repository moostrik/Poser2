"""PoseInstrument — the heart of the piece (design: ``docs/POSE_INSTRUMENT.md``; the layer:
``docs/LAYERS.md``, pose_instrument).

Each person stands in a dim blue **band** at their azimuth. Around them, mirror-symmetric and
masked by every band, lie full white and full blue **lines** drawn from their pose: per channel a
spatial LFO thresholded into lines (``LinePattern.lines``), whose five parameters (duty, interval,
harmonic, harmonic phase, phase) are each patched from one of six pose controls in the settings.
Nothing moves by itself: the lines change only as the pose does.

A person's pattern shows over ``reach`` each side; above ``sync_threshold`` a pair's reach grows
toward each other until each pattern reaches the partner, and overlapping patterns union. No line
or gap is narrower than ``min_feature``, except where a band or a reach edge cuts a line: lines
slide out from behind the band and into view. On the tick the playhead crosses a person
(``PlayheadCrossing``), all of that person's lines widen by ``hit_widen``.

The drawing math lives in class methods here and in ``LinePattern``, both hot-reloaded while the
app runs; enum values are compared through ``int`` because a reload redefines the enum classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, auto

import numpy as np

from modules.pose import features
from modules.settings import BaseSettings, Field, Group
from modules.utils import HotReloadMethods

from .._base_layer import ProjectionLayer, LayerSettings
from .._utilities import normalize_azimuth
from .line_pattern import LinePattern
from ...frame import Frame
from ....pose import PlayheadCrossing, PlayheadOffset, playhead_step


class PoseControl(IntEnum):
    """A pose value as a control in 0..1; a mirrored pose gives the same arm and bend controls."""
    CONSTANT   = 0        # always 1: the patch holds its high value
    LIFT       = auto()   # both arms raised: 0 hanging, 1 straight up
    ARM_SPLIT  = auto()   # one arm higher than the other
    BEND       = auto()   # both elbows bent
    BEND_SPLIT = auto()   # one elbow more bent than the other
    LEGS       = auto()   # LegDeviation
    TILT       = auto()   # torso lean: 0 one way, 0.5 upright, 1 the other


class PatchSettings(BaseSettings):
    """One pattern parameter: a pose control mapped linearly (after the curve) from low to high."""
    source: Field[PoseControl] = Field(PoseControl.CONSTANT,                     description="Pose control driving this parameter")
    low:    Field[float]       = Field(0.0, min=-1.0, max=1.0, step=0.01, description="Parameter at control 0")
    high:   Field[float]       = Field(0.0, min=-1.0, max=1.0, step=0.01, description="Parameter at control 1; CONSTANT holds this")
    curve:  Field[float]       = Field(1.0, min=0.25, max=4.0, step=0.05, description="Response exponent on the control")


class ChannelPatternSettings(BaseSettings):
    """One channel's line pattern: the interval range and five patched parameters (each 0..1)."""
    interval_min:   Field[float] = Field(4.0,  min=0.5, max=90.0, step=0.5, description="Line interval at patch value 0 (deg)")
    interval_max:   Field[float] = Field(24.0, min=0.5, max=90.0, step=0.5, description="Line interval at patch value 1 (deg)")
    harmonic_order: Field[int]   = Field(2,    min=2,   max=4,    step=1,   description="Sub-line LFO multiple of the interval")
    duty:           Group[PatchSettings] = Group(PatchSettings)
    interval:       Group[PatchSettings] = Group(PatchSettings)
    harmonic:       Group[PatchSettings] = Group(PatchSettings)
    harmonic_phase: Group[PatchSettings] = Group(PatchSettings)
    phase:          Group[PatchSettings] = Group(PatchSettings)


class PoseInstrumentSettings(LayerSettings):
    min_feature:     Field[float] = Field(2.0,  min=0.1, max=10.0,  step=0.1,  description="Narrowest line or gap the projection resolves (deg)")
    band_width:      Field[float] = Field(3.0,  min=0.1, max=36.0,  step=0.1,  description="Blue band width (deg, scaled by pose length)", newline=True)
    band_level:      Field[float] = Field(0.3,  min=0.0, max=1.0,   step=0.01, description="Blue band level")
    reach:           Field[float] = Field(45.0, min=0.0, max=180.0, step=0.5,  description="Pattern shown each side of a person (deg)", newline=True)
    sync_threshold:  Field[float] = Field(0.75, min=0.0, max=0.99,  step=0.01, description="Pair similarity above which patterns grow toward each other")
    hit_widen:       Field[float] = Field(1.0,  min=0.0, max=10.0,  step=0.1,  description="Line widening each side on a playhead crossing (deg)")
    attack_seconds:  Field[float] = Field(1.0,  min=0.0, max=10.0,  step=0.1,  description="Reach grows in after arrival (s)", newline=True)
    release_seconds: Field[float] = Field(1.5,  min=0.0, max=10.0,  step=0.1,  description="Reach and band fade after leaving (s)")
    white:           Group[ChannelPatternSettings] = Group(ChannelPatternSettings)
    blue:            Group[ChannelPatternSettings] = Group(ChannelPatternSettings)


@dataclass
class _Participant:
    """One person's input state (the six pose values), presence, reach and hit."""
    position:       float = 0.0     # normalized azimuth
    length:         float = 1.0     # BBox height (pose length)
    left_shoulder:  float = 0.0     # the four arm angles (rad, 0 = neutral)
    right_shoulder: float = 0.0
    left_elbow:     float = 0.0
    right_elbow:    float = 0.0
    legs:           float = 0.0     # LegDeviation [0, 1]
    tilt:           float = 0.0     # TorsoTilt [-1, 1]
    similarity:     np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    present:        bool  = False   # seen this tick
    hit:            bool  = False   # the playhead crosses this person this tick
    envelope:       float = 0.0     # presence 0..1 (attack / release)
    reach_left:     float = 0.0     # this tick's reach each side (normalized azimuth)
    reach_right:    float = 0.0


class PoseInstrument(ProjectionLayer):
    """The pose instrument; see the module docstring."""

    def __init__(self, resolution: int, config: PoseInstrumentSettings, board,
                 pose_stage: int, tick_interval: float) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage
        self._tick_interval = tick_interval
        self._participants: dict[int, _Participant] = {}
        self._crossing = PlayheadCrossing()
        self._distance = np.arange(resolution + 1, dtype=np.float64)     # px from a person
        self._white_lines = np.zeros(resolution, dtype=bool)
        self._blue_lines = np.zeros(resolution, dtype=bool)
        self._band = np.zeros(resolution, dtype=bool)
        self._band_level = np.zeros(resolution, dtype=np.float32)
        self._hot_reloaders = (HotReloadMethods(self.__class__, True), HotReloadMethods(LinePattern, True))

    def reset(self) -> None:
        """A fresh instrument (S6 entry): forget every participant and pass."""
        self._participants.clear()
        self._crossing.reset()

    # -- Per tick --------------------------------------------------------------

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        R = self.resolution
        self._update_participants(frame)
        if not self._participants:
            return
        self._set_reaches()

        min_px = max(1, int(round(P.min_feature / 360.0 * R)))
        widen_px = max(0, int(round(P.hit_widen / 360.0 * R)))
        white_lines, blue_lines = self._white_lines, self._blue_lines
        band, band_level = self._band, self._band_level
        white_lines.fill(False)
        blue_lines.fill(False)
        band.fill(False)
        band_level.fill(0.0)

        for p in self._participants.values():
            centre = int(round(p.position * R)) % R
            controls = self._controls(p)
            widen = widen_px if p.hit else 0
            self._draw_lines(white_lines, p, centre, controls, P.white, min_px, widen)
            self._draw_lines(blue_lines, p, centre, controls, P.blue, min_px, widen)
            self._draw_band(band, band_level, p, centre)

        # Each pattern is legible on its own; overlaps can only leave narrow gaps, so the union
        # fills those. The reach edges and the bands cut lines as they are: a line slides out.
        white_mask = LinePattern.fill_gaps(white_lines, min_px) & ~band
        blue_mask = LinePattern.fill_gaps(blue_lines, min_px) & ~band
        white += white_mask
        blue += np.where(band, band_level, blue_mask)

    # -- Participants ------------------------------------------------------------

    def _update_participants(self, frame: Frame) -> None:
        P = self._config
        dt = frame.tick.dt
        for p in self._participants.values():
            p.present = False
        offsets: dict[int, float] = {}
        for id, pose in self._board.get_frames(self._pose_stage).items():
            azimuth = pose[features.Azimuth].value
            if math.isnan(azimuth):
                continue
            p = self._participants.setdefault(id, _Participant())
            p.present = True
            p.position = normalize_azimuth(azimuth)
            height = pose[features.BBox][features.BBoxElement.height]
            p.length = height if not math.isnan(height) and height > 0.0 else p.length
            angles = pose[features.Angles].values
            p.left_shoulder  = self._value(angles[features.AngleLandmark.left_shoulder],  p.left_shoulder)
            p.right_shoulder = self._value(angles[features.AngleLandmark.right_shoulder], p.right_shoulder)
            p.left_elbow     = self._value(angles[features.AngleLandmark.left_elbow],     p.left_elbow)
            p.right_elbow    = self._value(angles[features.AngleLandmark.right_elbow],    p.right_elbow)
            p.legs = self._value(pose[features.LegDeviation].value, p.legs)
            p.tilt = self._value(pose[features.TorsoTilt].value, p.tilt)
            p.similarity = pose[features.Similarity].values
            offsets[id] = pose[PlayheadOffset].value

        step = playhead_step(frame.motor_command.beam_rpm, self._tick_interval)
        hits = self._crossing.update(offsets, step, 1)

        gone: list[int] = []
        for id, p in self._participants.items():
            p.hit = id in hits
            if p.present:
                p.envelope = 1.0 if P.attack_seconds <= 0.0 else min(1.0, p.envelope + dt / P.attack_seconds)
            else:
                p.envelope = 0.0 if P.release_seconds <= 0.0 else max(0.0, p.envelope - dt / P.release_seconds)
                if p.envelope <= 0.0:
                    gone.append(id)
        for id in gone:
            del self._participants[id]

    @staticmethod
    def _value(x: float, fallback: float) -> float:
        return fallback if math.isnan(x) else float(x)

    @staticmethod
    def _controls(p: _Participant) -> list[float]:
        """The six pose controls in 0..1, indexed by ``PoseControl``."""
        controls = [0.0] * len(PoseControl)
        left_shoulder, right_shoulder = abs(p.left_shoulder), abs(p.right_shoulder)
        left_elbow, right_elbow = abs(p.left_elbow), abs(p.right_elbow)
        controls[int(PoseControl.CONSTANT)]   = 1.0
        controls[int(PoseControl.LIFT)]       = min(1.0, (left_shoulder + right_shoulder) / math.tau)
        controls[int(PoseControl.ARM_SPLIT)]  = min(1.0, abs(left_shoulder - right_shoulder) / math.pi)
        controls[int(PoseControl.BEND)]       = min(1.0, (left_elbow + right_elbow) / math.tau)
        controls[int(PoseControl.BEND_SPLIT)] = min(1.0, abs(left_elbow - right_elbow) / math.pi)
        controls[int(PoseControl.LEGS)]       = min(1.0, max(0.0, p.legs))
        controls[int(PoseControl.TILT)]       = min(1.0, max(0.0, (p.tilt + 1.0) / 2.0))
        return controls

    @staticmethod
    def _patch(patch: PatchSettings, controls: list[float]) -> float:
        """The patched parameter: low → high over the control raised to the curve."""
        control = controls[int(patch.source)]
        return patch.low + (patch.high - patch.low) * control ** max(patch.curve, 0.01)

    # -- Reach and sync ------------------------------------------------------------

    def _set_reaches(self) -> None:
        """Each side's reach: ``reach`` scaled by presence, then grown toward every
        similarity-matched partner along the shorter arc, up to the partner's position."""
        P = self._config
        base = min(P.reach / 360.0, 0.5)
        for p in self._participants.values():
            p.reach_left = p.reach_right = base * p.envelope
        threshold = P.sync_threshold
        ids = list(self._participants)
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                sim = self._pair_similarity(id_a, id_b)
                if math.isnan(sim) or sim < threshold:
                    continue
                a, b = self._participants[id_a], self._participants[id_b]
                t = self._ease((sim - threshold) / max(1.0 - threshold, 1e-6)) * min(a.envelope, b.envelope)
                delta = self._signed_offset(a.position, b.position)
                amount = t * abs(delta)
                if delta >= 0.0:
                    a.reach_right = max(a.reach_right, amount)
                    b.reach_left = max(b.reach_left, amount)
                else:
                    a.reach_left = max(a.reach_left, amount)
                    b.reach_right = max(b.reach_right, amount)

    def _pair_similarity(self, id_a: int, id_b: int) -> float:
        """Mean of both directions' pairwise similarity (one side may be NaN)."""
        sims = []
        for me, other in ((id_a, id_b), (id_b, id_a)):
            row = self._participants[me].similarity
            if other < len(row) and not math.isnan(float(row[other])):
                sims.append(float(row[other]))
        return float(np.mean(sims)) if sims else float('nan')

    @staticmethod
    def _signed_offset(a: float, b: float) -> float:
        """Signed shortest offset a → b around the projection (normalized azimuth, in [-0.5, 0.5))."""
        return ((b - a + 0.5) % 1.0) - 0.5

    @staticmethod
    def _ease(t: float) -> float:
        return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))

    # -- Drawing ---------------------------------------------------------------------

    def _draw_lines(self, lines: np.ndarray, p: _Participant, centre: int, controls: list[float],
                    C: ChannelPatternSettings, min_px: int, widen: int) -> None:
        """OR one channel of one person's pattern into ``lines``: the half pattern mirrored about
        the centre pixel, widened on a hit, made legible, cut to each side's reach."""
        R = self.resolution
        left = min(int(round(p.reach_left * R)), R // 2)
        right = min(int(round(p.reach_right * R)), R - 1 - left)
        side = max(left, right)
        if side <= 0:
            return

        span = C.interval_min + (C.interval_max - C.interval_min) * min(max(self._patch(C.interval, controls), 0.0), 1.0)
        interval = max(span / 360.0 * R, 2.0 * min_px)
        duty = min(max(self._patch(C.duty, controls), 0.0), 1.0)
        harmonic = min(max(self._patch(C.harmonic, controls), 0.0), 1.0)
        harmonic_phase = self._patch(C.harmonic_phase, controls) % 1.0
        phase = self._patch(C.phase, controls) % 1.0

        # The strip runs a margin past the reach, so the circular morphology's wrap at its ends
        # never reaches a pixel that is shown.
        n = min(side + widen + min_px, R)
        half = LinePattern.lines(self._distance[:n + 1], interval, duty, harmonic,
                                 max(1, int(C.harmonic_order)), harmonic_phase, phase)
        strip = np.concatenate((half[:0:-1], half))                       # offsets −n … n
        if widen > 0:
            strip = LinePattern.dilate(strip, widen, widen)
        strip = LinePattern.legible(strip, min_px)[n - side:n + side + 1]    # offsets −side … side
        offsets = np.arange(-left, right + 1)
        idx = (centre + offsets) % R
        lines[idx] |= strip[offsets + side]

    def _draw_band(self, band: np.ndarray, band_level: np.ndarray, p: _Participant, centre: int) -> None:
        """Mark the person's band: masks every pattern, lit dim blue by presence."""
        P = self._config
        R = self.resolution
        half = int(round(P.band_width / 360.0 * R * (0.5 + 0.5 * p.length) / 2.0))
        idx = (centre + np.arange(-half, half + 1)) % R
        band[idx] = True
        band_level[idx] = np.maximum(band_level[idx], P.band_level * p.envelope)
