"""PoseInstrument — the heart of the piece (design: ``docs/POSE_INSTRUMENT.md``; the layer:
``docs/LAYERS.md``, pose_instrument).

Each person stands in a dim blue **mask** at their azimuth. Around them, mirrored about their
centre pixel and cut to the **window** each side, their **pattern**: full white and full blue
lines, one spatial oscillator per colour thresholded into lines (``LinePattern.lines``): two
drawbars, the fundamental and the harmonic, over one interval shared by the colours, the blue
detuned and its registration inverted. The sources are pose features and nothing else;
``connect`` is the connections of the design's Part 3 written out, turning a person's measures
into the pattern's parameters. Every number it uses is a setting of the ``PI`` group
(``PoseInstrumentSettings``): the panel keeps the values, the code keeps the routing. The lines
follow the pose; on top of that they **drift** by themselves, each colour's phase advancing by
its ``drift`` per second, blue inward and white outward.

Above ``window.sync_threshold`` a pair's window grows toward each other until each pattern
reaches the partner, and overlapping patterns union. No line or gap is narrower than the visual
limit (``max_lines`` per revolution), except where a mask or a window edge cuts a line: lines slide
out from behind the mask and into view. On the ticks the playhead crosses a person
(``PlayheadCrossing``, ``events.hit_frames``) the person is marked: the mask flashes to
``mask.flash_brightness``, each colour's lines take the other colour by its ``tint`` (the central
fraction of every line; 1 is the swap), and a **push** raises the drift by ``push_strength`` and
lets it settle back over ``push_seconds``; the phase keeps what it gained.

The instrument has two kinds of numbers. Its **inputs** are what the pipeline delivers per person
(the arm angles, the leg deviation, the bend, the similarity, the playhead crossing), which the
dummy can fake. Its **parameters** are what the pattern is drawn from, the ``Pattern`` that
``connect`` returns. ``PI.override`` sets the parameters by hand: with ``on``, every ticked
parameter comes from the panel for everyone and its connection is skipped, the window can be held
open without a partner, and ``hit`` marks everyone as the playhead would.

``connect``, the drawing methods and ``LinePattern`` are hot-reloaded while the app runs; enum
values are compared through ``int`` because a reload redefines the enum classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from modules.pose import features
from modules.settings import BaseSettings, Field, Group, Widget
from modules.utils import HotReloadMethods

from .._base_layer import ProjectionLayer, LayerSettings
from .._utilities import normalize_azimuth, mask_half_width
from .line_pattern import LinePattern, Waveform
from ...frame import Frame
from ....pose import PlayheadCrossing, PlayheadOffset, playhead_step, DummySettings


# -- Settings: the PI root group ---------------------------------------------------------------

class OscillatorSettings(BaseSettings):
    """One colour's oscillator: what no measure is connected to."""
    waveform: Field[Waveform] = Field(Waveform.SINE,                          description="Wave thresholded into lines")
    cutoff:   Field[int]      = Field(2,   min=2,   max=4,   step=1,    description="Overtones per interval the filter passes")
    drift:    Field[float]    = Field(0.0, min=0.0, max=2.0, step=0.01, description="Own motion (intervals per second)")


class PatternSettings(BaseSettings):
    """The pattern's rests and ranges: what ``connect`` reads."""
    interval:             Field[float] = Field(14.0,            min=1.0,  max=90.0, step=0.5,  description="Interval at rest (deg)")
    octaves:              Field[float] = Field(1.0,             min=0.0,  max=3.0,  step=0.05, description="Pitch bend range each way (octaves)")
    detune:               Field[float] = Field(0.1,             min=0.0,  max=1.0,  step=0.01, description="Blue's interval past white's at full detune (fraction)")
    blue_phase:           Field[float] = Field(0.5,             min=-1.0, max=1.0,  step=0.01, description="Blue's rest phase from white's (intervals)")
    phase_range:          Field[float] = Field(0.5,             min=-1.0, max=1.0,  step=0.01, description="How far a measure moves the lines (intervals)")
    overtone_phase_range: Field[float] = Field(0.5,             min=-1.0, max=1.0,  step=0.01, description="How far a measure moves the overtone (intervals)")
    white:                Group[OscillatorSettings] = Group(OscillatorSettings)
    blue:                 Group[OscillatorSettings] = Group(OscillatorSettings)


class MaskSettings(BaseSettings):
    """The dim blue mask at the person."""
    width:            Field[float] = Field(3.0, min=0.1, max=36.0, step=0.1,  description="Mask width (deg)")
    brightness:       Field[float] = Field(0.3, min=0.0, max=1.0,  step=0.01, description="Mask blue level")
    playhead_at_mask: Field[float] = Field(0.3, min=0.0, max=1.0,  step=0.01, description="Playhead level inside a mask (fraction)")
    flash_brightness: Field[float] = Field(1.0, min=0.0, max=1.0,  step=0.01, description="Mask blue level on a hit")


class WindowSettings(BaseSettings):
    """The visible part of the pattern each side of the person."""
    width:          Field[float] = Field(45.0, min=0.0, max=180.0, step=0.5,  description="Window each side of a person (deg)")
    sync_threshold: Field[float] = Field(0.75, min=0.0, max=0.99,  step=0.01, description="Pair similarity above which windows open toward each other")


class EventSettings(BaseSettings):
    """The hit's mark and the push."""
    hit_frames:    Field[int]   = Field(1,   min=1,   max=3,    step=1,    description="Hit length: the ticks closest to the crossing, 1-3")
    tint_white:    Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, description="White lines take blue on a hit: 0 none, 1 swap")
    tint_blue:     Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, description="Blue lines take white on a hit: 0 none, 1 swap")
    push_strength: Field[float] = Field(0.0, min=0.0, max=5.0,  step=0.05, description="Drift added on a hit (intervals per second)")
    push_seconds:  Field[float] = Field(1.0, min=0.05, max=10.0, step=0.05, description="Push settle time (s)")


class PresenceSettings(BaseSettings):
    """The voice's envelope on the window."""
    attack_seconds:  Field[float] = Field(1.0, min=0.0, max=10.0, step=0.1, description="Window opens after arrival (s)")
    release_seconds: Field[float] = Field(1.5, min=0.0, max=10.0, step=0.1, description="Window closes and mask fades after leaving (s)")


class OverrideSettings(BaseSettings):
    """The pattern's parameters by hand: with ``on``, every ticked parameter comes from here for
    everyone and its connection is skipped; ``on`` never sets the other toggles."""
    on:                      Field[bool]  = Field(False,                                     description="Override the ticked parameters for everyone")
    hit:                     Field[bool]  = Field(False, widget=Widget.button,               description="Hit everyone on the next ticks")
    interval_on:             Field[bool]  = Field(True,                                      description="Interval from here", newline=True)
    interval:                Field[float] = Field(14.0, min=1.0,  max=90.0,  step=0.5,       description="Interval (deg)")
    detune_on:               Field[bool]  = Field(True,                                      description="Detune from here")
    detune:                  Field[float] = Field(0.0,  min=0.0,  max=1.0,   step=0.01,      description="Blue's interval past white's (fraction)")
    window_on:               Field[bool]  = Field(True,                                      description="Window from here, no sync growth")
    window:                  Field[float] = Field(45.0, min=0.0,  max=180.0, step=0.5,       description="Window each side of a person (deg)")
    white_fundamental_on:    Field[bool]  = Field(True,                                      description="White's fundamental from here", newline=True)
    white_fundamental:       Field[float] = Field(0.5,  min=0.0,  max=1.0,   step=0.01,      description="White's fundamental drawbar")
    white_harmonic_on:       Field[bool]  = Field(True,                                      description="White's harmonic from here")
    white_harmonic:          Field[float] = Field(0.0,  min=0.0,  max=1.0,   step=0.01,      description="White's harmonic drawbar")
    white_phase_on:          Field[bool]  = Field(True,                                      description="White's phase from here")
    white_phase:             Field[float] = Field(0.0,  min=-1.0, max=1.0,   step=0.01,      description="White's phase (intervals, positive outward)")
    white_overtone_phase_on: Field[bool]  = Field(True,                                      description="White's overtone phase from here")
    white_overtone_phase:    Field[float] = Field(0.0,  min=-1.0, max=1.0,   step=0.01,      description="White's overtone phase (intervals)")
    blue_fundamental_on:     Field[bool]  = Field(True,                                      description="Blue's fundamental from here", newline=True)
    blue_fundamental:        Field[float] = Field(0.5,  min=0.0,  max=1.0,   step=0.01,      description="Blue's fundamental drawbar")
    blue_harmonic_on:        Field[bool]  = Field(True,                                      description="Blue's harmonic from here")
    blue_harmonic:           Field[float] = Field(0.0,  min=0.0,  max=1.0,   step=0.01,      description="Blue's harmonic drawbar")
    blue_phase_on:           Field[bool]  = Field(True,                                      description="Blue's phase from here")
    blue_phase:              Field[float] = Field(0.5,  min=-1.0, max=1.0,   step=0.01,      description="Blue's phase (intervals, positive outward)")
    blue_overtone_phase_on:  Field[bool]  = Field(True,                                      description="Blue's overtone phase from here")
    blue_overtone_phase:     Field[float] = Field(0.0,  min=-1.0, max=1.0,   step=0.01,      description="Blue's overtone phase (intervals)")


class PoseInstrumentSettings(BaseSettings):
    """The ``PI`` root group: tweakable values only, no routing (``docs/POSE_INSTRUMENT.md``, Settings)."""
    max_lines: Field[int] = Field(90, min=10, max=360, step=1, description="Visual limit: lines per revolution, line and gap equal")
    pattern:   Group[PatternSettings]  = Group(PatternSettings)
    mask:      Group[MaskSettings]     = Group(MaskSettings)
    window:    Group[WindowSettings]   = Group(WindowSettings)
    events:    Group[EventSettings]    = Group(EventSettings)
    presence:  Group[PresenceSettings] = Group(PresenceSettings)
    override:  Group[OverrideSettings] = Group(OverrideSettings)
    dummy:     Group[DummySettings]    = Group(DummySettings)


# -- The pattern's parameters -------------------------------------------------------------------

@dataclass
class Oscillator:
    """One colour's connected parameters."""
    fundamental:    float   # drawbar 0..1
    harmonic:       float   # drawbar 0..1
    phase:          float   # intervals, positive outward
    overtone_phase: float   # intervals, within the interval


@dataclass
class Pattern:
    """What ``connect`` returns for one person: the shared interval and the two oscillators."""
    interval: float         # degrees, after the pitch bend
    detune:   float         # blue's interval is interval × (1 + detune)
    white:    Oscillator
    blue:     Oscillator


@dataclass
class _Player:
    """One person's measures, presence, window and hit."""
    position:       float = 0.0     # normalized azimuth
    left_shoulder:  float = 0.0     # the four arm angles (rad)
    right_shoulder: float = 0.0
    left_elbow:     float = 0.0
    right_elbow:    float = 0.0
    legs:           float = 0.0     # LegDeviation [0, 1]
    tilt:           float = 0.0     # TorsoTilt [-1, 1]
    symmetry:       np.ndarray = field(default_factory=lambda: np.zeros(len(features.SymmetryElement), dtype=np.float32))
    similarity:     np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    present:        bool  = False   # seen this tick
    hit:            bool  = False   # the playhead crosses this person this tick
    envelope:       float = 0.0     # presence 0..1 (attack / release)
    window_left:    float = 0.0     # this tick's window each side (normalized azimuth)
    window_right:   float = 0.0
    phase_white:    float = 0.0     # the drift's gain per colour (intervals, signed: white out, blue in)
    phase_blue:     float = 0.0
    push_white:     float = 0.0     # the push per colour: extra drift settling back (intervals per second)
    push_blue:      float = 0.0


class PoseInstrument(ProjectionLayer):
    """The pose instrument; see the module docstring."""

    def __init__(self, resolution: int, config: LayerSettings, instrument: PoseInstrumentSettings, board,
                 pose_stage: int, tick_interval: float) -> None:
        super().__init__(resolution, config, board)
        self._instrument = instrument
        self._pose_stage = pose_stage
        self._tick_interval = tick_interval
        self._players: dict[int, _Player] = {}
        self._crossing = PlayheadCrossing()
        self._distance = np.arange(resolution + 1, dtype=np.float64)     # px from a person
        self._white_lines = np.zeros(resolution, dtype=bool)
        self._blue_lines = np.zeros(resolution, dtype=bool)
        self._mask = np.zeros(resolution, dtype=bool)
        self._mask_level = np.zeros(resolution, dtype=np.float32)
        self._manual_hits = 0                                             # ticks left of a hit from the panel
        instrument.override.bind(OverrideSettings.hit, self._on_hit)
        self._hot_reloaders = (HotReloadMethods(self.__class__, True), HotReloadMethods(LinePattern, True))

    def reset(self) -> None:
        """A fresh instrument (S6 entry): forget every player and pass."""
        self._players.clear()
        self._crossing.reset()
        self._manual_hits = 0

    def _on_hit(self, _: bool) -> None:
        """The panel's hit button: everyone is hit for ``hit_frames`` ticks, from the next one."""
        self._manual_hits = int(self._instrument.events.hit_frames)

    # -- Per tick --------------------------------------------------------------

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._instrument
        R = self.resolution
        self._update_players(frame)
        if not self._players:
            return
        self._set_windows()

        min_px = self._min_px()
        white_lines, blue_lines = self._white_lines, self._blue_lines
        mask, mask_level = self._mask, self._mask_level
        white_lines.fill(False)
        blue_lines.fill(False)
        mask.fill(False)
        mask_level.fill(0.0)

        for p in self._players.values():
            centre = int(round(p.position * R)) % R
            pattern = self._override(self.connect(p))
            pattern.white.phase += p.phase_white
            pattern.blue.phase += p.phase_blue
            left, right, side = self._window_px(p)
            if side > 0:
                interval = pattern.interval / 360.0 * R
                white_strip = self._strip(pattern.white, interval, P.pattern.white, min_px, side)
                blue_strip = self._strip(pattern.blue, interval * (1.0 + pattern.detune), P.pattern.blue, min_px, side)
                if p.hit:
                    white_strip, blue_strip = self._tint(white_strip, blue_strip, min_px)
                self._paint(white_lines, white_strip, centre, left, right, side)
                self._paint(blue_lines, blue_strip, centre, left, right, side)
            self._draw_mask(mask, mask_level, p, centre)

        # Each pattern is visible on its own; overlaps can only leave narrow gaps, so the union
        # fills those. The window edges and the masks cut lines as they are: a line slides out.
        white_mask = LinePattern.fill_gaps(white_lines, min_px) & ~mask
        blue_mask = LinePattern.fill_gaps(blue_lines, min_px) & ~mask
        white += white_mask
        blue += np.where(mask, mask_level, blue_mask)

    def _min_px(self) -> int:
        """The visual limit in pixels: half a period of ``max_lines`` per revolution."""
        return max(1, int(round(self.resolution / (2.0 * self._instrument.max_lines))))

    # -- Players ------------------------------------------------------------

    def _update_players(self, frame: Frame) -> None:
        P = self._instrument
        dt = frame.tick.dt
        for p in self._players.values():
            p.present = False
        offsets: dict[int, float] = {}
        for id, pose in self._board.get_frames(self._pose_stage).items():
            azimuth = pose[features.Azimuth].value
            if math.isnan(azimuth):
                continue
            p = self._players.setdefault(id, _Player())
            p.present = True
            p.position = normalize_azimuth(azimuth)
            angles = pose[features.Angles].values
            p.left_shoulder  = self._value(angles[features.AngleLandmark.left_shoulder],  p.left_shoulder)
            p.right_shoulder = self._value(angles[features.AngleLandmark.right_shoulder], p.right_shoulder)
            p.left_elbow     = self._value(angles[features.AngleLandmark.left_elbow],     p.left_elbow)
            p.right_elbow    = self._value(angles[features.AngleLandmark.right_elbow],    p.right_elbow)
            p.legs = self._value(pose[features.LegDeviation].value, p.legs)
            p.tilt = self._value(pose[features.TorsoTilt].value, p.tilt)
            p.symmetry = np.where(np.isnan(pose[features.AngleSymmetry].values), p.symmetry, pose[features.AngleSymmetry].values)
            p.similarity = pose[features.Similarity].values
            offsets[id] = pose[PlayheadOffset].value

        step = playhead_step(frame.motor_command.beam_rpm, self._tick_interval)
        hits = self._crossing.update(offsets, step, int(P.events.hit_frames))
        manual = self._manual_hits > 0
        if manual:
            self._manual_hits -= 1

        gone: list[int] = []
        for id, p in self._players.items():
            p.hit = manual or id in hits
            self._advance_drift(p, dt)
            if p.present:
                p.envelope = 1.0 if P.presence.attack_seconds <= 0.0 else min(1.0, p.envelope + dt / P.presence.attack_seconds)
            else:
                p.envelope = 0.0 if P.presence.release_seconds <= 0.0 else max(0.0, p.envelope - dt / P.presence.release_seconds)
                if p.envelope <= 0.0:
                    gone.append(id)
        for id in gone:
            del self._players[id]

    @staticmethod
    def _value(x: float, fallback: float) -> float:
        return fallback if math.isnan(x) else float(x)

    def _advance_drift(self, p: _Player, dt: float) -> None:
        """The lines' own motion this tick: each colour's phase gains its drift and its push,
        white outward and blue inward. A hit raises the push by ``push_strength``; it settles
        back exponentially over ``push_seconds`` and the phase keeps what it gained."""
        P = self._instrument
        E = P.events
        if p.hit:
            p.push_white += E.push_strength
            p.push_blue += E.push_strength
        p.phase_white += (P.pattern.white.drift + p.push_white) * dt
        p.phase_blue -= (P.pattern.blue.drift + p.push_blue) * dt
        settle = math.exp(-dt / E.push_seconds) if E.push_seconds > 0.0 else 0.0
        p.push_white *= settle
        p.push_blue *= settle

    # -- Connections -----------------------------------------------------------------

    def connect(self, p: _Player) -> Pattern:
        """The first connections (``docs/POSE_INSTRUMENT.md``, Part 3) written out: a person's
        measures into the pattern's parameters, every number a setting of ``PI.pattern``.

        - left shoulder: the fundamental's drawbar, white 0 → 1 and blue 1 → 0
        - right shoulder: the harmonic's drawbar, white 0 → 1 and blue 1 → 0
        - left elbow: white's phase, 0 → ``phase_range`` outward
        - right elbow: white's overtone phase, 0 → ``overtone_phase_range``
        - leg deviation: the detune, 0 → ``detune``
        - body bend: the interval, ``interval`` bent by ± ``octaves``
        - the symmetries: unconnected
        """
        S = self._instrument.pattern
        fundamental = self._measure(p.left_shoulder)
        harmonic = self._measure(p.right_shoulder)
        white = Oscillator(fundamental, harmonic,
                           phase=self._measure(p.left_elbow) * S.phase_range,
                           overtone_phase=self._measure(p.right_elbow) * S.overtone_phase_range)
        blue = Oscillator(1.0 - fundamental, 1.0 - harmonic, phase=S.blue_phase, overtone_phase=0.0)
        interval = S.interval * 2.0 ** (min(max(p.tilt, -1.0), 1.0) * S.octaves)
        detune = S.detune * min(max(p.legs, 0.0), 1.0)
        return Pattern(interval, detune, white, blue)

    def _override(self, pattern: Pattern) -> Pattern:
        """The panel's parameters in place of the connected ones: with ``override.on``, each
        ticked parameter is replaced; otherwise the pattern is returned as connected."""
        O = self._instrument.override
        if not O.on:
            return pattern
        if O.interval_on:
            pattern.interval = O.interval
        if O.detune_on:
            pattern.detune = O.detune
        for osc, colour in ((pattern.white, 'white'), (pattern.blue, 'blue')):
            for name in ('fundamental', 'harmonic', 'phase', 'overtone_phase'):
                if getattr(O, f'{colour}_{name}_on'):
                    setattr(osc, name, getattr(O, f'{colour}_{name}'))
        return pattern

    @staticmethod
    def _measure(angle: float) -> float:
        """An angle as a measure 0..1: the pipeline's angles are calibrated so neutral is 0 and
        the raised pose π (``AngleCalibrator``), and π is the feature's range, not a tunable. The
        sign is the side of the body the limb passes, which the design gives no meaning, so the
        absolute is taken."""
        return min(max(abs(angle) / math.pi, 0.0), 1.0)

    # -- Window and sync ------------------------------------------------------------

    def _set_windows(self) -> None:
        """Each side's window: ``window.width`` scaled by presence, then grown toward every
        similarity-matched partner along the shorter arc, up to the partner's position. Held from
        the panel (``override.window``), it is that width by presence and does not grow."""
        P = self._instrument.window
        O = self._instrument.override
        held = O.on and O.window_on
        base = min((O.window if held else P.width) / 360.0, 0.5)
        for p in self._players.values():
            p.window_left = p.window_right = base * p.envelope
        if held:
            return
        threshold = P.sync_threshold
        ids = list(self._players)
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                sim = self._pair_similarity(id_a, id_b)
                if math.isnan(sim) or sim < threshold:
                    continue
                a, b = self._players[id_a], self._players[id_b]
                t = self._ease((sim - threshold) / max(1.0 - threshold, 1e-6)) * min(a.envelope, b.envelope)
                delta = self._signed_offset(a.position, b.position)
                amount = t * abs(delta)
                if delta >= 0.0:
                    a.window_right = max(a.window_right, amount)
                    b.window_left = max(b.window_left, amount)
                else:
                    a.window_left = max(a.window_left, amount)
                    b.window_right = max(b.window_right, amount)

    def _pair_similarity(self, id_a: int, id_b: int) -> float:
        """Mean of both directions' pairwise similarity (one side may be NaN)."""
        sims = []
        for me, other in ((id_a, id_b), (id_b, id_a)):
            row = self._players[me].similarity
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

    def _window_px(self, p: _Player) -> tuple[int, int, int]:
        """This tick's window each side in pixels, and the larger of the two."""
        R = self.resolution
        left = min(int(round(p.window_left * R)), R // 2)
        right = min(int(round(p.window_right * R)), R - 1 - left)
        return left, right, max(left, right)

    def _strip(self, osc: Oscillator, interval: float, C: OscillatorSettings, min_px: int, side: int) -> np.ndarray:
        """One colour of one person's pattern over offsets −side … side: the half pattern
        mirrored about the centre pixel and made visible."""
        R = self.resolution
        interval = max(interval, 2.0 * min_px)                            # never below one period
        # The strip runs a margin past the window, so the circular morphology's wrap at its ends
        # never reaches a pixel that is shown.
        n = min(side + min_px, R)
        half = LinePattern.lines(self._distance[:n + 1], interval, int(C.waveform), osc.fundamental,
                                 osc.harmonic, int(C.cutoff), osc.overtone_phase, osc.phase)
        strip = np.concatenate((half[:0:-1], half))                       # offsets −n … n
        return LinePattern.visible(strip, min_px)[n - side:n + side + 1]  # offsets −side … side

    def _tint(self, white: np.ndarray, blue: np.ndarray, min_px: int) -> tuple[np.ndarray, np.ndarray]:
        """The hit's tint: the central ``tint`` fraction of each colour's lines takes the other
        colour; at 1 the colours swap."""
        E = self._instrument.events
        white_core = LinePattern.core(white, E.tint_white, min_px)
        blue_core = LinePattern.core(blue, E.tint_blue, min_px)
        return (white & ~white_core) | blue_core, (blue & ~blue_core) | white_core

    @staticmethod
    def _paint(lines: np.ndarray, strip: np.ndarray, centre: int, left: int, right: int, side: int) -> None:
        """OR a strip over offsets −side … side into ``lines`` at ``centre``, cut to each side's window."""
        R = lines.size
        offsets = np.arange(-left, right + 1)
        idx = (centre + offsets) % R
        lines[idx] |= strip[offsets + side]

    def _draw_mask(self, mask: np.ndarray, mask_level: np.ndarray, p: _Player, centre: int) -> None:
        """Mark the person's mask: it goes over every pattern, lit dim blue by presence, and
        flashes on the hit."""
        P = self._instrument.mask
        R = self.resolution
        half = mask_half_width(P.width, R)
        idx = (centre + np.arange(-half, half + 1)) % R
        mask[idx] = True
        brightness = P.flash_brightness if p.hit else P.brightness
        mask_level[idx] = np.maximum(mask_level[idx], brightness * p.envelope)
