"""PoseInstrument — the heart of the piece: the bridge between the pose data and the light synth
(``docs/POSE_INSTRUMENT.md``; the synth: ``docs/LIGHT_SYNTH.md``; the layer: ``docs/LAYERS.md``,
pose_instrument).

Each person gets a **voice** of the light synth (``light/synth``): two oscillators drawing lines
outward from the person, mirrored (or, with an oscillator's Mirror off, passing behind them), one
sent to white and one to blue, thinned to nothing toward the window's **reach** each side. The
bridge is everything the synth does not know:

- the **measures**: pose features and nothing else, read from the LERP frames, their dead zones
  already applied by the pipeline (``ArmTravel``, the body bend's and the leg deviation's
  extractors); ``connect`` is the wiring of the document's *The connections* written out, a
  person's measures into the sources of the synth's slots. The bases and amounts are settings of
  the ``PI`` group; the wiring is code.
- the **events**: presence (a pose is seen), the hit (``PlayheadCrossing``, the tick closest to
  the crossing: each oscillator's push and the mask's flash), and sync, which grows the reach on
  a partner's side until it reaches them, from ``window.sync_threshold`` on.
- the **mask**: a band at the person, over every pattern, lit at its levels by presence (dim blue
  in the preset) and flashing to its flash levels on a hit; and the playhead's **marker** over
  it all (``PlayheadMarker``), dimmed inside the masks. With ``PI.opposite`` the patterns are
  drawn half a turn from their people while the masks stay on them.
- the colours: output 1 is white, output 2 is blue; where voices overlap the fuller one shows.

Playing by hand: every parameter has its slot as a row in the panel (``PI.white_lines``,
``PI.blue_lines``, ``PI.lfo``): Base, Amount, Curve and Bypass. A bypassed parameter is its base
while the others follow the body, so a pose can be taken apart parameter by parameter; an
oscillator's Bypass All button sets its five at once, and clears them when all are set.
``window.width_bypass`` holds both reaches without a partner.

``connect``, the drawing methods and the synth's classes are hot-reloaded while the app runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from modules.pose import features
from modules.settings import BaseSettings, Field, Group, Widget
from modules.utils import HotReloadMethods

from .._base_layer import ProjectionLayer, LayerSettings
from .._utilities import normalize_azimuth, mask_half_width
from .playhead_marker import PlayheadMarker, PlayheadMarkerSettings
from ...frame import Frame
from ...synth import (Voice, Parameter, Sources, Oscillator, Envelope, Slot,
                      OscillatorSettings, WindowSettings as SynthWindowSettings, LfoSettings)
from ....pose import PlayheadCrossing, PlayheadOffset, playhead_step, DummySettings

KNOB = Widget.knob
_BREATH_POSITION = np.zeros(1)          # the breath is in time: one position


# -- Settings: the PI root group, grouped by what is tuned together ------------------------------

class WindowSettings(SynthWindowSettings):
    """The window: how far the pattern shows each side of a person, and when. The synth's part
    (taper, attack, release) with the bridge's: the reach at rest, its bypass, and sync."""
    width:          Field[float] = Field(45.0, min=0.0, max=180.0, step=0.5,  widget=KNOB, label="Width",          description="Reach each side of a person at rest (deg)", row_label="Reach", newline=True)
    width_bypass:   Field[bool]  = Field(False,                                            label="Bypass",         description="Both reaches at the width: no sync growth")
    sync_threshold: Field[float] = Field(0.75, min=0.0, max=0.99,  step=0.01, widget=KNOB, label="Sync Threshold", description="Pair similarity from which the reach grows toward the partner, fully at 1 (alike)")


class MaskSettings(BaseSettings):
    """The mask at the person: its width and its level per channel (0 is off), and the flash, the
    levels it goes to on a hit and the release it falls back over."""
    width:                 Field[float] = Field(3.0, min=0.5, max=20.0, step=0.1,  widget=KNOB, label="Width",   description="Mask width (deg)", row_label="Mask", newline=True)
    white:                 Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="White",   description="Mask white level")
    blue:                  Field[float] = Field(0.3, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Blue",    description="Mask blue level")
    flash_white:           Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="White",   description="Mask white level at a hit", row_label="Flash", newline=True)
    flash_blue:            Field[float] = Field(1.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Blue",    description="Mask blue level at a hit")
    flash_release_seconds: Field[float] = Field(0.3, min=0.0, max=2.0,  step=0.05, widget=KNOB, label="Release", description="Flash falls back to the mask's levels over (s)")


class BreathSettings(BaseSettings):
    """The breath: a sine in time the bridge makes per person, swinging the width of the colour
    whose shoulder is the higher. A depth of ½ or less keeps both fixed points exact."""
    rate:  Field[float] = Field(0.5, min=0.0, max=4.0, step=0.05, widget=KNOB, label="Rate",  description="Breaths per second (Hz)", row_label="Breath", newline=True)
    depth: Field[float] = Field(0.4, min=0.0, max=0.5, step=0.01, widget=KNOB, label="Depth", description="Width swing per unit of the higher shoulder's excess (fraction)")


class PoseInstrumentSettings(BaseSettings):
    """The ``PI`` root group, a group per concept: the mask, the playhead's marker, the window,
    the breath, the two oscillators and the LFO (the synth's patch, a slot per parameter), the
    dummy. The wiring is ``connect``; the measures' dead zones are the pipeline's."""
    max_lines:   Field[int]  = Field(90, min=30, max=180, step=1, description="Visual limit: lines per revolution; no pitch goes above it")
    opposite:    Field[bool] = Field(False, description="Draw each person's lines half a turn away; the masks stay on the people")
    mask:        Group[MaskSettings]           = Group(MaskSettings)
    playhead:    Group[PlayheadMarkerSettings] = Group(PlayheadMarkerSettings)
    window:      Group[WindowSettings]         = Group(WindowSettings)
    breath:      Group[BreathSettings]         = Group(BreathSettings)
    white_lines: Group[OscillatorSettings]     = Group(OscillatorSettings)
    blue_lines:  Group[OscillatorSettings]     = Group(OscillatorSettings)
    lfo:         Group[LfoSettings]            = Group(LfoSettings)
    dummy:       Group[DummySettings]          = Group(DummySettings)


# -- A person ------------------------------------------------------------------------------------

@dataclass
class _Player:
    """One person: their voice, their measures, and this tick's events."""
    voice:          Voice
    position:       float = 0.0     # normalized azimuth
    left_shoulder:  float = 0.0     # the four arm travels, the absolute of ArmTravel [0, 1]
    right_shoulder: float = 0.0
    left_elbow:     float = 0.0
    right_elbow:    float = 0.0
    left_turn:      float = 0.0     # the two elbow angles (rad), for the turn
    right_turn:     float = 0.0
    legs:           float = 0.0     # LegDeviation [0, 1]
    tilt:           float = 0.0     # TorsoTilt [-1, 1]
    distance:       float = 0.0     # Distance [0, 1]
    symmetry:      np.ndarray = field(default_factory=lambda: np.zeros(len(features.SymmetryElement), dtype=np.float32))
    similarity:     np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    present:        bool  = False   # seen this tick
    hit:            bool  = False   # the playhead crosses this person this tick
    flash:          Envelope = field(default_factory=Envelope)   # the mask's flash: up on the hit, then its release
    breath:         Oscillator = field(default_factory=Oscillator)   # the breath: a sine in time on the widths
    reach_left:     float = 0.0     # this tick's reach each side (deg), before presence
    reach_right:    float = 0.0


class PoseInstrument(ProjectionLayer):
    """The pose instrument; see the module docstring."""

    HIT_TICKS = 1                   # a hit is the one tick closest to the crossing

    def __init__(self, resolution: int, config: LayerSettings, instrument: PoseInstrumentSettings, board,
                 pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._instrument = instrument
        self._pose_stage = pose_stage
        self._players: dict[int, _Player] = {}
        self._crossing = PlayheadCrossing()
        self._offsets = np.arange(-(resolution // 2), resolution // 2 + 1, dtype=np.float64)   # px from a person
        self._mask = np.zeros(resolution, dtype=bool)
        self._mask_white = np.zeros(resolution, dtype=np.float32)
        self._mask_blue = np.zeros(resolution, dtype=np.float32)
        for patch in (instrument.white_lines, instrument.blue_lines):
            patch.bind(OscillatorSettings.bypass_all, partial(self._bypass_all, patch))
        self._hot_reloaders = tuple(HotReloadMethods(cls, True) for cls in (self.__class__, Voice, Oscillator, Envelope, Slot, PlayheadMarker))

    def reset(self) -> None:
        """A fresh instrument (S6 entry): forget every player and pass."""
        self._players.clear()
        self._crossing.reset()

    @staticmethod
    def _bypass_all(patch: OscillatorSettings, _: bool) -> None:
        """The panel's Bypass All button: every slot of the oscillator set, or cleared when all
        are already set."""
        bypass = not (patch.pitch_bypass and patch.pulse_width_bypass and patch.phase_bypass
                      and patch.speed_bypass and patch.hardness_bypass)
        patch.pitch_bypass = patch.pulse_width_bypass = patch.phase_bypass = bypass
        patch.speed_bypass = patch.hardness_bypass = bypass

    # -- Per tick --------------------------------------------------------------

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        self._update_players(frame)
        self._set_reaches()
        mask, mask_white, mask_blue = self._mask, self._mask_white, self._mask_blue
        mask.fill(False)
        mask_white.fill(0.0)
        mask_blue.fill(0.0)

        half_turn = (self.resolution // 2) if self._instrument.opposite else 0
        for p in self._players.values():
            centre = int(round(p.position * self.resolution)) % self.resolution
            self._draw_voice(white, blue, p, (centre + half_turn) % self.resolution)
            self._draw_mask(mask, mask_white, mask_blue, p, centre)

        # The masks go over every pattern, each channel at the mask's level; the marker over all.
        np.copyto(white, mask_white, where=mask)
        np.copyto(blue, mask_blue, where=mask)
        PlayheadMarker.draw(white, blue, self.resolution, frame.playhead, self._instrument.playhead, mask)

    def _min_interval(self) -> float:
        """The visual limit on the interval, in degrees: one period of ``max_lines`` per revolution."""
        return 360.0 / self._instrument.max_lines

    # -- Players ------------------------------------------------------------

    def _update_players(self, frame: Frame) -> None:
        P = self._instrument
        for p in self._players.values():
            p.present = False
        offsets: dict[int, float] = {}
        for id, pose in self._board.get_frames(self._pose_stage).items():
            azimuth = pose[features.Azimuth].value
            if math.isnan(azimuth):
                continue
            p = self._players.get(id)
            if p is None:
                p = self._players[id] = _Player(Voice(P.white_lines, P.blue_lines, P.window, P.lfo, turn=360.0))
            p.present = True
            p.position = normalize_azimuth(azimuth)
            travel = np.abs(pose[features.ArmTravel].values)
            p.left_shoulder  = self._value(travel[features.TravelElement.left_shoulder],  p.left_shoulder)
            p.right_shoulder = self._value(travel[features.TravelElement.right_shoulder], p.right_shoulder)
            p.left_elbow     = self._value(travel[features.TravelElement.left_elbow],     p.left_elbow)
            p.right_elbow    = self._value(travel[features.TravelElement.right_elbow],    p.right_elbow)
            angles = pose[features.Angles].values
            p.left_turn  = self._value(angles[features.AngleLandmark.left_elbow],  p.left_turn)
            p.right_turn = self._value(angles[features.AngleLandmark.right_elbow], p.right_turn)
            p.legs = self._value(pose[features.LegDeviation].value, p.legs)
            p.tilt = self._value(pose[features.TorsoTilt].value, p.tilt)
            p.distance = self._value(pose[features.Distance].value, p.distance)
            p.symmetry = np.where(np.isnan(pose[features.AngleSymmetry].values), p.symmetry, pose[features.AngleSymmetry].values)
            p.similarity = pose[features.Similarity].values
            offsets[id] = pose[PlayheadOffset].value

        step = playhead_step(frame.motor_command.beam_rpm, frame.tick.interval)
        hits = self._crossing.update(offsets, step, self.HIT_TICKS)

        min_interval = self._min_interval()
        gone: list[int] = []
        for id, p in self._players.items():
            p.hit = id in hits
            p.breath.update(frame.tick.dt, 1.0, P.breath.rate)                  # before connect reads it
            p.voice.update_lfo(frame.tick.dt, self.connect_lfo(p))              # first: connect reads its output
            p.voice.update(frame.tick.dt, p.present, p.hit, self._sources(p), min_interval)
            p.flash.update(p.hit, frame.tick.dt, 0.0, P.mask.flash_release_seconds)
            if not p.present and not p.voice.alive:
                gone.append(id)
        for id in gone:
            del self._players[id]

    @staticmethod
    def _value(x: float, fallback: float) -> float:
        return fallback if math.isnan(x) else float(x)

    # -- Connections -----------------------------------------------------------------

    def _sources(self, p: _Player) -> tuple[Sources, Sources]:
        """The voice's sources this tick: the connections."""
        return self.connect(p)

    def connect(self, p: _Player) -> tuple[Sources, Sources]:
        """The connections (``docs/POSE_INSTRUMENT.md``, *The connections*) written out: a person's
        measures into the sources of the white and the blue oscillator's slots. The bases and the
        amounts, the range and the direction, are the ``PI.white_lines`` / ``PI.blue_lines``
        settings.

        The arms are ``docs/MATRIX.md``'s Option 3:

        - the shoulders, the mean of the two: where both pulse widths rest (white's base 0 and
          amount 1, blue's base 1 and amount −1: arms hanging is full blue, arms raised full white)
        - the higher shoulder's excess over the other: its own colour's width breathes around
          that rest, ``PI.breath.depth`` × excess × the breath, the left the white and the right
          the blue; level shoulders have no excess, so the fixed points and a T are still
        - each elbow plays its own colour, the left the white and the right the blue: its fold is
          the pitch, its turn (the sine of its signed angle) the speed, one way at +90° and the
          other at −90°, still when straight and when fully folded
        - the body bend, signed, added to both elbow turns: white drifts outward and blue inward,
          so a lean one way makes the white faster and the blue slower, the other way the reverse
        - the LFO, the symmetries, the distance, the phases, the hardness: unconnected

        The measures come with their dead zones from the pipeline: the arm travels
        (``ArmTravel``, ``pose.arm_travel_extractor``), the absolute taken since the sign is the
        side of the body the limb passes, which the design gives no meaning; the body bend
        (``pose.torso_tilt_extractor``). The turn is the raw elbow angle. The mean, the excess,
        the turn and the sums are computed here while the matrix is tried; once liked they move
        into the pipeline. The breath is the bridge's own, as the mask's flash is.
        """
        left, right = p.left_shoulder, p.right_shoulder
        shoulders = (left + right) / 2.0
        swing = self._instrument.breath.depth * self._breath(p)
        white = {
            Parameter.PULSE_WIDTH: shoulders + swing * max(0.0, left - right),
            Parameter.PITCH:       p.left_elbow,
            Parameter.SPEED:       math.sin(p.left_turn) + p.tilt,
        }
        blue = {
            Parameter.PULSE_WIDTH: shoulders + swing * max(0.0, right - left),
            Parameter.PITCH:       p.right_elbow,
            Parameter.SPEED:       math.sin(p.right_turn) + p.tilt,
        }
        return white, blue

    @staticmethod
    def _breath(p: _Player) -> float:
        """The person's breath this tick, −1..1."""
        return float(Oscillator.sine(p.breath.cycle(_BREATH_POSITION, 1.0, 0.0), 1.0)[0])

    def connect_lfo(self, p: _Player) -> float:
        """The source of the LFO's level: the leg deviation, its dead zone the pipeline's
        (``pose.leg_deviation_extractor``). The LFO feeds no parameter in Option 1; it is there
        when one needs it."""
        return p.legs

    # -- Reach and sync ---------------------------------------------------------------

    def _set_reaches(self) -> None:
        """Each side's reach in degrees, before presence: ``window.width``, grown toward every
        similarity-matched partner along the shorter arc, full reaching them; the partner's
        presence scales the growth, so a partner leaving lets go smoothly. Bypassed
        (``window.width_bypass``), it is the width and does not grow."""
        W = self._instrument.window
        base = min(W.width, 180.0)
        for p in self._players.values():
            p.reach_left = p.reach_right = base
        if W.width_bypass:
            return
        threshold = W.sync_threshold
        ids = list(self._players)
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                sim = self._pair_similarity(id_a, id_b)
                if math.isnan(sim) or sim < threshold:
                    continue
                a, b = self._players[id_a], self._players[id_b]
                t = self._ease((sim - threshold) / max(1.0 - threshold, 1e-6))
                delta = self._signed_offset(a.position, b.position)
                distance = abs(delta) * 360.0
                grow_a = base + (distance - base) * t * b.voice.presence    # toward b, as far as b is there
                grow_b = base + (distance - base) * t * a.voice.presence
                if delta >= 0.0:
                    a.reach_right = max(a.reach_right, grow_a)
                    b.reach_left = max(b.reach_left, grow_b)
                else:
                    a.reach_left = max(a.reach_left, grow_a)
                    b.reach_right = max(b.reach_right, grow_b)

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

    def _draw_voice(self, white: np.ndarray, blue: np.ndarray, p: _Player, centre: int) -> None:
        """Paint a person's two outputs over their window: output 1 into white, output 2 into
        blue, the fuller of overlapping voices showing. ``centre`` is where the pattern is drawn,
        the person or, with ``opposite``, half a turn from them. Distances are taken from the
        person's own azimuth, not from their centre pixel, so a walking person's lines move
        smoothly."""
        R = self.resolution
        px_per_degree = R / 360.0
        presence = p.voice.presence
        left = min(int(math.ceil(p.reach_left * presence * px_per_degree)) + 1, R // 2)
        right = min(int(math.ceil(p.reach_right * presence * px_per_degree)) + 1, R // 2, R - 1 - left)
        if left <= 0 and right <= 0:
            return
        mid = R // 2
        offsets = self._offsets[mid - left:mid + right + 1]                # px from the centre pixel
        signed = (offsets - (p.position * R - round(p.position * R))) / px_per_degree    # deg from the person
        output_1, output_2 = p.voice.render(signed, p.reach_left, p.reach_right, self._sources(p))
        idx = (centre + offsets.astype(np.int64)) % R
        white[idx] = np.maximum(white[idx], output_1)
        blue[idx] = np.maximum(blue[idx], output_2)

    def _draw_mask(self, mask: np.ndarray, mask_white: np.ndarray, mask_blue: np.ndarray, p: _Player,
                   centre: int) -> None:
        """Mark the person's mask: it goes over every pattern, each channel lit at the mask's level
        by presence, raised to the flash's level as far as the flash is up."""
        P = self._instrument.mask
        R = self.resolution
        half = mask_half_width(P.width, R)
        idx = (centre + np.arange(-half, half + 1)) % R
        mask[idx] = True
        flash = p.flash.value
        presence = p.voice.presence
        for levels, base, at_hit in ((mask_white, P.white, P.flash_white), (mask_blue, P.blue, P.flash_blue)):
            level = (base + (at_hit - base) * flash) * presence
            levels[idx] = np.maximum(levels[idx], level)
