"""PoseInstrument — the heart of the piece: the bridge between the pose data and the light synth
(``docs/POSE_INSTRUMENT.md``; the synth: ``docs/LIGHT_SYNTH.md``; the layer: ``docs/LAYERS.md``,
pose_instrument).

Each person gets a **voice** of the light synth (``light/synth``): two oscillators drawing lines
outward from the person, mirrored, one sent to white and one to blue, thinned to nothing toward
the window's **reach** each side. The bridge is everything the synth does not know:

- the **measures**: pose features and nothing else, read from the LERP frames; ``connect`` is the
  wiring of the document's *The connections* written out, a person's measures into the sources of
  the synth's slots. The bases and amounts are settings of the ``PI`` group; the wiring is code.
- the **events**: presence (a pose is seen), the hit (``PlayheadCrossing``, ``hit.frames``: a
  push on both oscillators and the mask's flash), and sync, which grows the reach on a partner's
  side until it reaches them, from ``window.sync_threshold`` on.
- the **mask**: a dim blue band at the person, over every pattern and lit by presence.
- the colours: output 1 is white, output 2 is blue; where voices overlap the fuller one shows.

Playing by hand: every input is a modulation matrix row in the panel (``PI.white``, ``PI.blue``,
``PI.lfo``): its knob, the Amount, a read-only Source knob showing the live source (the shown
person's: the dummy while it is enabled, else the first person present), a Curve and a Bypass. A
bypassed input is its knob while the others follow the body, so a pose can be taken apart input
by input. ``PI.bypass_all`` is the master: every source is muted and the panel draws for everyone;
``window.width_bypass`` holds both reaches without a partner; ``hit.hit`` marks everyone as the
playhead would.

``connect``, the drawing methods and the synth's classes are hot-reloaded while the app runs.
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
from ...frame import Frame
from ...synth import (Voice, Input, Sources, Oscillator, Envelope, Slot,
                      OscillatorSettings, WindowSettings as SynthWindowSettings, PushSettings, LfoSettings)
from ....pose import PlayheadCrossing, PlayheadOffset, playhead_step, DummySettings

KNOB = Widget.knob


# -- Settings: the PI root group, grouped by what is tuned together ------------------------------

class WindowSettings(SynthWindowSettings):
    """The window: how far the pattern shows each side of a person, and when. The synth's part
    (taper, attack, release) with the bridge's: the reach at rest, its bypass, and sync."""
    width:          Field[float] = Field(45.0, min=0.0, max=180.0, step=0.5,  widget=KNOB, label="Width",          description="Reach each side of a person at rest (deg)", newline=True)
    width_bypass:   Field[bool]  = Field(False,                                            label="Bypass",         description="Both reaches at the width: no sync growth")
    sync_threshold: Field[float] = Field(0.75, min=0.0, max=0.99,  step=0.01, widget=KNOB, label="Sync Threshold", description="Pair similarity from which the reach grows toward the partner, fully at 1 (alike)")


class HitSettings(PushSettings):
    """The hit: the playhead crossing a person. The synth's push (its settle time) with the
    bridge's: how many ticks, the mask's flash, and the button that hits everyone."""
    frames:           Field[int]   = Field(1,   min=1,   max=3,   step=1,    widget=KNOB, label="Frames", description="Hit length: the ticks closest to the crossing, 1-3")
    flash_brightness: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, widget=KNOB, label="Flash",  description="Mask blue level on a hit")
    hit:              Field[bool]  = Field(False, widget=Widget.button,                     label="Hit",    description="Hit everyone on the next ticks")


class MaskSettings(BaseSettings):
    """The dim blue mask at the person."""
    width:            Field[float] = Field(3.0, min=0.1, max=36.0, step=0.1,  widget=KNOB, label="Width",      description="Mask width (deg)")
    brightness:       Field[float] = Field(0.3, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Brightness", description="Mask blue level")
    playhead_at_mask: Field[float] = Field(0.3, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Playhead",   description="Playhead level inside a mask (fraction)")


class PoseInstrumentSettings(BaseSettings):
    """The ``PI`` root group, a group per concept: the two oscillators and the LFO (the synth's
    patch, a matrix row per input), the window, the hit, the mask, the dummy. The wiring is
    ``connect``; ``bypass_all`` is the master bypass: every source muted, every input its knob."""
    max_lines:  Field[int]  = Field(90, min=10, max=360, step=1, description="Visual limit: lines per revolution; no interval goes below one period")
    bypass_all: Field[bool] = Field(False,                       description="Master bypass: mute every source, every input is its knob; the panel draws")
    white:      Group[OscillatorSettings] = Group(OscillatorSettings)
    blue:       Group[OscillatorSettings] = Group(OscillatorSettings)
    lfo:        Group[LfoSettings]        = Group(LfoSettings)
    window:     Group[WindowSettings]     = Group(WindowSettings)
    hit:        Group[HitSettings]        = Group(HitSettings)
    mask:       Group[MaskSettings]       = Group(MaskSettings)
    dummy:      Group[DummySettings]      = Group(DummySettings)


# -- A person ------------------------------------------------------------------------------------

@dataclass
class _Player:
    """One person: their voice, their measures, and this tick's events."""
    voice:          Voice
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
    reach_left:     float = 0.0     # this tick's reach each side (deg), before presence
    reach_right:    float = 0.0


class PoseInstrument(ProjectionLayer):
    """The pose instrument; see the module docstring."""

    def __init__(self, resolution: int, config: LayerSettings, instrument: PoseInstrumentSettings, board,
                 pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._instrument = instrument
        self._pose_stage = pose_stage
        self._players: dict[int, _Player] = {}
        self._crossing = PlayheadCrossing()
        self._offsets = np.arange(-(resolution // 2), resolution // 2 + 1, dtype=np.float64)   # px from a person
        self._mask = np.zeros(resolution, dtype=bool)
        self._mask_level = np.zeros(resolution, dtype=np.float32)
        self._manual_hits = 0                                             # ticks left of a hit from the panel
        instrument.hit.bind(HitSettings.hit, self._on_hit)
        self._hot_reloaders = tuple(HotReloadMethods(cls, True) for cls in (self.__class__, Voice, Oscillator, Envelope, Slot))

    def reset(self) -> None:
        """A fresh instrument (S6 entry): forget every player and pass."""
        self._players.clear()
        self._crossing.reset()
        self._manual_hits = 0

    def _on_hit(self, _: bool) -> None:
        """The panel's hit button: everyone is hit for ``hit.frames`` ticks, from the next one."""
        self._manual_hits = int(self._instrument.hit.frames)

    # -- Per tick --------------------------------------------------------------

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        self._update_players(frame)
        if not self._players:
            return
        self._set_reaches()
        mask, mask_level = self._mask, self._mask_level
        mask.fill(False)
        mask_level.fill(0.0)

        for p in self._players.values():
            centre = int(round(p.position * self.resolution)) % self.resolution
            self._draw_voice(white, blue, p, centre)
            self._draw_mask(mask, mask_level, p, centre)

        # The masks go over every pattern: no white at a person, and the mask's own blue.
        white[mask] = 0.0
        np.copyto(blue, mask_level, where=mask)

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
                p = self._players[id] = _Player(Voice(P.white, P.blue, P.window, P.hit, P.lfo))
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

        step = playhead_step(frame.motor_command.beam_rpm, frame.tick.interval)
        hits = self._crossing.update(offsets, step, int(P.hit.frames))
        manual = self._manual_hits > 0
        if manual:
            self._manual_hits -= 1

        min_interval = self._min_interval()
        gone: list[int] = []
        muted = P.bypass_all
        for id, p in self._players.items():
            p.hit = manual or id in hits
            p.voice.update_lfo(frame.tick.dt, 0.0 if muted else self.connect_lfo(p))    # first: connect reads its output
            p.voice.update(frame.tick.dt, p.present, p.hit, self._sources(p), min_interval)
            if not p.present and not p.voice.alive:
                gone.append(id)
        for id in gone:
            del self._players[id]
        self._show_sources()

    def _show_sources(self) -> None:
        """The panel's Source knobs: the live sources of the shown person, the dummy while it is
        enabled (its id is above every live player's, and the ghosts are not among the poses), else
        the first person present; 0 for an input nobody plays. The one place the bridge writes
        settings, and the fields are read-only to the panel."""
        P = self._instrument
        present = sorted(id for id, p in self._players.items() if p.present)
        shown = self._players[present[-1] if P.dummy.enabled else present[0]] if present else None
        white, blue = self._sources(shown) if shown is not None else ({}, {})
        lfo_level = self.connect_lfo(shown) if shown is not None and not P.bypass_all else 0.0
        for patch, sources in ((P.white, white), (P.blue, blue)):
            patch.interval_source = float(sources.get(Input.INTERVAL, 0.0))
            patch.pulse_width_source = float(sources.get(Input.PULSE_WIDTH, 0.0))
            patch.phase_source = float(sources.get(Input.PHASE, 0.0))
            patch.speed_source = float(sources.get(Input.SPEED, 0.0))
            patch.hardness_source = float(sources.get(Input.HARDNESS, 0.0))
        P.lfo.level_source = float(lfo_level)

    @staticmethod
    def _value(x: float, fallback: float) -> float:
        return fallback if math.isnan(x) else float(x)

    # -- Connections -----------------------------------------------------------------

    def _sources(self, p: _Player) -> tuple[Sources, Sources]:
        """The voice's sources this tick: the connections, or nothing while the panel plays."""
        return ({}, {}) if self._instrument.bypass_all else self.connect(p)

    def connect(self, p: _Player) -> tuple[Sources, Sources]:
        """The connections (``docs/POSE_INSTRUMENT.md``, *The connections*) written out: a person's
        measures into the sources of the white and the blue oscillator's slots. The bases and the
        amounts, the range and the direction, are the ``PI.white`` / ``PI.blue`` settings.

        Each arm plays one oscillator, the left the white and the right the blue:

        - the shoulder: its pulse width (white's base 0 and amount 1, blue's base 1 and amount −1:
          arms hanging is full blue, arms raised full white)
        - the elbow: its interval, finer as the arm folds
        - the body bend, signed: both speeds, so a lean makes the lines flow one way or the other
        - the LFO (its level played by the legs, ``connect_lfo``): white's phase, a sway
        - the symmetries, blue's phase, the hardness: unconnected
        """
        flow = min(max(p.tilt, -1.0), 1.0)
        white = {
            Input.PULSE_WIDTH: self._measure(p.left_shoulder),
            Input.INTERVAL:    self._measure(p.left_elbow),
            Input.SPEED:       flow,
            Input.PHASE:       p.voice.lfo,
        }
        blue = {
            Input.PULSE_WIDTH: self._measure(p.right_shoulder),
            Input.INTERVAL:    self._measure(p.right_elbow),
            Input.SPEED:       flow,
        }
        return white, blue

    def connect_lfo(self, p: _Player) -> float:
        """The source of the LFO's level: the leg deviation, so bent knees bring the sway in."""
        return min(max(p.legs, 0.0), 1.0)

    @staticmethod
    def _measure(angle: float) -> float:
        """An angle as a measure 0..1: the pipeline's angles are calibrated so neutral is 0 and
        the raised pose π (``AngleCalibrator``), and π is the feature's range, not a tunable. The
        sign is the side of the body the limb passes, which the design gives no meaning, so the
        absolute is taken."""
        return min(max(abs(angle) / math.pi, 0.0), 1.0)

    # -- Reach and sync ---------------------------------------------------------------

    def _set_reaches(self) -> None:
        """Each side's reach in degrees, before presence: ``window.width``, grown toward every
        similarity-matched partner along the shorter arc, full reaching them; the partner's
        presence scales the growth, so a partner leaving lets go smoothly. Bypassed
        (``window.width_bypass``, or the master), it is the width and does not grow."""
        W = self._instrument.window
        base = min(W.width, 180.0)
        for p in self._players.values():
            p.reach_left = p.reach_right = base
        if W.width_bypass or self._instrument.bypass_all:
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
        blue, the fuller of overlapping voices showing. Distances are taken from the person's own
        azimuth, not from their centre pixel, so a walking person's lines move smoothly."""
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
        output_1, output_2 = p.voice.render(np.abs(signed), signed < 0.0, p.reach_left, p.reach_right, self._sources(p))
        idx = (centre + offsets.astype(np.int64)) % R
        white[idx] = np.maximum(white[idx], output_1)
        blue[idx] = np.maximum(blue[idx], output_2)

    def _draw_mask(self, mask: np.ndarray, mask_level: np.ndarray, p: _Player, centre: int) -> None:
        """Mark the person's mask: it goes over every pattern, lit dim blue by presence, and
        flashes on the hit."""
        P = self._instrument.mask
        R = self.resolution
        half = mask_half_width(P.width, R)
        idx = (centre + np.arange(-half, half + 1)) % R
        mask[idx] = True
        brightness = self._instrument.hit.flash_brightness if p.hit else P.brightness
        mask_level[idx] = np.maximum(mask_level[idx], brightness * p.voice.presence)
