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
  the ``pose_instrument`` group; the wiring is code.
- the **events**: presence (a pose is seen), the hit (``PlayheadCrossing``, the tick closest to
  the crossing: each oscillator's push and the mask's flash), and **full sync**: the room's one
  gate (``_set_full_sync``), every present pair alike over the band (on from ``pose_instrument.on_at``, off
  below ``pose_instrument.off_at``), published on ``pose_instrument.full_sync`` for the show (``docs/STATES.md``, S7) and
  taken by three mechanisms, each through its own bypass into its own envelope: the **sync
  strobe** — every drawn pattern's white and blue swap on strobe frames (or go dark,
  ``sync_strobe.mode``), the rate climbing the nested ladder over the attack and stepping back
  down over the release — the **growth** (``pose_instrument.grow``), every window toward the full circle over
  its attack, the overlap clip shaping how they meet, falling back over its release — and the
  **blend** (``pose_instrument.blend``), everyone's measures eased toward the room's presence-weighted mean,
  one pattern for everyone; positions, masks and hits stay each person's own.
- the **scale**: the voices work in pattern degrees, real degrees over ``window.scale`` shrunk
  by ``window.crowd`` per further participant (the sum of presences, so it moves smoothly), and
  the bridge divides the positions it hands them; the whole pattern — reach, lines, motion —
  compresses about its person as one picture, the synth none the wiser. ``window.overlap``
  clips neighbouring reaches toward their midpoints: windows meet as territories at 0.
- the **mask**: a band at the person, over every pattern, lit at its levels by presence (dim blue
  in the preset) and flashing to its flash levels on a hit; its **border**, a line centerd on
  each edge of the band, half over the blue, that does not flash; and the playhead's **marker** over
  it all (``PlayheadMarker``), dimmed inside the masks. With ``pose_instrument.opposite`` the patterns are
  drawn half a turn from their people (ON) or in both places (BOTH) while the masks stay on
  them; under BOTH the copies win the overlap — own windows retreat to a copy's settled edge,
  and yield entirely to a copy reaching over their person (``_clip_against_copies``).
- the colors: output 1 is white, output 2 is blue; where voices overlap the fuller one shows.

Playing by hand: every parameter has its slot as a row in the panel (``pose_instrument.white_lines``,
``pose_instrument.blue_lines``, ``pose_instrument.white_strobe``, ``pose_instrument.blue_strobe``, ``pose_instrument.lfo``): Base, Amount, Curve and
Bypass. A bypassed parameter is its base while the others follow the body, so a pose can be taken
apart parameter by parameter; an oscillator's Bypass All button sets its five at once, a strobe's
its four, and clears them when all are set. ``window.width_bypass`` holds both reaches without a
partner.

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import partial

import numpy as np

from modules.pose import features
from modules.settings import BaseSettings, Field, Group, Widget

from .._base_layer import ProjectionLayer, LayerSettings
from .._utilities import normalize_azimuth, mask_half_width
from .playhead_marker import PlayheadMarker, PlayheadMarkerSettings
from ...frame import LightFrame
from ...synth import (Voice, Parameter, Sources, Envelope, Strobe,
                      OscillatorSettings, EnvelopeSettings, LfoSettings, StrobeSettings)
from ....pose import PlayheadCrossing, PlayheadOffset, playhead_step, DummySettings

KNOB = Widget.knob


# -- Settings: the pose_instrument root group, grouped by what is tuned together -----------------

class InstrumentWindowSettings(EnvelopeSettings):
    """The window: how far the pattern shows each side of a person, and when. The synth's part
    (taper, attack, release) with the bridge's: the reach at rest, its bypass, the scale,
    which compresses the whole pattern — reach, lines, motion — as one picture, and the overlap,
    which pulls neighbouring windows apart into territories. Sync's growth is its own mechanism
    (``GrowSettings``) and is switched off by the width's Bypass."""
    width:          Field[float] = Field(45.0, min=0.0, max=180.0, step=0.5,  widget=KNOB, label="Width",          description="Reach each side of a player at rest (deg)", row_label="Reach", newline=True)
    width_bypass:   Field[bool]  = Field(False,                                            label="Bypass",         description="Both reaches at the width: no sync growth")
    scale:          Field[float] = Field(1.0,  min=0.1, max=2.0,   step=0.01, widget=KNOB, label="Scale",          description="Overall window scale: the whole pattern compresses, nothing needs recalibrating", row_label="Scale", newline=True)
    crowd:          Field[float] = Field(1.0,  min=0.5, max=1.0,   step=0.01, widget=KNOB, label="Crowd",          description="Each further player multiplies the scale by this; 1 = off")
    overlap:        Field[float] = Field(1.0,  min=-1.0, max=1.0,  step=0.01, widget=KNOB, label="Overlap",        description="Reach into a neighbour's window: 1 full, 0 to the midpoint, negative a gap")


class Opposite(IntEnum):
    """Where a person's pattern is drawn; the mask stays on the person in every mode."""
    OFF  = 0      # at the person
    ON   = 1      # half a turn away
    BOTH = 2      # both places, the copies winning the overlap


class SyncStrobeMode(IntEnum):
    """What a sync strobe frame does to the patterns."""
    INVERT = 0      # white and blue trade places
    BLACK  = 1      # the patterns go dark


class SyncStrobeSettings(BaseSettings):
    """The sync strobe, full sync's visualization: while the room's gate is open (every present
    pair alike over the band, ``PoseInstrumentSettings``), every drawn pattern's white and blue
    swap on strobe frames (or go dark, the mode). The gate, through this bypass, drives an
    envelope whose value picks the rate from the nested powers of two: the strobe comes in
    climbing 1 to 16 per second over the attack and fades out stepping back down over the
    release, every change a seamless infill. The bridge's, after the voices render; the
    per-oscillator strobes are the patch's and independent. The band is the root's, shared
    with the growth and the blend; each mechanism keeps its own bypass and envelope."""
    bypass:          Field[bool]           = Field(False,                 label="Bypass", description="No sync strobe: full sync shows nothing", row_label="Sync Strobe", newline=True)
    mode:            Field[SyncStrobeMode] = Field(SyncStrobeMode.INVERT, label="Mode",   description="A strobe frame swaps white and blue, or goes dark")
    on_at:           Field[float]          = Field(0.95, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="On At",   description="Full sync comes on when every pair reads at least this", visible=False)
    off_at:          Field[float]          = Field(0.90, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Off At",  description="Full sync goes off when any pair falls below this", visible=False)
    attack_seconds:  Field[float]          = Field(2.0,  min=0.0, max=10.0, step=0.1,  widget=KNOB, label="Attack",  description="The rate climbs 1 to 16 per second over (s)")
    release_seconds: Field[float]          = Field(4.0,  min=0.0, max=10.0, step=0.1,  widget=KNOB, label="Release", description="The rate steps back down to silence over (s)")


class GrowSettings(BaseSettings):
    """The growth, full sync's other reach: while the room's gate is open (the band is the
    root's, shared with the sync strobe and the blend, each mechanism its own bypass and
    envelope), every window grows toward the full circle over the attack — the overlap clip
    shapes how the grown windows meet — and falls back over the release. ``window.width_bypass``
    switches it off."""
    on_at:           Field[float] = Field(0.95, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="On At",   description="Full sync comes on when every pair reads at least this", visible=False)
    off_at:          Field[float] = Field(0.90, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Off At",  description="Full sync goes off when any pair falls below this", visible=False)
    attack_seconds:  Field[float] = Field(3.0,  min=0.0, max=10.0, step=0.1,  widget=KNOB, label="Attack",  description="Windows grow to fill the circle while full sync holds, over (s)", row_label="Grow", newline=True)
    release_seconds: Field[float] = Field(3.0,  min=0.0, max=10.0, step=0.1,  widget=KNOB, label="Release", description="The grown windows fall back to the width over (s)")


class BlendSettings(BaseSettings):
    """The blend, full sync's third effect: while the room's gate is open (the band is the
    root's, shared with the strobe and the growth, each mechanism its own bypass and
    envelope), everyone's measures ease toward the room's presence-weighted mean, so every
    pattern becomes one; off the band everyone returns to their own. The blend is read-time
    only — the stored measures stay each person's own — and positions, masks and hits are
    never blended."""
    bypass:          Field[bool]  = Field(False,                                           label="Bypass",  description="No blend: everyone keeps their own pattern", row_label="Blend", newline=True)
    on_at:           Field[float] = Field(0.95, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="On At",   description="Full sync comes on when every pair reads at least this", visible=False)
    off_at:          Field[float] = Field(0.90, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Off At",  description="Full sync goes off when any pair falls below this", visible=False)
    attack_seconds:  Field[float] = Field(2.0,  min=0.0, max=10.0, step=0.1,  widget=KNOB, label="Attack",  description="Measures ease to the room's mean while full sync holds, over (s)")
    release_seconds: Field[float] = Field(2.0,  min=0.0, max=10.0, step=0.1,  widget=KNOB, label="Release", description="Everyone returns to their own measures over (s)")


class MaskSettings(BaseSettings):
    """The mask at the person: its width and its level per channel (0 is off); the border, a line
    on each edge of the band, centerd on it so half overlaps the blue; and the flash, the levels
    the band goes to on a hit and the release it falls back over. The border does not flash."""
    width:                 Field[float] = Field(3.0, min=0.5, max=20.0, step=0.1,  widget=KNOB, label="Width",   description="Mask width (deg)", row_label="Mask", newline=True)
    white:                 Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="White",   description="Mask white level")
    blue:                  Field[float] = Field(0.3, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Blue",    description="Mask blue level")
    border_width:          Field[float] = Field(1.0, min=0.0, max=10.0, step=0.1,  widget=KNOB, label="Width",   description="Border line width, centerd on each mask edge (deg); 0 = off", row_label="Border", newline=True)
    border_white:          Field[float] = Field(1.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="White",   description="Border white level")
    border_blue:           Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Blue",    description="Border blue level")
    flash_white:           Field[float] = Field(0.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="White",   description="Mask white level at a hit", row_label="Flash", newline=True)
    flash_blue:            Field[float] = Field(1.0, min=0.0, max=1.0,  step=0.01, widget=KNOB, label="Blue",    description="Mask blue level at a hit")
    flash_release_seconds: Field[float] = Field(0.3, min=0.0, max=2.0,  step=0.05, widget=KNOB, label="Release", description="Flash falls back to the mask's levels over (s)")


class PoseInstrumentSettings(BaseSettings):
    """The ``pose_instrument`` root group, a group per concept: the mask, the playhead's marker, the window,
    the sync strobe, the growth and the blend (the full-sync band declared here, shared into
    all three, and the room's gate it opens published as ``full_sync``, read by the show), the
    two oscillators, their strobes and the LFO (the synth's patch, a slot per parameter), the
    dummy. The wiring is ``connect``; the measures' dead zones are the pipeline's."""
    max_lines:    Field[int]  = Field(90, min=30, max=180, step=1, description="Visual limit: lines per revolution; no pitch goes above it")
    opposite:     Field[Opposite] = Field(Opposite.OFF, widget=Widget.select, description="Each player's lines at them, half a turn away, or both (copies win the overlap)")
    lean_share:   Field[float] = Field(2.0, min=0.0, max=4.0, step=0.05, widget=KNOB, label="Lean Share", description="Arms' pull a lean hands to one color: 0 off, 1 all at full lean, >1 overshoots")
    on_at:        Field[float] = Field(0.95, min=0.0, max=1.0, step=0.01, widget=KNOB, label="On At",  description="Full sync comes on when every pair reads at least this", row_label="Sync", newline=True)
    off_at:       Field[float] = Field(0.90, min=0.0, max=1.0, step=0.01, widget=KNOB, label="Off At", description="Full sync goes off when any pair falls below this")
    full_sync:    Field[bool]  = Field(False, access=Field.READ, label="Full Sync", description="Every present pair alike over the band")
    mask:         Group[MaskSettings]           = Group(MaskSettings)
    playhead:     Group[PlayheadMarkerSettings] = Group(PlayheadMarkerSettings)
    window:       Group[InstrumentWindowSettings] = Group(InstrumentWindowSettings)
    sync_strobe:  Group[SyncStrobeSettings]     = Group(SyncStrobeSettings, share=[on_at, off_at])
    grow:         Group[GrowSettings]           = Group(GrowSettings, share=[on_at, off_at])
    blend:        Group[BlendSettings]          = Group(BlendSettings, share=[on_at, off_at])
    white_lines:  Group[OscillatorSettings]     = Group(OscillatorSettings)
    blue_lines:   Group[OscillatorSettings]     = Group(OscillatorSettings)
    white_strobe: Group[StrobeSettings]         = Group(StrobeSettings)
    blue_strobe:  Group[StrobeSettings]         = Group(StrobeSettings)
    lfo:          Group[LfoSettings]            = Group(LfoSettings)
    dummy:        Group[DummySettings]          = Group(DummySettings)


# -- A person ------------------------------------------------------------------------------------

@dataclass
class _Player:
    """One person: their voice, their measures, and this tick's events."""
    voice:          Voice
    position:       float = 0.0     # normalized azimuth
    arms_mean:      float = 0.0     # the arms' mean travel, ArmTravel's arms_mean element [0, 1]
    shoulder_difference: float = 0.0     # ShoulderDifference [-1, 1]
    left_elbow_travel:     float = 0.0   # the elbows' travel, the absolute of ArmTravel [0, 1]
    right_elbow_travel:    float = 0.0
    legs:           float = 0.0     # LegDeviation [0, 1]
    body_bend:      float = 0.0     # TorsoTilt [-1, 1]
    distance:       float = 0.0     # Distance [0, 1]
    symmetry:      np.ndarray = field(default_factory=lambda: np.zeros(len(features.SymmetryElement), dtype=np.float32))
    similarity:     np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    present:        bool  = False   # seen this tick
    hit:            bool  = False   # the playhead crosses this person this tick
    flash:          Envelope = field(default_factory=Envelope)   # the mask's flash: up on the hit, then its release
    reach_left:     float = 0.0     # this tick's reach each side (deg), before presence
    reach_right:    float = 0.0
    opposite_reach_left:  float = 0.0   # the opposite copy's reaches (BOTH): never clipped by own windows
    opposite_reach_right: float = 0.0
    sync_strobe:    bool  = False   # this tick is a sync strobe frame, on the strobes' grid


class PoseInstrument(ProjectionLayer):
    """The pose instrument; see the module docstring."""

    HIT_TICKS = 1                   # a hit is the one tick closest to the crossing

    def __init__(self, resolution: int, settings: LayerSettings, instrument: PoseInstrumentSettings, board,
                 pose_stage: int) -> None:
        super().__init__(resolution, settings, board)
        self._instrument = instrument
        self._pose_stage = pose_stage
        self._players: dict[int, _Player] = {}
        self._scale = 1.0                                                   # this tick's window scale
        self._ticks_per_second = 1                                          # this tick's grid, the strobes' and the pulse's
        self._full_sync = False                                             # the room's gate: every present pair alike over the band
        self._sync_strobe = Envelope()                                      # the strobe's envelope: its value picks the rate
        self._grow = Envelope()                                             # the growth's envelope: the windows toward the full circle
        self._blend = Envelope()                                            # the blend's envelope: the measures toward the room's mean
        self._means: dict[str, float] = {}                                  # this tick's presence-weighted mean per measure
        self._crossing = PlayheadCrossing()
        self._pixels = np.arange(resolution, dtype=np.float64)              # the strip's pixel indices
        self._mask = np.zeros(resolution, dtype=bool)
        self._mask_white = np.zeros(resolution, dtype=np.float32)
        self._mask_blue = np.zeros(resolution, dtype=np.float32)
        for patch in (instrument.white_lines, instrument.blue_lines):
            patch.bind(OscillatorSettings.bypass_all, partial(self._bypass_all, patch))
        for strobe in (instrument.white_strobe, instrument.blue_strobe):
            strobe.bind(StrobeSettings.bypass_all, partial(self._strobe_bypass_all, strobe))

    def reset(self) -> None:
        """A fresh instrument (S6 entry): forget every player, pass and sync."""
        self._players.clear()
        self._crossing.reset()
        self._full_sync = False
        self._instrument.full_sync = False
        self._sync_strobe.reset()
        self._grow.reset()
        self._blend.reset()
        self._means = {}

    @staticmethod
    def _bypass_all(patch: OscillatorSettings, _: bool) -> None:
        """The panel's Bypass All button: every slot of the oscillator set, or cleared when all
        are already set."""
        bypass = not (patch.pitch_bypass and patch.pulse_width_bypass and patch.phase_bypass
                      and patch.speed_bypass and patch.hardness_bypass)
        patch.pitch_bypass = patch.pulse_width_bypass = patch.phase_bypass = bypass
        patch.speed_bypass = patch.hardness_bypass = bypass

    @staticmethod
    def _strobe_bypass_all(strobe: StrobeSettings, _: bool) -> None:
        """The panel's Bypass All button of a strobe: its four slots set, or cleared when all are
        already set."""
        bypass = not (strobe.rate_bypass and strobe.width_bypass and strobe.phase_bypass and strobe.shift_bypass)
        strobe.rate_bypass = strobe.width_bypass = strobe.phase_bypass = strobe.shift_bypass = bypass

    # -- Per tick --------------------------------------------------------------

    def _draw(self, frame: LightFrame, white: np.ndarray, blue: np.ndarray) -> None:
        self._update_players(frame)
        self._set_grow(frame.tick.dt)                                      # first: the reaches read its envelope
        self._set_reaches()
        self._set_sync_strobe(frame)
        mask, mask_white, mask_blue = self._mask, self._mask_white, self._mask_blue
        mask.fill(False)
        mask_white.fill(0.0)
        mask_blue.fill(0.0)

        R = self.resolution
        opposite = self._instrument.opposite
        half_turn = (R // 2) if opposite == Opposite.ON else 0
        players = list(self._players.values())
        if players:
            # Every voice in one batched pass (``Voice.render_all``): each row of ``signed`` is a
            # voice's pixels' angle from its person, the shortest way round, taken from the
            # person's own azimuth (half a turn away with ``opposite`` ON) so a walking person's
            # lines move smoothly. With BOTH each voice gets a second row at the antipode, the
            # copy, with its own reaches; ``render_all`` reads voices read-only and merges rows
            # by max, so a doubled voice is just two windows of one pattern. The angles are
            # pattern degrees — real degrees over the window scale — so the whole pattern
            # compresses about its person and the synth stays scale-agnostic: its knobs keep
            # their meaning relative to the pattern.
            px_per_degree = R / 360.0 * self._scale
            centers = [p.position * R + half_turn for p in players]
            reaches_left = [p.reach_left for p in players]
            reaches_right = [p.reach_right for p in players]
            if opposite == Opposite.BOTH:
                centers += [p.position * R + R / 2.0 for p in players]
                reaches_left += [p.opposite_reach_left for p in players]
                reaches_right += [p.opposite_reach_right for p in players]
            center_px = np.array(centers).reshape(-1, 1)
            signed = (((self._pixels - center_px + R / 2.0) % R) - R / 2.0) / px_per_degree
            reach_left = np.array(reaches_left).reshape(-1, 1)
            reach_right = np.array(reaches_right).reshape(-1, 1)
            sources = [self._sources(p) for p in players]
            n = len(players)                                # row k belongs to players[k % n]
            # On a sync strobe frame the strobing patterns render as their own batch, so their
            # outputs swap channels (or stay dark); off the strobe everyone is one batch as ever.
            steady = [k for k in range(len(centers)) if not players[k % n].sync_strobe]
            strobing = [k for k in range(len(centers)) if players[k % n].sync_strobe]
            for idx, strobed in ((steady, False), (strobing, True)):
                if not idx or (strobed and self._instrument.sync_strobe.mode == SyncStrobeMode.BLACK):
                    continue
                out_1, out_2 = Voice.render_all(
                    [players[k % n].voice for k in idx], signed[idx], reach_left[idx], reach_right[idx],
                    [sources[k % n] for k in idx])
                out_white, out_blue = (out_2, out_1) if strobed else (out_1, out_2)
                np.maximum(white, out_white, out=white)
                np.maximum(blue, out_blue, out=blue)

        for p in players:
            self._draw_mask(mask, mask_white, mask_blue, p, int(round(p.position * R)) % R)

        # The masks go over every pattern, each channel at the mask's level; the marker over all.
        np.copyto(white, mask_white, where=mask)
        np.copyto(blue, mask_blue, where=mask)
        PlayheadMarker.draw(white, blue, R, frame.playhead, self._instrument.playhead, mask)

    def _min_interval(self) -> float:
        """The visual limit on the interval, in degrees: one period of ``max_lines`` per revolution."""
        return 360.0 / self._instrument.max_lines

    # -- Players ------------------------------------------------------------

    def _update_players(self, frame: LightFrame) -> None:
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
                p = self._players[id] = _Player(Voice(P.white_lines, P.blue_lines, P.window, P.lfo,
                                                      P.white_strobe, P.blue_strobe, turn=360.0))
            p.present = True
            p.position = normalize_azimuth(azimuth)
            travel = np.abs(pose[features.ArmTravel].values)
            p.arms_mean           = self._value(travel[features.TravelElement.arms_mean],   p.arms_mean)
            p.shoulder_difference = self._value(pose[features.ShoulderDifference].value,    p.shoulder_difference)
            p.left_elbow_travel   = self._value(travel[features.TravelElement.left_elbow],  p.left_elbow_travel)
            p.right_elbow_travel  = self._value(travel[features.TravelElement.right_elbow], p.right_elbow_travel)
            p.legs = self._value(pose[features.LegDeviation].value, p.legs)
            p.body_bend = self._value(pose[features.TorsoTilt].value, p.body_bend)
            p.distance = self._value(pose[features.Distance].value, p.distance)
            p.symmetry = np.where(np.isnan(pose[features.AngleSymmetry].values), p.symmetry, pose[features.AngleSymmetry].values)
            p.similarity = pose[features.Similarity].values
            offsets[id] = pose[PlayheadOffset].value

        step = playhead_step(frame.motor_command.beam_rpm, frame.tick.interval)
        hits = self._crossing.update(offsets, step, self.HIT_TICKS)

        min_interval = self._min_interval()
        ticks_per_second = self._ticks_per_second = max(1, round(1.0 / max(frame.tick.interval, 1e-3)))   # the strobes' grid
        # The window scale: the master knob, shrunk per further participant. The participation is
        # the sum of presences — smooth as people arrive and leave — read before the voices
        # advance them, a one-tick lag the attack/release easing makes invisible. The voices work
        # in pattern degrees (real degrees over the scale), so the visual limit is converted.
        W = P.window
        participation = max(sum(p.voice.presence for p in self._players.values()), 1.0)
        self._scale = W.scale * W.crowd ** (participation - 1.0)
        self._set_full_sync()                                               # the room's gate, on this tick's presences and similarities
        self._set_blend(frame.tick.dt)                                      # before the loop: connect reads through it
        gone: list[int] = []
        for id, p in self._players.items():
            p.hit = id in hits
            p.voice.update_lfo(frame.tick.dt, self.connect_lfo(self._blended(p)))   # first: connect reads its output
            p.voice.update(frame.tick.dt, p.present, p.hit, self._sources(p), min_interval / self._scale,
                           frame.tick.index, ticks_per_second)
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
        """The voice's sources this tick: the connections, the measures read through the
        blend (``_blended``)."""
        return self.connect(self._blended(p))

    def connect(self, p: _Player) -> tuple[Sources, Sources]:
        """The connections (``docs/POSE_INSTRUMENT.md``, *The connections*) written out: a person's
        measures into the sources of the white and the blue oscillator's slots. The bases and the
        amounts, the range and the direction, are the ``pose_instrument.white_lines`` / ``pose_instrument.blue_lines``
        settings.

        The arms are ``docs/MATRIX_V2.md``'s Step 5:

        - the arms' mean, of the two arm travels: both pulse widths (white's base 0 and amount
          1, blue's base 1 and amount −1: arms hanging is full blue, arms raised full white,
          between them the two colors tile). An arm's travel is the weighted union of its
          shoulder's and its elbow's — the fold lifts the arm, so folded elbows draw with the
          shoulders down, and a raised shoulder is a full arm whatever the elbow
        - each elbow its own color's pitch, the left the white and the right the blue: folding an
          elbow makes its color's lines finer; equal elbows keep the colors in tune
        - the shoulders' pull, shared by the body bend: each color's phase takes its share,
          ``difference × (1 ± share × bend) / 2`` (white +, blue −, clamped to a source's −1..1;
          the share ``pose_instrument.lean_share``), the amounts opposite (white +¼, blue −¼). The arms alone
          say how far the colors pull together — the relative pull is the difference's, a
          quarter interval at full, whatever the bend — and the lean says which color carries
          it: upright both concede equally, leaned one color stands on its grid and the other
          walks. At share 1 a full lean hands over exactly the whole pull; above it the carrying
          color overshoots the meeting and the pair stretches open again, the clamp a
          standstill, not a jump. A lean with level arms says nothing

        Every other parameter sits at its base; connections are added one at a time, each earning
        its place under the matrix's rule 4.

        Every measure is the pipeline's, dead zones and all: the arms' union, its mean and the
        shoulders' difference come computed from ``ArmTravel`` and ``ShoulderDifference`` (the
        lift ``pose.arm_travel_extractor.elbow_lift``), the elbows' travel with its absolute
        taken since the sign is the side of the body the limb passes, which the design gives no
        meaning; the body bend from ``pose.torso_tilt_extractor``. The sound reads the same
        values (``docs/SOUND.md``). The pulls are computed here while the matrix is tried; once
        liked they move into the pipeline.
        """
        skew = self._instrument.lean_share * p.body_bend
        white_pull = min(1.0, max(-1.0, p.shoulder_difference * (1.0 + skew) / 2.0))
        blue_pull  = min(1.0, max(-1.0, p.shoulder_difference * (1.0 - skew) / 2.0))
        white: Sources = {
            Parameter.PULSE_WIDTH: p.arms_mean,
            Parameter.PITCH:       p.left_elbow_travel,
            Parameter.PHASE:       white_pull,
        }
        blue: Sources = {
            Parameter.PULSE_WIDTH: p.arms_mean,
            Parameter.PITCH:       p.right_elbow_travel,
            Parameter.PHASE:       blue_pull,
        }
        return white, blue

    def connect_lfo(self, p: _Player) -> float:
        """The source of the LFO's level: the leg deviation, its dead zone the pipeline's
        (``pose.leg_deviation_extractor``). The LFO feeds no parameter in the matrix yet; it is
        there when one needs it."""
        return p.legs

    # -- Reach, sync and the pulse ----------------------------------------------------

    def _set_reaches(self) -> None:
        """Each side's reach in pattern degrees, before presence: ``window.width``, grown toward
        the full circle by the sync growth (``_set_grow``'s envelope;
        ``window.width_bypass`` gates it off), then clipped toward the midpoint to every
        neighbour as far as ``window.overlap`` allows (the neighbour's presence scales the
        clip, so a neighbour leaving lets go smoothly). Grown windows meet through the same
        clip: at full sync the circle fills, lines everywhere, territories at overlap 0."""
        W = self._instrument.window
        full = 180.0 / self._scale                          # pattern degrees: half the circle real
        base = min(W.width + (full - W.width) * self._grow.value, full)
        for p in self._players.values():
            p.reach_left = p.reach_right = base
        self._clip_reaches(base)
        # The copies (BOTH) are the constellation rotated half a turn: their pairwise arcs equal
        # the owns', so the symmetric clip above is theirs too. They settle first and never
        # yield; the own windows then clip against them — the copies win.
        for p in self._players.values():
            p.opposite_reach_left, p.opposite_reach_right = p.reach_left, p.reach_right
        if self._instrument.opposite == Opposite.BOTH:
            self._clip_against_copies(base)

    def _clip_reaches(self, base: float) -> None:
        """Pull neighbouring windows apart into territories: each pair clips its facing sides
        toward the midpoint of the arc between them and its away sides toward the midpoint of the
        arc around the back, as far as ``window.overlap`` allows — 1 leaves the reaches alone, 0
        meets at the midpoints, negative retreats past them, a gap. A neighbour's presence scales
        their clip, and the min over every pair makes the nearest present neighbour binding."""
        W = self._instrument.window
        overlap = W.overlap
        if overlap >= 1.0:
            return
        turn = 360.0 / self._scale
        players = list(self._players.values())
        for i, a in enumerate(players):
            for b in players[i + 1:]:
                delta = self._signed_offset(a.position, b.position)
                d_short = abs(delta) * turn                             # the facing arc, pattern degrees
                d_long = turn - d_short                                 # the arc around the back
                limits = []
                for d, presence in ((d_short, b.voice.presence), (d_short, a.voice.presence),
                                    (d_long, b.voice.presence), (d_long, a.voice.presence)):
                    half = d / 2.0
                    target = max(half + (base - half) * overlap, 0.0) if base > half else base
                    limits.append(base + (target - base) * presence)
                if delta >= 0.0:
                    a.reach_right = min(a.reach_right, limits[0])
                    b.reach_left = min(b.reach_left, limits[1])
                    a.reach_left = min(a.reach_left, limits[2])
                    b.reach_right = min(b.reach_right, limits[3])
                else:
                    a.reach_left = min(a.reach_left, limits[0])
                    b.reach_right = min(b.reach_right, limits[1])
                    a.reach_right = min(a.reach_right, limits[2])
                    b.reach_left = min(b.reach_left, limits[3])

    def _set_full_sync(self) -> None:
        """Whether the room is in **full sync**: every present pair alike over the band. The gate
        opens when the least alike present pair reads ``on_at`` and closes only when one falls
        below ``off_at``, so a room hovering at one level never swaps it. A pair that cannot be
        read holds the room off, as does a person at neutral (the neutral weight is in the
        feature); below two people there is no pair. Computed once per tick, once the players'
        presences and similarities are this tick's; published on ``pose_instrument.full_sync`` for the show
        (``docs/STATES.md``, S7: PLAY winds down once its hold has built), and taken by the sync strobe,
        the growth and the blend, each through its own bypass into its own envelope."""
        P = self._instrument
        present = [id for id, p in self._players.items() if p.present]
        if len(present) < 2:
            self._full_sync = False
        else:
            sims = [self._pair_similarity(a, b) for i, a in enumerate(present) for b in present[i + 1:]]
            least = -1.0 if any(math.isnan(s) for s in sims) else min(sims)
            self._full_sync = least >= (min(P.off_at, P.on_at) if self._full_sync else P.on_at)
        P.full_sync = self._full_sync

    def _set_sync_strobe(self, frame: LightFrame) -> None:
        """Whether this tick is a sync strobe frame. The room's gate (``_set_full_sync``), through
        the strobe's bypass, drives an envelope whose value picks the rate from the nested powers
        of two (``Strobe.rate``, ``Strobe.period``): the strobe comes in climbing 1 to 16 per
        second over the attack — every other tick at a power-of-two tick rate — and fades out
        stepping back down over the release, every change a seamless infill on the strobes'
        shared grid, and a gate reopened mid-fade climbs from where it is. While it strobes,
        every drawn pattern strobes: full sync is the whole room.

        The envelope's value is also published on the frame (``frame.sync``) for the sound
        (``docs/SOUND.md``, /global/sync); the compositor carries it out only while this
        layer draws, so the wire reads 0 whenever the instrument is out of the mix."""
        P = self._instrument.sync_strobe
        value = self._sync_strobe.update(self._full_sync and not P.bypass, frame.tick.dt,
                                         P.attack_seconds, P.release_seconds)
        frame.sync = value
        strobe = False
        if self._sync_strobe.level > 0.0:
            rate = Strobe.rate(float(2.0 ** round(4.0 * value)), self._ticks_per_second)
            strobe = frame.tick.index % Strobe.period(rate, self._ticks_per_second) == 0
        for p in self._players.values():
            p.sync_strobe = strobe

    def _clip_against_copies(self, base: float) -> None:
        """BOTH only: pull every own window out of the copies' way — the copies win. For each
        own window and each copy (every person's, their own included, half a turn from its
        person), the facing side clips with the boundary at the copy's settled edge instead of
        the midpoint, each arc as in ``_clip_reaches``; and once a copy's window reaches over
        the person, both own sides scale down by the penetration — edge-clipping cannot carve
        a copy's interior out of a window — continuous with the edge clip where the copy's
        edge crosses the person, so walking into exact opposition ends in a full pattern swap,
        no jump anywhere. The copy-owner's presence scales the whole effect, and
        ``window.overlap`` lifts it toward 1."""
        W = self._instrument.window
        overlap = W.overlap
        if overlap >= 1.0:
            return
        turn = 360.0 / self._scale
        keep = min(max(overlap, 0.0), 1.0)                  # how much of the containment is lifted
        players = list(self._players.values())
        for a in players:
            for b in players:
                delta = self._signed_offset(a.position, (b.position + 0.5) % 1.0)
                d_short = abs(delta) * turn
                d_long = turn - d_short
                presence = b.voice.presence
                if delta >= 0.0:
                    sides = (('reach_right', d_short, b.opposite_reach_left),
                             ('reach_left', d_long, b.opposite_reach_right))
                else:
                    sides = (('reach_left', d_short, b.opposite_reach_right),
                             ('reach_right', d_long, b.opposite_reach_left))
                for side, d, copy_reach in sides:
                    floor = d - copy_reach
                    if base > floor:
                        target = max(floor + (base - floor) * overlap, 0.0)
                        limit = base + (target - base) * presence
                        setattr(a, side, min(getattr(a, side), limit))
                copy_facing = sides[0][2]
                if copy_facing > 0.0 and d_short < copy_facing:             # the copy reaches over the person
                    shrink = 1.0 - (1.0 - d_short / copy_facing) * (1.0 - keep) * presence
                    a.reach_left *= shrink
                    a.reach_right *= shrink

    def _set_grow(self, dt: float) -> None:
        """Whether full sync grows the windows. The room's gate (``_set_full_sync``), with the
        growth's own envelope so the mechanisms stay apart: while the gate is open the envelope
        rises over ``grow.attack_seconds`` and ``_set_reaches`` grows every window toward the
        full circle; closed, it falls back over ``grow.release_seconds``.
        ``window.width_bypass`` holds the growth off."""
        P = self._instrument.grow
        self._grow.update(self._full_sync and not self._instrument.window.width_bypass, dt,
                          P.attack_seconds, P.release_seconds)

    # What the blend averages: never similarity or position. The derived measures (the arms' mean,
    # the difference) blend linearly themselves; at full blend everyone reads the same value either way.
    MEASURES = ('arms_mean', 'shoulder_difference', 'left_elbow_travel', 'right_elbow_travel',
                'legs', 'body_bend', 'distance')

    def _set_blend(self, dt: float) -> None:
        """Whether full sync blends the measures. The room's gate (``_set_full_sync``), through
        the blend's bypass, with the blend's own envelope: while the gate is open the envelope
        rises over ``blend.attack_seconds`` and ``connect`` reads everyone's measures eased
        toward the room's presence-weighted mean (``_blended``), one pattern for everyone;
        closed, it falls back over ``blend.release_seconds``. The presences the mean weighs
        are last tick's, as the scale's participation is."""
        P = self._instrument.blend
        self._blend.update(self._full_sync and not P.bypass, dt, P.attack_seconds, P.release_seconds)
        total = sum(p.voice.presence for p in self._players.values())
        if self._blend.level > 0.0 and total > 0.0:
            self._means = {name: sum(getattr(p, name) * p.voice.presence for p in self._players.values()) / total
                           for name in self.MEASURES}
        else:
            self._means = {}

    def _blended(self, p: _Player) -> _Player:
        """The player as ``connect`` reads them: their measures eased toward the room's mean
        by the blend's envelope. A copy, so the stored measures stay each person's own — a
        lerp written back each tick would slew everyone to the mean and break the NaN hold."""
        t = self._blend.value
        if t <= 0.0 or not self._means:
            return p
        return replace(p, **{name: getattr(p, name) + (self._means[name] - getattr(p, name)) * t
                             for name in self.MEASURES})

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

    # -- Drawing ---------------------------------------------------------------------

    def _draw_mask(self, mask: np.ndarray, mask_white: np.ndarray, mask_blue: np.ndarray, p: _Player,
                   center: int) -> None:
        """Mark the person's mask: it goes over every pattern, each channel lit at the mask's level
        by presence, raised to the flash's level as far as the flash is up. The border is a line
        centerd on each edge of the band, half over the blue: its own levels by presence, part of
        the mask like the band, and it does not flash."""
        P = self._instrument.mask
        R = self.resolution
        half = mask_half_width(P.width, R)
        idx = (center + np.arange(-half, half + 1)) % R
        mask[idx] = True
        flash = p.flash.value
        presence = p.voice.presence
        for levels, base, at_hit in ((mask_white, P.white, P.flash_white), (mask_blue, P.blue, P.flash_blue)):
            level = (base + (at_hit - base) * flash) * presence
            levels[idx] = np.maximum(levels[idx], level)
        if P.border_width > 0.0 and (P.border_white > 0.0 or P.border_blue > 0.0):
            border_half = mask_half_width(P.border_width, R)
            offsets = np.arange(-border_half, border_half + 1)
            edges = np.concatenate((center - half + offsets, center + half + offsets)) % R
            mask[edges] = True
            for levels, base in ((mask_white, P.border_white), (mask_blue, P.border_blue)):
                levels[edges] = np.maximum(levels[edges], base * presence)
