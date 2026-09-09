"""PoseInstrument — the heart of the piece (see ``data/LAYERS.md``).

Each person stands in a blue **anchor** line at their azimuth, and around them a
mirror-symmetric pattern of white and blue **lines** derived from their pose — the visual
analogue of how the sound works: pose → pattern as pose → sound. A neutral pose is
"boring": one white line each side. Arms up is the bass: many thick lines.

**The line world is anchored to the people, not to the ring.** The strip is divided into
segments between neighbouring participants; each segment fits a whole number of lines
(``n = round(gap / line_spacing)``), so its actual spacing is ``gap / n``. Every person is a
mirror point of their own pattern, and the run of lines between two people is *the same
lines* counted from either side — matched patterns join seamlessly by construction, with no
global grid to disagree with the symmetry. Line parameters (thickness, density, levels) are
blended by position along a segment, so a thick pattern thins toward a neutral neighbour.

**Sync**: above ``sync_threshold`` the patterns of a similarity-matched pair grow toward each
other along the shortest arc — through intermediate people too — until they meet.

**Motion** (``line_motion`` / ``line_flow``) is a shared phase φ, the same for everyone. Any
motion breaks instantaneous symmetry (it holds exactly at φ ∈ {0, ½}) — a symmetric flow
collides at segment midpoints, a global flow approaches on one side and departs on the other —
so ``STATIC`` is the default and the moving modes are for evaluation on the machine.

**Input contract** (all six pose parameters are read into ``_Participant`` every tick,
whether or not the current mapping draws with them): the four arm angles, ``LegDeviation``,
``TorsoTilt``; plus BBox length, presence (tracklets) and pairwise ``Similarity``. The
mapping in ``_Participant.lift`` / ``.bend`` and the colour balance is an initial proposal —
the composition work happens here, on settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING

import numpy as np
import pytweening

from modules.pose import features
from modules.settings import Field

from .._base_layer import HighLayer, LayerSettings
from .._utilities import BlendType, angle_to_strip_position, draw_field
from ...frame import Frame

if TYPE_CHECKING:
    from ....board import Board


class LineMotion(IntEnum):
    """What drives the shared line phase φ."""
    STATIC   = 0
    CONSTANT = auto()    # φ accumulates line_speed (spacings / s)
    PLAYHEAD = auto()    # φ follows the playhead bars × lines_per_bar


class LineFlow(IntEnum):
    """Which way a moving φ carries the lines."""
    SYMMETRIC = 0        # outward from (or inward to) every person; flows meet at midpoints
    GLOBAL    = auto()   # all lines move one way round the ring


class PoseInstrumentSettings(LayerSettings):
    line_spacing:   Field[float]      = Field(10.0, min=1.0,  max=90.0,  step=0.5,  description="Nominal line spacing (deg); each segment between neighbours fits a whole number of lines")
    line_motion:    Field[LineMotion] = Field(LineMotion.STATIC,                    description="Line phase drive: static, constant rate, or the playhead bars")
    line_flow:      Field[LineFlow]   = Field(LineFlow.SYMMETRIC,                   description="Moving lines flow outward from every person (symmetric) or one way round the ring (global)")
    line_speed:     Field[float]      = Field(0.0,  min=-2.0, max=2.0,   step=0.01, description="CONSTANT: spacings per second (negative = inward / the other way)")
    lines_per_bar:  Field[float]      = Field(1.0,  min=0.0,  max=36.0,  step=0.5,  description="PLAYHEAD: spacings travelled per playhead bar")
    line_phase:     Field[float]      = Field(0.0,  min=0.0,  max=1.0,   step=0.01, description="Phase offset (spacings) added to the motion; the first line sits (1 + phase) spacings out")
    n_blend:        Field[float]      = Field(0.1,  min=0.0,  max=0.5,   step=0.01, description="Crossfade band around the half-spacing where a segment's line count steps (0 = hard)")
    extent_min:     Field[float]      = Field(10.0, min=0.0,  max=180.0, step=0.5,  description="Pattern reach each side (deg) at neutral — one spacing shows the first line left and right", newline=True)
    extent_max:     Field[float]      = Field(60.0, min=0.0,  max=180.0, step=0.5,  description="Pattern reach each side (deg) with the arms up")
    line_edge:      Field[float]      = Field(3.0,  min=0.1,  max=36.0,  step=0.1,  description="Softness of the reach's outer end (deg)")
    line_min:       Field[float]      = Field(0.1,  min=0.01, max=1.0,   step=0.01, description="Line thickness (fraction of spacing) at neutral", newline=True)
    line_max:       Field[float]      = Field(0.6,  min=0.01, max=1.0,   step=0.01, description="Line thickness (fraction of spacing) with the arms up")
    line_soft:      Field[float]      = Field(0.3,  min=0.0,  max=1.0,   step=0.01, description="Line edge softness (fraction of the thickness)")
    harmonics:      Field[int]        = Field(2,    min=1,    max=4,     step=1,    description="Bent elbows subdivide the spacing up to this harmonic (1 = never)")
    level:          Field[float]      = Field(0.8,  min=0.0,  max=1.0,   step=0.01, description="White line level", newline=True)
    legs_dim:       Field[float]      = Field(0.3,  min=0.0,  max=1.0,   step=0.01, description="How much bent legs dim the white lines")
    blue_min:       Field[float]      = Field(0.0,  min=0.0,  max=1.0,   step=0.01, description="Blue between-line level with the legs straight")
    blue_max:       Field[float]      = Field(0.6,  min=0.0,  max=1.0,   step=0.01, description="Blue between-line level with the legs bent")
    anchor_width:   Field[float]      = Field(3.0,  min=0.1,  max=36.0,  step=0.1,  description="Blue anchor width (deg; scaled by pose length) — the person's own light", newline=True)
    anchor_level:   Field[float]      = Field(0.8,  min=0.0,  max=1.0,   step=0.01, description="Blue anchor level")
    sync_threshold: Field[float]      = Field(0.75, min=0.0,  max=0.99,  step=0.01, description="Pairwise similarity above which two patterns grow toward each other", newline=True)
    attack_seconds: Field[float]      = Field(1.0,  min=0.0,  max=10.0,  step=0.1,  description="Fade-in of a newly present person (s)")
    release_seconds:Field[float]      = Field(1.5,  min=0.0,  max=10.0,  step=0.1,  description="Fade-out after a person is gone (s; the last pose is held)")


# -- Geometry helpers (strip positions and offsets are turns in [0, 1)) ---------------

def _signed_offset(a: float, b: float) -> float:
    """Signed shortest offset a → b on the ring (turns, in [-0.5, 0.5))."""
    return ((b - a + 0.5) % 1.0) - 0.5


def _segment_counts(gap: float, spacing: float, n_blend: float) -> tuple[int, int, float]:
    """How many lines a segment of ``gap`` turns fits at the nominal ``spacing``: the two
    candidate counts and the crossfade weight toward the higher one. Away from the
    half-spacing boundary the count is simply the rounded ratio (weight 0); within
    ±``n_blend`` of the boundary the two counts crossfade so lines slide instead of jump."""
    ratio = gap / spacing
    low = max(1, int(math.floor(ratio)))
    frac = ratio - math.floor(ratio)
    if n_blend > 0.0 and abs(frac - 0.5) < n_blend:
        return low, low + 1, (frac - (0.5 - n_blend)) / (2.0 * n_blend)
    n = max(1, int(round(ratio)))
    return n, n, 0.0


def _line_centres(u: np.ndarray, harmonic: int, between: bool = False) -> np.ndarray:
    """The nearest line centre, in units of the harmonic's own spacing, for positions
    ``u`` measured in base spacings from the origin: white lines sit at the integers,
    blue between-lines half-way between them."""
    v = u * harmonic
    return np.floor(v) + 0.5 if between else np.round(v)


def _line_distance(u: np.ndarray, harmonic: int) -> np.ndarray:
    """Distance to the nearest white line (units of the harmonic's spacing)."""
    return np.abs(u * harmonic - _line_centres(u, harmonic))


def _between_distance(u: np.ndarray, harmonic: int) -> np.ndarray:
    """Distance to the nearest blue between-line (units of the harmonic's spacing)."""
    return np.abs(u * harmonic - _line_centres(u, harmonic, between=True))


def _line_profile(distance: np.ndarray, thickness: np.ndarray | float, soft: float) -> np.ndarray:
    """A line of ``thickness`` (fraction of spacing) around distance 0: full inside
    ``thickness/2 × (1 − soft)``, fading to 0 at ``thickness/2``."""
    half = np.asarray(thickness, dtype=np.float32) * 0.5
    ramp = np.maximum(half * soft, 1e-6)
    return np.clip((half - distance) / ramp, 0.0, 1.0)


def _anchor_gate(centre_offset: np.ndarray) -> np.ndarray:
    """Whole-line gate by the line centre's distance from the anchor (base spacings): the
    anchor's own quarter spacing holds no line, a line is fully born by half a spacing —
    so the "line zero" at the person never shows, and a moving φ births lines smoothly."""
    return np.clip((centre_offset - 0.25) / 0.25, 0.0, 1.0)


def _lines(u: np.ndarray, harmonic: int, thickness: np.ndarray | float, soft: float,
           flow_phase: float, between: bool) -> np.ndarray:
    """The gated line profile for one harmonic: ``u`` in base spacings from the person
    (``u = offset / spacing + flow_phase``, so a centre ``c`` sits ``c/harmonic − flow_phase``
    spacings from the anchor)."""
    centres = _line_centres(u, harmonic, between)
    profile = _line_profile(np.abs(u * harmonic - centres), thickness, soft)
    return profile * _anchor_gate(centres / harmonic - flow_phase)


def _ease(t: float) -> float:
    return pytweening.easeInOutSine(min(max(t, 0.0), 1.0))


# -- Participants and segments ------------------------------------------------------

@dataclass
class _Participant:
    """One person's input state (the six-parameter contract) plus presence and reach."""
    position:       float = 0.0     # strip position (turns)
    length:         float = 1.0     # BBox height (pose length)
    left_shoulder:  float = 0.0     # the four arm angles (rad, 0 = neutral)
    right_shoulder: float = 0.0
    left_elbow:     float = 0.0
    right_elbow:    float = 0.0
    legs:           float = 0.0     # LegDeviation [0, 1]
    tilt:           float = 0.0     # TorsoTilt [-1, 1]
    similarity:     np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    present:        bool  = False   # seen this tick
    envelope:       float = 0.0     # presence 0..1 (attack / release)
    extent_left:    float = 0.0     # this tick's reach each side (turns)
    extent_right:   float = 0.0

    # -- The initial mapping (a proposal; the composition work lives here) --
    @property
    def lift(self) -> float:
        """Arms raised: 0 hanging → 1 straight up (mean over both shoulders)."""
        return min(1.0, (abs(self.left_shoulder) + abs(self.right_shoulder)) / (2.0 * math.pi))

    @property
    def bend(self) -> float:
        """Elbows bent: 0 straight → 1 fully bent (mean over both elbows)."""
        return min(1.0, (abs(self.left_elbow) + abs(self.right_elbow)) / (2.0 * math.pi))


@dataclass
class _Segment:
    """The strip between two neighbouring participants, walking in increasing position."""
    start: int          # participant id at the low-position end
    end:   int          # participant id at the high-position end (== start when alone)
    gap:   float        # turns
    n_low: int          # candidate line counts (crossfaded by ``blend``)
    n_high: int
    blend: float


def _value(x: float, fallback: float) -> float:
    return fallback if math.isnan(x) else float(x)


class PoseInstrument(HighLayer):
    """The pose instrument; see the module docstring."""

    def __init__(self, resolution: int, config: PoseInstrumentSettings,
                 board: Board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage
        self._participants: dict[int, _Participant] = {}
        self._phase: float = 0.0            # CONSTANT motion's accumulated phase (spacings)
        self._pixel = np.arange(1, resolution + 1, dtype=np.float32) / resolution   # offsets 1..R px

    def reset(self) -> None:
        """A fresh instrument (S6 entry): forget every participant. The line phase is a
        world property and keeps running."""
        self._participants.clear()

    # -- Per tick --------------------------------------------------------------

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        dt = frame.tick.dt
        self._update_participants(dt)
        order = sorted((p.position, id) for id, p in self._participants.items())
        if not order:
            return
        ids = [id for _, id in order]
        phi = self._advance_phase(dt)
        segments = self._build_segments(ids)
        self._set_extents(ids, segments)

        spacing = P.line_spacing / 360.0
        edge = P.line_edge / 360.0
        for i, id in enumerate(ids):
            p = self._participants[id]
            # Right side: the segment starting here; left side: the segment ending here.
            self._draw_side(white, blue, p, +1, segments[i], self._participants[segments[i].end], p.extent_right, phi, edge, spacing)
            self._draw_side(white, blue, p, -1, segments[i - 1], self._participants[segments[i - 1].start], p.extent_left, phi, edge, spacing)
            anchor_w = P.anchor_width / 360.0 * (0.5 + 0.5 * p.length)
            draw_field(blue, p.position, anchor_w, P.anchor_level * p.envelope,
                       int(anchor_w * 0.3 * self.resolution), BlendType.MAX)

    # -- Participants ------------------------------------------------------------

    def _update_participants(self, dt: float) -> None:
        P = self._config
        tracklets = self._board.get_tracklets()
        frames = self._board.get_frames(self._pose_stage)
        for p in self._participants.values():
            p.present = False
        for id, pose in frames.items():
            tracklet = tracklets.get(id)
            if tracklet is None or not tracklet.is_active:
                continue
            azimuth = pose[features.Azimuth].value
            if math.isnan(azimuth):
                continue
            p = self._participants.setdefault(id, _Participant())
            p.present = True
            p.position = angle_to_strip_position(azimuth)
            height = pose[features.BBox][features.BBoxElement.height]
            p.length = height if not math.isnan(height) and height > 0.0 else p.length
            angles = pose[features.Angles].values
            p.left_shoulder  = _value(angles[features.AngleLandmark.left_shoulder],  p.left_shoulder)
            p.right_shoulder = _value(angles[features.AngleLandmark.right_shoulder], p.right_shoulder)
            p.left_elbow     = _value(angles[features.AngleLandmark.left_elbow],     p.left_elbow)
            p.right_elbow    = _value(angles[features.AngleLandmark.right_elbow],    p.right_elbow)
            p.legs = _value(pose[features.LegDeviation].value, p.legs)
            p.tilt = _value(pose[features.TorsoTilt].value, p.tilt)
            p.similarity = pose[features.Similarity].values
        gone: list[int] = []
        for id, p in self._participants.items():
            if p.present:
                p.envelope = 1.0 if P.attack_seconds <= 0.0 else min(1.0, p.envelope + dt / P.attack_seconds)
            else:
                p.envelope = 0.0 if P.release_seconds <= 0.0 else max(0.0, p.envelope - dt / P.release_seconds)
                if p.envelope <= 0.0:
                    gone.append(id)
        for id in gone:
            del self._participants[id]

    # -- The line world ------------------------------------------------------------

    def _advance_phase(self, dt: float) -> float:
        P = self._config
        motion = LineMotion(int(P.line_motion))
        if motion == LineMotion.CONSTANT:
            self._phase += P.line_speed * dt
            return P.line_phase + self._phase
        if motion == LineMotion.PLAYHEAD:
            return P.line_phase + self._board.get_playhead_signals().bars * P.lines_per_bar
        return P.line_phase

    def _build_segments(self, ids: list[int]) -> list[_Segment]:
        """Segment i runs from ids[i] to ids[i + 1] (the last wraps to the first; one
        participant alone bounds the full turn with themself)."""
        P = self._config
        spacing = P.line_spacing / 360.0
        segments: list[_Segment] = []
        for i, id in enumerate(ids):
            next_id = ids[(i + 1) % len(ids)]
            gap = 1.0 if next_id == id else (self._participants[next_id].position - self._participants[id].position) % 1.0
            n_low, n_high, blend = _segment_counts(gap, spacing, P.n_blend)
            segments.append(_Segment(id, next_id, gap, n_low, n_high, blend))
        return segments

    def _set_extents(self, ids: list[int], segments: list[_Segment]) -> None:
        """Each person's reach per side: the pose-driven extent, then the sync growth
        toward every similarity-matched partner along the shortest arc (carried across
        intermediate people segment by segment)."""
        P = self._config
        edge = P.line_edge / 360.0
        for i, id in enumerate(ids):
            p = self._participants[id]
            reach = (P.extent_min + (P.extent_max - P.extent_min) * p.lift) / 360.0
            p.extent_right = min(reach, segments[i].gap)
            p.extent_left  = min(reach, segments[i - 1].gap)
        threshold = P.sync_threshold
        for i, id_a in enumerate(ids):
            for j in range(i + 1, len(ids)):
                id_b = ids[j]
                sim = self._pair_similarity(id_a, id_b)
                if math.isnan(sim) or sim < threshold:
                    continue
                t = _ease((sim - threshold) / max(1.0 - threshold, 1e-6))
                delta = _signed_offset(self._participants[id_a].position, self._participants[id_b].position)
                direction = 1 if delta >= 0.0 else -1
                amount = t * (abs(delta) / 2.0 + edge)
                self._grow(ids, segments, i, direction, amount)
                self._grow(ids, segments, j, -direction, amount)

    def _grow(self, ids: list[int], segments: list[_Segment], index: int, direction: int, amount: float) -> None:
        """Extend the reach from ``ids[index]`` by ``amount`` turns in ``direction`` (+1 =
        increasing position), handing the remainder to each intermediate person."""
        remaining = amount
        for _ in range(len(ids)):
            if remaining <= 0.0:
                return
            p = self._participants[ids[index]]
            seg = segments[index] if direction > 0 else segments[index - 1]
            reach = min(remaining, seg.gap)
            if direction > 0:
                p.extent_right = max(p.extent_right, reach)
            else:
                p.extent_left = max(p.extent_left, reach)
            remaining -= seg.gap
            index = (index + direction) % len(ids)

    def _pair_similarity(self, id_a: int, id_b: int) -> float:
        """Mean of both directions' pairwise similarity (one side may be NaN)."""
        sims = []
        for me, other in ((id_a, id_b), (id_b, id_a)):
            row = self._participants[me].similarity
            if other < len(row) and not math.isnan(float(row[other])):
                sims.append(float(row[other]))
        return float(np.mean(sims)) if sims else float('nan')

    # -- Drawing ---------------------------------------------------------------------

    def _draw_side(self, white: np.ndarray, blue: np.ndarray, p: _Participant, sign: int,
                   seg: _Segment, neighbour: _Participant, extent: float, phi: float,
                   edge: float, spacing: float) -> None:
        """Draw one side of a person's pattern: the segment's lines, generated outward
        from the person, inside a soft-ended reach window, with the line parameters
        blended toward the neighbour's along the segment."""
        P = self._config
        R = self.resolution
        count = min(R, int(round((extent + edge) * R)))
        if count <= 0 or p.envelope <= 0.0:
            return
        base = int(round(p.position * R))
        idx = (base + sign * np.arange(1, count + 1)) % R
        offset = self._pixel[:count] + sign * (base / R - p.position)     # turns from the person, ≥ 0

        # Parameters blended along the segment toward the neighbour (f = 0 at the person).
        f = np.clip(offset / max(seg.gap, 1e-6), 0.0, 1.0).astype(np.float32)
        lift  = p.lift + (neighbour.lift - p.lift) * f
        bend  = p.bend + (neighbour.bend - p.bend) * f
        legs  = p.legs + (neighbour.legs - p.legs) * f
        thickness = P.line_min + (P.line_max - P.line_min) * lift
        white_level = P.level * (1.0 - legs * P.legs_dim)
        blue_level  = P.blue_min + (P.blue_max - P.blue_min) * legs

        # The reach window: full to the extent, fading over the edge.
        window = np.clip((extent + edge - offset) / edge, 0.0, 1.0)

        # Positions in base spacings from the person, for each candidate line count; the
        # line at the anchor itself is gated away (see _anchor_gate).
        flow_phase = -phi if (sign > 0 or LineFlow(int(P.line_flow)) == LineFlow.SYMMETRIC) else phi
        harmonic = max(1, int(P.harmonics))
        white_profile = np.zeros(count, dtype=np.float32)
        blue_profile  = np.zeros(count, dtype=np.float32)
        for n, weight in ((seg.n_low, 1.0 - seg.blend), (seg.n_high, seg.blend)):
            if weight <= 0.0:
                continue
            u = offset * n / seg.gap + flow_phase
            w1 = _lines(u, 1, thickness, P.line_soft, flow_phase, between=False)
            b1 = _lines(u, 1, thickness, P.line_soft, flow_phase, between=True)
            if harmonic > 1:
                wh = _lines(u, harmonic, thickness, P.line_soft, flow_phase, between=False)
                bh = _lines(u, harmonic, thickness, P.line_soft, flow_phase, between=True)
                w1 = w1 + (wh - w1) * bend
                b1 = b1 + (bh - b1) * bend
            white_profile += weight * w1
            blue_profile  += weight * b1

        env = p.envelope
        white[idx] = np.maximum(white[idx], window * white_profile * white_level * env)
        blue[idx]  = np.maximum(blue[idx],  window * blue_profile  * blue_level  * env)
