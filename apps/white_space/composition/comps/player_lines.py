"""PlayerLines composition — visualises each tracked player as a coloured triplet on the LED strip.

Each active player produces three lines at their world position:
  - A centre line (blue by default, white when inverted)
  - A left and right flank line (white by default, blue when inverted)

Line width is modulated by the player's vertical position (BBox center_y) scaled by
``depth_scale``, giving a simple depth cue: a person closer to the camera (larger bbox,
higher center_y) produces wider lines.
"""

from dataclasses import dataclass

import numpy as np

from modules.settings import BaseSettings, Field
from modules.utils import HotReloadMethods
from modules.tracker import Tracklet
from modules.pose.frame import Frame
from modules.pose import features

from ..base import Composition
from ..transport import Transport
from ..draw import BlendType, apply_circular


class PlayerLinesSettings(BaseSettings):
    center_width:  Field[float] = Field(0.02,  min=0.0, max=0.2, step=0.01, description="Centre line width (strip fraction)")
    flank_width:   Field[float] = Field(0.03,  min=0.0, max=0.2, step=0.01, description="Flank line width (strip fraction)")
    depth_scale:   Field[float] = Field(0.0,   min=0.0, max=1.0, step=0.01, description="Depth scaling: 0=flat, 1=far player vanishes (centre_width is max)")
    invert:        Field[bool]  = Field(False,                              description="Swap centre/flank colours", newline=True)
    level_center:  Field[float] = Field(1.0,   min=0.0, max=1.0, step=0.01, description="Centre line level")
    level_flank:   Field[float] = Field(1.0,   min=0.0, max=1.0, step=0.01, description="Flank line level")


@dataclass
class _PlayerState:
    strip_pos: float = 0.5
    feet_y:    float = 0.5   # BBox bottom (feet) [0=top/far, 1=bottom/close]
    active:    bool  = False




class PlayerLines(Composition):
    """Draws a centre line + two flanking lines for each tracked player."""

    def __init__(self, resolution: int, config: PlayerLinesSettings) -> None:
        super().__init__(resolution, config)
        self._config = config
        self._states: dict[int, _PlayerState] = {}
        self._flank_buf: np.ndarray = np.zeros(resolution, dtype=np.float32)
        self._zeros_buf: np.ndarray = np.zeros(resolution, dtype=np.float32)
        self.hot_reloader = HotReloadMethods(self.__class__, True)

    # ------------------------------------------------------------------
    # Composition interface
    # ------------------------------------------------------------------

    def set_pose_inputs(self, frames: list[Frame], tracklets: dict[int, Tracklet]) -> None:
        seen: set[int] = set()

        for frame in frames:
            track_id = frame.track_id
            tracklet = tracklets.get(track_id)
            if tracklet is None or not tracklet.is_active:
                continue

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            strip_pos: float = world_angle % 360.0 / 360.0

            bottom_y: float = frame[features.BBox].to_rect().bottom
            if np.isnan(bottom_y):
                bottom_y = 0.5

            state = self._states.setdefault(track_id, _PlayerState())
            state.strip_pos = strip_pos
            state.feet_y    = float(np.clip(bottom_y, 0.0, 1.0))
            state.active    = True
            seen.add(track_id)

        for track_id, state in self._states.items():
            if track_id not in seen:
                state.active = False

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        depth_scale:   float = P.depth_scale
        level_center:  float = P.level_center
        level_flank:   float = P.level_flank
        invert:        bool  = P.invert

        resolution: int = self.resolution

        # --- collect geometry for all active players ---
        players: list[tuple[float, float, float, float, float]] = []
        for state in self._states.values():
            if not state.active:
                continue
            w_center: float = P.center_width * (1.0 - depth_scale * (1.0 - state.feet_y))
            w_flank:  float = P.flank_width
            half_gap: float = (w_center + w_flank) / 2.0
            players.append((
                state.strip_pos,
                w_center,
                (state.strip_pos - half_gap) % 1.0,
                (state.strip_pos + half_gap) % 1.0,
                w_flank,
            ))

        if not players:
            return

        center_arr = white if invert else blue
        flank_arr  = blue  if invert else white

        # --- draw all flanks into a temp buffer ---
        fw: int = int(P.flank_width * resolution)
        lv: np.ndarray | None = np.full(fw, level_flank, dtype=np.float32) if fw > 0 else None
        self._flank_buf[:] = 0.0
        if lv is not None:
            for (_, _, pos_left, pos_right, _) in players:
                l_start: int = int((int(pos_left  * resolution) - fw // 2) % resolution)
                r_start: int = int((int(pos_right * resolution) - fw // 2) % resolution)
                apply_circular(self._flank_buf, lv, l_start, BlendType.ADD)
                apply_circular(self._flank_buf, lv, r_start, BlendType.ADD)

        # --- punch holes at every centre so no flank can cover any centre ---
        for (pos_center, w_center, _, _, _) in players:
            cw: int = int(w_center * resolution)
            if cw > 0:
                c_start: int = int((int(pos_center * resolution) - cw // 2) % resolution)
                apply_circular(self._flank_buf, self._zeros_buf[:cw], c_start, BlendType.NONE)

        # --- commit flanks, then draw centres on top ---
        np.add(flank_arr, self._flank_buf, out=flank_arr)
        np.clip(flank_arr, 0.0, 1.0, out=flank_arr)
        for (pos_center, w_center, _, _, _) in players:
            cw: int = int(w_center * resolution)
            if cw > 0:
                c_start = int((int(pos_center * resolution) - cw // 2) % resolution)
                cv: np.ndarray = np.full(cw, level_center, dtype=np.float32)
                apply_circular(center_arr, cv, c_start, BlendType.ADD)
        np.clip(center_arr, 0.0, 1.0, out=center_arr)

    def reset(self) -> None:
        self._states.clear()
