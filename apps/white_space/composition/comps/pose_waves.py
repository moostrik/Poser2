"""Pose-driven void and wave pattern composition."""

import numpy as np
from time import time

from modules.utils import OneEuroFilter
from modules.tracker import Tracklet
from modules.pose.frame import Frame
from modules.pose import features
from modules.settings import BaseSettings, Field

from ..base import Composition
from ..transport import Transport
from ..output import COMP_DTYPE
from ..draw import BlendType, draw_waves, draw_field

import logging
logger = logging.getLogger(__name__)


class PoseWavesSettings(BaseSettings):
    """Settings for the pose-driven void and wave pattern composition."""
    enabled: Field[bool]  = Field(True,  description="Enable PoseWaves composition")
    gain:    Field[float] = Field(1.0,   min=0.0, max=2.0, step=0.01, description="Output gain")
    fov_degrees: Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5,
                                      description="Camera horizontal FOV — must match PanoramicTracker.fov")

    # Void zones
    void_width:    Field[float] = Field(0.05,  min=0.0, max=1.0,   step=0.01,  description="Void width (normalised)")
    void_edge:     Field[float] = Field(0.01,  min=0.0, max=1.0,   step=0.005, description="Void edge softness")
    use_void:      Field[bool]  = Field(True,                                   description="Enable void zones")

    # Wave pattern
    pattern_width:  Field[float] = Field(0.2,  min=0.0, max=1.0,   step=0.01, description="Pattern width (normalised)")
    pattern_edge:   Field[float] = Field(0.2,  min=0.0, max=1.0,   step=0.01, description="Pattern edge softness")
    line_sharpness: Field[float] = Field(1.5,  min=0.0, max=10.0,  step=0.1,  description="Line sharpness")
    line_speed:     Field[float] = Field(1.5,  min=0.0, max=10.0,  step=0.1,  description="Line speed")
    line_width:     Field[float] = Field(0.1,  min=0.0, max=1.0,   step=0.01, description="Line width (normalised)")
    line_amount:    Field[float] = Field(20.0, min=0.0, max=100.0, step=1.0,  description="Number of lines")


class PlayerState:
    """Tracks presence, age, and per-tick pose scalars for one player slot."""

    def __init__(self) -> None:
        self.present:    bool  = False
        self.start_age:  float = 0.0
        self.age:        float = 0.0
        self.world_position:     float = 0.0
        self.pose_length:        float = 1.0
        self.left_shoulder:      float = 0.0
        self.right_shoulder:     float = 0.0
        self.left_elbow:         float = 0.0
        self.right_elbow:        float = 0.0
        self.left_pattern_time:  float = 0.0
        self.right_pattern_time: float = 0.0

    def update_presence(self, tracklet: Tracklet) -> None:
        if tracklet.is_removed:
            self.reset()
            return
        if tracklet.is_active and tracklet.age_in_seconds > 2.0:
            if self.start_age == 0.0:
                self.start_age = time()
            self.present = True
            self.age = time() - self.start_age

    def reset(self) -> None:
        self.present   = False
        self.start_age = 0.0
        self.age       = 0.0


class PoseWaves(Composition):
    """Pose-driven void + wave pattern composition.

    Ported 1-to-1 from the original Compositor._draw / make_voids / make_patterns.
    """

    def __init__(
        self,
        resolution: int,
        num_players: int,
        config: PoseWavesSettings,
        tick_interval: float,
    ) -> None:
        super().__init__(resolution, config)
        self._config = config

        self._player_states: dict[int, PlayerState] = {
            i: PlayerState() for i in range(num_players)
        }
        self._num_active_smoother: OneEuroFilter = OneEuroFilter(
            freq=1.0 / tick_interval, mincutoff=1.0, beta=0.0
        )

        self._Wh_L: np.ndarray = np.zeros(resolution, dtype=COMP_DTYPE)
        self._Wh_R: np.ndarray = np.zeros(resolution, dtype=COMP_DTYPE)
        self._blue: np.ndarray = np.zeros(resolution, dtype=COMP_DTYPE)
        self._void: np.ndarray = np.zeros(resolution, dtype=COMP_DTYPE)

        self._latest_frames:    list[Frame]         = []
        self._latest_tracklets: dict[int, Tracklet] = {}

    # ------------------------------------------------------------------
    # Composition interface
    # ------------------------------------------------------------------

    def set_pose_inputs(self, frames: list[Frame], tracklets: dict[int, Tracklet]) -> None:
        self._latest_frames    = frames
        self._latest_tracklets = tracklets

    def reset(self) -> None:
        for state in self._player_states.values():
            state.reset()

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        if not P.enabled:
            return

        fov_degrees: float = P.fov_degrees
        dt:          float = transport.dt

        for frame in self._latest_frames:
            track_id = frame.track_id
            if track_id not in self._player_states:
                continue
            tracklet = self._latest_tracklets.get(track_id)
            if tracklet is None:
                continue

            state = self._player_states[track_id]
            state.update_presence(tracklet)
            if not state.present:
                continue

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            bbox    = frame[features.BBox]
            points  = frame[features.Points2D]
            nose_xy = points[features.PointLandmark.nose]
            nose_conf: float = points.get_score(features.PointLandmark.nose)
            bbox_rect = bbox.to_rect()
            if nose_conf > 0.3 and not np.isnan(nose_xy[0]) and not np.isnan(bbox_rect.width):
                nose_offset_x: float = float(nose_xy[0]) - 0.5
                world_angle += nose_offset_x * bbox_rect.width * fov_degrees
            state.world_position = float(np.deg2rad(world_angle - 180))

            bbox_height: float = bbox[features.BBoxElement.height]
            state.pose_length = (
                bbox_height if not np.isnan(bbox_height) and bbox_height > 0.0 else 1.0
            )

            angles = frame[features.Angles]
            state.left_shoulder  = angles.values[features.AngleLandmark.left_shoulder]
            state.right_shoulder = angles.values[features.AngleLandmark.right_shoulder]
            state.left_elbow     = angles.values[features.AngleLandmark.left_elbow]
            state.right_elbow    = angles.values[features.AngleLandmark.right_elbow]

        num_active: int = sum(1 for s in self._player_states.values() if s.present)
        smooth_active: float = self._num_active_smoother(float(num_active)) or 1.0

        self._Wh_L.fill(0.0)
        self._Wh_R.fill(0.0)
        self._blue.fill(0.0)

        PoseWaves._make_voids(self._void, self._player_states, P, dt)
        PoseWaves._make_patterns(
            self._Wh_L, self._Wh_R, self._blue,
            self._player_states, smooth_active, P, dt,
        )

        if P.use_void:
            inverted_void = 1.0 - self._void
            np.multiply(self._Wh_L, inverted_void, out=self._Wh_L)
            np.multiply(self._Wh_R, inverted_void, out=self._Wh_R)
            np.multiply(self._blue, inverted_void, out=self._blue)
            self._blue += self._void * 0.5

        g = P.gain
        white += (self._Wh_L + self._Wh_R) * g
        blue  += self._blue * g

    # ------------------------------------------------------------------
    # Internal draw helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_voids(
        array: np.ndarray,
        player_states: dict[int, PlayerState],
        P: PoseWavesSettings,
        interval: float,
    ) -> None:
        array -= interval * 4.0
        np.clip(array, 0, 1, out=array)

        for state in player_states.values():
            if not state.present:
                continue
            centre:     float = (state.world_position + np.pi) / (2 * np.pi)
            length:     float = state.pose_length
            age:        float = state.age
            strength:   float = pow(min(age * 1.8, 1.0), 1.5)
            void_width: float = P.void_width * 0.5
            width:      float = void_width + length * void_width
            edge:       int   = int(P.void_edge * len(array))
            draw_field(array, centre, width, strength, edge, BlendType.MAX)

    @staticmethod
    def _make_patterns(
        W_L: np.ndarray,
        W_R: np.ndarray,
        blues: np.ndarray,
        player_states: dict[int, PlayerState],
        smooth_num_active: float,
        P: PoseWavesSettings,
        interval: float,
    ) -> None:
        resolution: int = len(W_L)
        W_L.fill(0.0)
        W_R.fill(0.0)
        blues.fill(0.0)

        num_player_width: float = 1.0 / max(smooth_num_active, 1)
        pattern_width: float    = (
            P.pattern_width * 0.25 + num_player_width * P.pattern_width * 0.25
        )

        for state in player_states.values():
            if not state.present:
                continue

            centre: float = (state.world_position + np.pi) / (2 * np.pi)
            age:    float = state.age
            length: float = state.pose_length

            patt_width: float = pattern_width * pow(min(age * 0.25, 1.0), 0.75)

            left_count: float = 5 + P.line_amount * (1.0 - (np.cos(state.left_shoulder)  + 1.0) / 2.0)
            rigt_count: float = 5 + P.line_amount * (1.0 - (np.cos(state.right_shoulder) + 1.0) / 2.0)
            left_width: float = (
                P.line_width
                * ((np.cos(state.left_elbow)  + 1.0) / 2.0)
                * ((np.cos(state.left_shoulder)  + 1.0) / 2.0)
                * 0.8 + 0.2
            )
            rigt_width: float = (
                P.line_width
                * ((np.cos(state.right_elbow) + 1.0) / 2.0)
                * ((np.cos(state.right_shoulder) + 1.0) / 2.0)
                * 0.8 + 0.2
            )
            left_speed: float = P.line_speed * (-np.sin(state.left_elbow))
            rigt_speed: float = P.line_speed * ( np.sin(state.right_elbow))
            sharpness:  float = P.line_sharpness

            state.left_pattern_time  += interval * left_speed
            state.right_pattern_time += interval * rigt_speed
            left_time: float = state.left_pattern_time
            rigt_time: float = state.right_pattern_time

            outer_edge: int   = int(P.pattern_edge * resolution)
            void_width: float = P.void_width * 0.5 + length * P.void_width * 0.5
            inner_edge: int   = int(void_width * resolution * 0.7)

            blend: BlendType = BlendType.MAX
            draw_waves(W_L, centre,  patt_width, left_count, left_width, sharpness, left_time, 0,   inner_edge, outer_edge, blend)
            draw_waves(W_L, centre, -patt_width, left_count, left_width, sharpness, left_time, 0,   outer_edge, inner_edge, blend)
            draw_waves(W_R, centre,  patt_width, rigt_count, rigt_width, sharpness, rigt_time, 0.5, inner_edge, outer_edge, blend)
            draw_waves(W_R, centre, -patt_width, rigt_count, rigt_width, sharpness, rigt_time, 0.5, outer_edge, inner_edge, blend)
