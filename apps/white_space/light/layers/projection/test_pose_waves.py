"""Pose-driven void and wave pattern composition."""

from __future__ import annotations

import numpy as np
from time import time
from typing import TYPE_CHECKING

from modules.utils import OneEuroFilter
from modules.pose import features
from modules.settings import Field

from .._base_layer import ProjectionLayer, LayerSettings
from ...frame import Frame, BUFFER_DTYPE
from .._utilities import BlendType, draw_waves, draw_field, normalize_azimuth

if TYPE_CHECKING:
    from ....board import Board

import logging
logger = logging.getLogger(__name__)


class TestPoseWavesSettings(LayerSettings):
    """Settings for the pose-driven void and wave pattern composition."""

    # Void zones
    void_width:    Field[float] = Field(18.0,  min=0.0, max=360.0, step=0.5,   description="Void width (deg)")
    void_edge:     Field[float] = Field(3.6,   min=0.0, max=360.0, step=0.5,   description="Void edge softness (deg)")
    use_void:      Field[bool]  = Field(True,                                   description="Enable void zones")

    # Wave pattern
    pattern_width:  Field[float] = Field(72.0, min=0.0, max=360.0, step=0.5,  description="Pattern width (deg)")
    pattern_edge:   Field[float] = Field(72.0, min=0.0, max=360.0, step=0.5,  description="Pattern edge softness (deg)")
    line_sharpness: Field[float] = Field(1.5,  min=0.0, max=10.0,  step=0.1,  description="Line sharpness")
    line_speed:     Field[float] = Field(1.5,  min=0.0, max=10.0,  step=0.1,  description="Line speed")
    line_width:     Field[float] = Field(0.1,  min=0.0, max=1.0,   step=0.01, description="Line thickness (fraction of a wave)")
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

    def update_presence(self, pose_age: float) -> None:
        """Present once the pose has existed for 2 s; `pose_age` is its `Age` feature (NaN on the
        first frame)."""
        if pose_age > 2.0:
            if self.start_age == 0.0:
                self.start_age = time()
            self.present = True
            self.age = time() - self.start_age

    def reset(self) -> None:
        self.present   = False
        self.start_age = 0.0
        self.age       = 0.0


class TestPoseWaves(ProjectionLayer):
    """Pose-driven void + wave pattern layer.

    Ported 1-to-1 from the original Compositor._draw / make_voids / make_patterns.
    """

    def __init__(
        self,
        resolution: int,
        max_players: int,
        config: TestPoseWavesSettings,
        board: Board,
        pose_stage: int,
    ) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

        self._player_states: dict[int, PlayerState] = {
            i: PlayerState() for i in range(max_players)
        }
        # Stepped with the tick's time, so the filter derives its rate from the timestamps; the constructed
        # frequency is only the first step's guess.
        self._num_active_smoother: OneEuroFilter = OneEuroFilter(freq=1.0, mincutoff=1.0, beta=0.0)

        self._Wh_L: np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        self._Wh_R: np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        self._blue: np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        self._void: np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)

    # ------------------------------------------------------------------
    # Layer interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        for state in self._player_states.values():
            state.reset()

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        dt: float = frame.tick.dt

        frames = list(self._board.get_frames(self._pose_stage).values())

        # A player without a pose this tick has left: presence and age start over when they return.
        seen: set[int] = {pose.track_id for pose in frames}
        for track_id, state in self._player_states.items():
            if track_id not in seen:
                state.reset()

        for pose in frames:
            track_id = pose.track_id
            if track_id not in self._player_states:
                continue

            state = self._player_states[track_id]
            state.update_presence(pose[features.Age].value)
            if not state.present:
                continue

            position: float = normalize_azimuth(pose[features.Azimuth].value)
            state.world_position = float((position - 0.5) * 2 * np.pi)

            bbox_height: float = pose[features.BBox][features.BBoxElement.height]
            state.pose_length = (
                bbox_height if not np.isnan(bbox_height) and bbox_height > 0.0 else 1.0
            )

            angles = pose[features.Angles]
            state.left_shoulder  = angles.values[features.AngleLandmark.left_shoulder]
            state.right_shoulder = angles.values[features.AngleLandmark.right_shoulder]
            state.left_elbow     = angles.values[features.AngleLandmark.left_elbow]
            state.right_elbow    = angles.values[features.AngleLandmark.right_elbow]

        num_active: int = sum(1 for s in self._player_states.values() if s.present)
        smooth_active: float = self._num_active_smoother(float(num_active), frame.tick.time) or 1.0

        self._Wh_L.fill(0.0)
        self._Wh_R.fill(0.0)
        self._blue.fill(0.0)

        TestPoseWaves._make_voids(self._void, self._player_states, P, dt)
        TestPoseWaves._make_patterns(
            self._Wh_L, self._Wh_R, self._blue,
            self._player_states, smooth_active, P, dt,
        )

        if P.use_void:
            inverted_void = 1.0 - self._void
            np.multiply(self._Wh_L, inverted_void, out=self._Wh_L)
            np.multiply(self._Wh_R, inverted_void, out=self._Wh_R)
            np.multiply(self._blue, inverted_void, out=self._blue)
            self._blue += self._void * 0.5

        white += (self._Wh_L + self._Wh_R)
        blue  += self._blue

    # ------------------------------------------------------------------
    # Internal draw helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_voids(
        array: np.ndarray,
        player_states: dict[int, PlayerState],
        P: TestPoseWavesSettings,
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
            void_width: float = P.void_width / 360.0 * 0.5
            width:      float = void_width + length * void_width
            edge:       int   = int(P.void_edge / 360.0 * len(array))
            draw_field(array, centre, width, strength, edge, BlendType.MAX)

    @staticmethod
    def _make_patterns(
        W_L: np.ndarray,
        W_R: np.ndarray,
        blues: np.ndarray,
        player_states: dict[int, PlayerState],
        smooth_num_active: float,
        P: TestPoseWavesSettings,
        interval: float,
    ) -> None:
        resolution: int = len(W_L)
        W_L.fill(0.0)
        W_R.fill(0.0)
        blues.fill(0.0)

        num_player_width: float = 1.0 / max(smooth_num_active, 1)
        pattern_normalized: float     = P.pattern_width / 360.0
        pattern_width: float    = (
            pattern_normalized * 0.25 + num_player_width * pattern_normalized * 0.25
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

            outer_edge: int   = int(P.pattern_edge / 360.0 * resolution)
            void_normalized:  float = P.void_width / 360.0
            void_width: float = void_normalized * 0.5 + length * void_normalized * 0.5
            inner_edge: int   = int(void_width * resolution * 0.7)

            blend: BlendType = BlendType.MAX
            draw_waves(W_L, centre,  patt_width, left_count, left_width, sharpness, left_time, 0,   inner_edge, outer_edge, blend)
            draw_waves(W_L, centre, -patt_width, left_count, left_width, sharpness, left_time, 0,   outer_edge, inner_edge, blend)
            draw_waves(W_R, centre,  patt_width, rigt_count, rigt_width, sharpness, rigt_time, 0.5, inner_edge, outer_edge, blend)
            draw_waves(W_R, centre, -patt_width, rigt_count, rigt_width, sharpness, rigt_time, 0.5, outer_edge, inner_edge, blend)
