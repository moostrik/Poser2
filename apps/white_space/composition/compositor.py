from threading import Event, Thread, Lock
from time import time, sleep
from typing import Callable

import numpy as np
from enum import Enum

from modules.utils.Smoothing import OneEuroFilter
from modules.tracker.Tracklet import Tracklet
from modules.pose.frame import Frame, FrameDict
from modules.pose.features.Angles import Angles, AngleLandmark
from modules.pose.features.BBox import BBox, BBoxElement
from modules.pose.features.Points2D import Points2D, PointLandmark

from apps.white_space.composition.settings import CompositorSettings
from apps.white_space.composition.output import CompositionOutput, CompositionDebug, COMP_DTYPE, CompositionOutputCallback
from apps.white_space.composition.test_composition import TestComposition, TestPattern

from modules.gl.Utils import FpsCounter
from modules.utils.HotReloadMethods import HotReloadMethods

import logging
logger = logging.getLogger(__name__)

PI: float = np.pi
TWOPI: float = 2 * np.pi
HALFPI: float = np.pi / 2


class EdgeSide(Enum):
    NONE  = 0
    LEFT  = 1
    RIGHT = 2
    BOTH  = 3


class BlendType(Enum):
    NONE     = "replace"
    ADD      = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    MAX      = "max"
    MIN      = "min"
    NON_ZERO = "non_zero"


# ---------------------------------------------------------------------------
# Per-player state
# ---------------------------------------------------------------------------

class PlayerState:
    """Tracks presence, age and per-tick pose scalars for one player slot."""

    def __init__(self) -> None:
        self.present:    bool  = False
        self.start_age:  float = 0.0
        self.age:        float = 0.0
        # Per-tick scalars populated in _draw before make_voids / make_patterns
        self.world_position:  float = 0.0
        self.pose_length:     float = 1.0
        self.left_shoulder:   float = 0.0
        self.right_shoulder:  float = 0.0
        self.left_elbow:      float = 0.0
        self.right_elbow:     float = 0.0
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


# ---------------------------------------------------------------------------
# Compositor — threaded LED composition loop running at a fixed rate
# ---------------------------------------------------------------------------

CompositionDebugCallback = Callable[[CompositionDebug], None]


class Compositor(Thread):
    """Runs the LED composition loop at a fixed rate (light_rate Hz) and sends
    the result over UDP to the installation hardware.

    Input:  pose frames (latest snapshot per player) + tracklets (locked snapshot)
    Output: CompositionOutput via UDP + board callbacks; CompositionDebug via board callbacks
    """

    def __init__(self, config: CompositorSettings) -> None:
        super().__init__(daemon=True, name="Compositor")

        self._stop_event = Event()
        self._tracklet_lock = Lock()
        self._frames_lock   = Lock()

        self._latest_tracklets: dict[int, Tracklet] = {}
        self._latest_frames:    dict[int, Frame]    = {}

        self._config   = config
        self.interval: float = 1.0 / config.light_rate
        resolution: int      = config.light_resolution
        num_players: int     = config.max_poses

        # LED work arrays
        self.Wh_L_array: np.ndarray = np.ones(resolution, dtype=COMP_DTYPE)
        self.Wh_R_array: np.ndarray = np.ones(resolution, dtype=COMP_DTYPE)
        self.blue_array: np.ndarray = np.ones(resolution, dtype=COMP_DTYPE)
        self.void_array: np.ndarray = np.zeros(resolution, dtype=COMP_DTYPE)

        self.output: CompositionOutput = CompositionOutput(resolution)
        self.debug:  CompositionDebug  = CompositionDebug(resolution)

        self._player_states: dict[int, PlayerState] = {i: PlayerState() for i in range(num_players)}
        self._num_active_smoother: OneEuroFilter = OneEuroFilter(
            freq=1.0 / self.interval, mincutoff=1.0, beta=0.0
        )

        # Test pattern override (active only when pattern != NONE)
        self.comp_test = TestComposition(resolution, config.test)

        self.fps_counter = FpsCounter()
        self._output_callbacks: list[CompositionOutputCallback]  = []
        self._debug_callbacks:  list[CompositionDebugCallback]   = []

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join()

    def run(self) -> None:
        next_time: float = time()
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Error in Compositor tick")
            next_time += self.interval
            sleep_time: float = next_time - time()
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                next_time = time()

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        frames = self._snapshot_frames()
        with self._tracklet_lock:
            tracklets = dict(self._latest_tracklets)

        self.output = CompositionOutput(self._config.light_resolution)
        self.debug  = CompositionDebug(self._config.light_resolution)

        self._draw(frames, tracklets)

        # Test pattern overrides the composition output when active
        if self._config.test.pattern != TestPattern.NONE:
            self.comp_test.update()
            np.copyto(self.output.light_img, self.comp_test.output_img)

        self._notify_output(self.output)
        self._notify_debug(self.debug)
        self.fps_counter.tick()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def add_poses(self, frames: FrameDict) -> None:
        """Store the latest frame per player; called from the pose pipeline thread."""
        with self._frames_lock:
            for track_id, frame in frames.items():
                self._latest_frames[track_id] = frame

    def set_tracklets(self, tracklets: dict[int, Tracklet]) -> None:
        with self._tracklet_lock:
            self._latest_tracklets = dict(tracklets)

    def _snapshot_frames(self) -> list[Frame]:
        """Consume the latest frames for this tick and clear the buffer."""
        with self._frames_lock:
            frames = list(self._latest_frames.values())
            self._latest_frames.clear()
        return frames

    # ------------------------------------------------------------------
    # Output callbacks
    # ------------------------------------------------------------------

    def add_output_callback(self, callback: CompositionOutputCallback) -> None:
        self._output_callbacks.append(callback)

    def add_debug_callback(self, callback: CompositionDebugCallback) -> None:
        self._debug_callbacks.append(callback)

    def _notify_output(self, output: CompositionOutput) -> None:
        for cb in self._output_callbacks:
            try:
                cb(output)
            except Exception:
                logger.exception("Error in Compositor output callback")

    def _notify_debug(self, debug: CompositionDebug) -> None:
        for cb in self._debug_callbacks:
            try:
                cb(debug)
            except Exception:
                logger.exception("Error in Compositor debug callback")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.fps_counter.get_fps()

    # ------------------------------------------------------------------
    # Composition draw entry point
    # ------------------------------------------------------------------

    def _draw(self, frames: list[Frame], tracklets: dict[int, Tracklet]) -> None:
        fov_degrees: float = self._config.fov_degrees

        for frame in frames:
            track_id = frame.track_id
            if track_id not in self._player_states:
                continue
            tracklet = tracklets.get(track_id)
            if tracklet is None:
                continue

            state = self._player_states[track_id]
            state.update_presence(tracklet)

            if not state.present:
                continue

            world_angle: float = getattr(tracklet.metadata, "world_angle", 0.0)
            bbox    = frame[BBox]
            points  = frame[Points2D]
            nose_xy = points[PointLandmark.nose]
            nose_conf: float = points.get_score(PointLandmark.nose)
            bbox_rect = bbox.to_rect()
            if nose_conf > 0.3 and not np.isnan(nose_xy[0]) and not np.isnan(bbox_rect.width):
                nose_offset_x: float = float(nose_xy[0]) - 0.5
                world_angle += nose_offset_x * bbox_rect.width * fov_degrees
            state.world_position = float(np.deg2rad(world_angle - 180))

            bbox_height: float = bbox[BBoxElement.height]
            state.pose_length = bbox_height if not np.isnan(bbox_height) and bbox_height > 0.0 else 1.0

            angles = frame[Angles]
            state.left_shoulder  = angles.values[AngleLandmark.left_shoulder]
            state.right_shoulder = angles.values[AngleLandmark.right_shoulder]
            state.left_elbow     = angles.values[AngleLandmark.left_elbow]
            state.right_elbow    = angles.values[AngleLandmark.right_elbow]

        # Smooth active player count
        num_active: int = sum(1 for s in self._player_states.values() if s.present)
        smooth_active: float = self._num_active_smoother(float(num_active)) or 1.0

        P = self._config
        Compositor.make_voids(self.void_array, self._player_states, P, self.interval)
        Compositor.make_patterns(
            self.Wh_L_array, self.Wh_R_array, self.blue_array,
            self._player_states, smooth_active, P, self.interval,
        )

        # Write debug snapshot (pre-void channels kept separate for WS_Lines.frag)
        self.debug.debug_img[0, :, 0] = self.Wh_L_array[:]
        self.debug.debug_img[0, :, 1] = self.Wh_R_array[:]
        self.debug.debug_img[0, :, 2] = self.blue_array[:]
        self.debug.debug_img[0, :, 3] = 0.0

        if P.use_void:
            self.debug.debug_img[0, :, 3] = self.void_array[:]
            inverted_void = 1.0 - self.void_array
            Compositor.blend_values(self.Wh_L_array, inverted_void, 0, BlendType.MULTIPLY)
            Compositor.blend_values(self.Wh_R_array, inverted_void, 0, BlendType.MULTIPLY)
            Compositor.blend_values(self.blue_array,  inverted_void, 0, BlendType.MULTIPLY)
            Compositor.blend_values(self.blue_array,  self.void_array * 0.5, 0, BlendType.ADD)

        self.output.light_0 = self.Wh_L_array[:] + self.Wh_R_array[:]
        self.output.light_1 = self.blue_array[:]

    def reset(self) -> None:
        for state in self._player_states.values():
            state.reset()

    # ------------------------------------------------------------------
    # LED composition
    # ------------------------------------------------------------------

    @staticmethod
    def make_voids(
        array: np.ndarray,
        player_states: dict[int, PlayerState],
        P: CompositorSettings,
        interval: float,
    ) -> None:
        array -= interval * 4.0
        np.clip(array, 0, 1, out=array)

        for i, state in player_states.items():
            if not state.present:
                continue
            centre: float    = (state.world_position + np.pi) / (2 * np.pi)
            length: float    = state.pose_length
            age:    float    = state.age
            strength: float  = pow(min(age * 1.8, 1.0), 1.5)
            void_width: float = P.void_width * 0.5
            width: float     = void_width + length * void_width
            edge: int        = int(P.void_edge * len(array))
            Compositor.draw_field(array, centre, width, strength, edge, BlendType.MAX)

    @staticmethod
    def make_patterns(
        W_L: np.ndarray, W_R: np.ndarray, blues: np.ndarray,
        player_states: dict[int, PlayerState],
        smooth_num_active: float,
        P: CompositorSettings,
        interval: float,
    ) -> None:
        resolution: int = len(W_L)
        W_L.fill(0.0)
        W_R.fill(0.0)
        blues.fill(0.0)

        num_player_width: float = 1.0 / max(smooth_num_active, 1)
        pattern_width: float    = P.pattern_width * 0.25 + num_player_width * P.pattern_width * 0.25

        for i, state in player_states.items():
            if not state.present:
                continue

            centre: float = (state.world_position + np.pi) / (2 * np.pi)
            age:    float = state.age
            length: float = state.pose_length

            age_pattern_speed: float = 0.25
            age_pattern_power: float = 0.75
            patt_width: float = pattern_width * pow(min(age * age_pattern_speed, 1.0), age_pattern_power)

            left_count:  float = 5 + P.line_amount * (1.0 - (np.cos(state.left_shoulder) + 1.0) / 2.0)
            rigt_count:  float = 5 + P.line_amount * (1.0 - (np.cos(state.right_shoulder) + 1.0) / 2.0)
            left_width:  float = P.line_width * ((np.cos(state.left_elbow)  + 1.0) / 2.0) * ((np.cos(state.left_shoulder)  + 1.0) / 2.0) * 0.8 + 0.2
            rigt_width:  float = P.line_width * ((np.cos(state.right_elbow) + 1.0) / 2.0) * ((np.cos(state.right_shoulder) + 1.0) / 2.0) * 0.8 + 0.2
            left_speed:  float = P.line_speed * (-np.sin(state.left_elbow))
            rigt_speed:  float = P.line_speed * ( np.sin(state.right_elbow))
            sharpness:   float = P.line_sharpness

            state.left_pattern_time  += interval * left_speed
            state.right_pattern_time += interval * rigt_speed
            left_time: float = state.left_pattern_time
            rigt_time: float = state.right_pattern_time

            outer_edge: int   = int(P.pattern_edge * resolution)
            void_width: float = P.void_width * 0.5 + length * P.void_width * 0.5
            inner_edge: int   = int(void_width * resolution * 0.7)

            blend: BlendType = BlendType.MAX

            Compositor.draw_waves(W_L, centre,  patt_width, left_count, left_width, sharpness, left_time, 0,   inner_edge, outer_edge, blend)
            Compositor.draw_waves(W_L, centre, -patt_width, left_count, left_width, sharpness, left_time, 0,   outer_edge, inner_edge, blend)
            Compositor.draw_waves(W_R, centre,  patt_width, rigt_count, rigt_width, sharpness, rigt_time, 0.5, inner_edge, outer_edge, blend)
            Compositor.draw_waves(W_R, centre, -patt_width, rigt_count, rigt_width, sharpness, rigt_time, 0.5, outer_edge, inner_edge, blend)

    @staticmethod
    def draw_waves(
        array: np.ndarray, anchor: float, span: float, num_waves: float,
        thickness: float, sharpness: float, time_value: float, phase: float,
        edge_left: int, edge_right: int, blend: BlendType,
    ) -> None:
        resolution: int = len(array)
        pixel_anchor: int = int(anchor * resolution)
        pixel_span: int   = abs(int(span * resolution))
        if pixel_span == 0 or thickness <= 0.0:
            return

        if thickness >= 1.0:
            intensities: np.ndarray = np.ones(pixel_span, dtype=array.dtype)
        else:
            thick_mode: bool = thickness > 0.5
            thick_trim: float = (thickness - 0.5) * 2.0 if thick_mode else 1.0 - thickness * 2.0
            thick_time_offset: float = (thickness - 0.5) * -TWOPI if thick_mode else thickness * TWOPI
            thick_phase_offset: float = HALFPI if thick_mode else -HALFPI
            wave_cycles: float = TWOPI * num_waves

            wave_time: float = time_value + thick_time_offset + phase * TWOPI
            positions = ((np.linspace(0, pixel_span - 1, pixel_span) + pixel_anchor) % resolution) / resolution

            indices = (positions - anchor) * wave_cycles - wave_time
            indices %= TWOPI
            indices /= TWOPI
            indices -= thick_trim
            np.clip(indices, 0, 1, out=indices)
            indices /= 1.0 - thick_trim
            indices *= TWOPI
            indices += thick_phase_offset

            intensities = (1.0 + np.sin(indices) * sharpness) * 0.5
            np.clip(intensities, 0, 1, out=intensities)

        pixel_start: int = pixel_anchor
        if span < 0:
            intensities = intensities[::-1]
            pixel_start = (pixel_start - pixel_span) % resolution

        Compositor.draw_edge(intensities, edge_left, 1.5, EdgeSide.LEFT)
        Compositor.draw_edge(intensities, edge_right, 1.5, EdgeSide.RIGHT)
        Compositor.apply_circular(array, intensities, pixel_start, blend)

    @staticmethod
    def draw_field(
        array: np.ndarray, centre: float, width: float, strength: float,
        edge: int, blend: BlendType,
    ) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int  = int(width * resolution)
        idx_start = int((field_centre - field_width // 2) % resolution)

        values = np.full(field_width, strength, dtype=array.dtype)
        edge_width: int = int(min(edge, field_width // 2))
        if edge_width > 0:
            Compositor.draw_edge(values, edge_width, 1.5, EdgeSide.BOTH)
        Compositor.apply_circular(array, values, idx_start, blend)

    @staticmethod
    def draw_edge(array: np.ndarray, edge: int, curve: float, edge_side: EdgeSide) -> None:
        if edge_side == EdgeSide.NONE or edge <= 0 or curve <= 0.0:
            return
        resolution: int = len(array)
        edge = min(edge, resolution)
        if edge == 0:
            return
        if edge_side in (EdgeSide.LEFT, EdgeSide.BOTH):
            array[:edge] *= np.linspace(0, 1, edge) ** curve
        if edge_side in (EdgeSide.RIGHT, EdgeSide.BOTH):
            array[-edge:] *= np.linspace(1, 0, edge) ** curve

    @staticmethod
    def apply_circular(
        array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType,
    ) -> None:
        resolution: int = len(array)
        start_idx = start_idx % resolution
        end_idx: int = (start_idx + len(values)) % resolution
        if start_idx < end_idx:
            Compositor.blend_values(array, values, start_idx, blend)
        else:
            Compositor.blend_values(array, values[:resolution - start_idx], start_idx, blend)
            Compositor.blend_values(array, values[resolution - start_idx:], 0, blend)

    @staticmethod
    def blend_values(
        array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType,
    ) -> None:
        resolution: int = len(array)
        end_idx = start_idx + len(values)
        start_idx = max(start_idx, 0)
        end_idx   = min(end_idx, resolution)
        if start_idx >= end_idx:
            return
        sl = slice(start_idx, end_idx)
        if   blend == BlendType.NONE:     array[sl]  = values
        elif blend == BlendType.ADD:      array[sl] += values
        elif blend == BlendType.SUBTRACT: array[sl] -= values
        elif blend == BlendType.MULTIPLY: array[sl] *= values
        elif blend == BlendType.MAX:      array[sl]  = np.maximum(array[sl], values)
        elif blend == BlendType.MIN:      array[sl]  = np.minimum(array[sl], values)
        elif blend == BlendType.NON_ZERO:
            mask = values != 0
            array[sl][mask] = values[mask]
        np.clip(array, 0, 1, out=array)
