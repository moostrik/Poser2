from turtle import width
import wave
from cv2 import threshold
from numpy._typing._array_like import NDArray
from modules.av.Definitions import *
from typing import Optional
import math
import time
import numpy as np
from enum import Enum

from modules.Settings import Settings
from modules.av.PoseMetrics import MultiPoseMetrics
from modules.pose.PoseDefinitions import Pose

from modules.av.DrawMethods import BlendType

from modules.utils.HotReloadMethods import HotReloadMethods

PI: float = np.pi
TWOPI: float = 2 * np.pi
HALFPI: float = np.pi / 2

class EdgeSide(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    BOTH = 3

class Comp():
    def __init__(self, general_settings: Settings) -> None:
        self.resolution: int = general_settings.light_resolution
        self.settings: CompSettings = CompSettings(
            interval=1.0 / general_settings.light_rate,
            num_players=general_settings.max_players)

        # self.draw_methods: DrawMethods = DrawMethods()
        self.pose_metrics: MultiPoseMetrics = MultiPoseMetrics(general_settings)

        self.Wh_L_array: np.ndarray = np.ones((self.resolution), dtype=IMG_TYPE)
        self.Wh_R_array: np.ndarray = np.ones((self.resolution), dtype=IMG_TYPE)
        self.blue_array: np.ndarray = np.ones((self.resolution), dtype=IMG_TYPE)
        self.void_array: np.ndarray = np.zeros((self.resolution), dtype=IMG_TYPE)
        self.output_light: np.ndarray = np.zeros((1, self.resolution, 3), dtype=IMG_TYPE)
        # self.output_sound: np.ndarray = np.zeros((1, self.resolution, 3), dtype=IMG_TYPE)
        self.output_comp: np.ndarray = np.zeros((1, self.resolution, 4), dtype=IMG_TYPE)

        self.hot_reloader = HotReloadMethods(self.__class__, True)


    # SETTERS
    def update_settings(self) -> None:
        self.pose_metrics.set_smoothness(self.settings.smoothness)
        self.pose_metrics.set_responsiveness(self.settings.responsiveness)


    def update(self, poses: list[Pose]) -> np.ndarray:

        self.pose_metrics.add_poses(poses)
        self.pose_metrics.update()

        self.make_voids(self.void_array, self.pose_metrics, self.settings)
        self.make_patterns(self.Wh_L_array, self.Wh_R_array, self.blue_array, self.pose_metrics, self.settings)

        self.output_comp[0, :, 0] = self.Wh_L_array[:]
        self.output_comp[0, :, 1] = self.Wh_R_array[:]
        self.output_comp[0, :, 2] = self.blue_array[:]
        self.output_comp[0, :, 3] = self.void_array[:]

        # add overlay to second channel
        if self.settings.use_void:
            inverted_void_array = 1.0 - self.void_array
            Comp.blend_values(self.Wh_L_array, inverted_void_array, 0, BlendType.MULTIPLY)
            Comp.blend_values(self.Wh_R_array, inverted_void_array, 0, BlendType.MULTIPLY)
            Comp.blend_values(self.blue_array, inverted_void_array, 0, BlendType.MULTIPLY)
            Comp.blend_values(self.blue_array, self.void_array * 0.5, 0, BlendType.ADD)

        self.output_light[0, :, 0] = self.Wh_L_array[:] + self.Wh_R_array[:]
        self.output_light[0, :, 1] = self.blue_array[:]

        return self.output_light

    def reset(self) -> None:
        self.pose_metrics.reset()

    @staticmethod
    def make_voids(array: np.ndarray, pose_metrics: MultiPoseMetrics, P: CompSettings) -> None:
        array -= P.interval * 4.0
        np.clip(array, 0, 1, out=array)

        world_positions: dict[int, float | None] = pose_metrics.world_positions
        ages: dict[int, float | None] = pose_metrics.ages
        pose_lengths: dict[int, float | None] = pose_metrics.pose_lengths

        for i in range(len(world_positions)):
            centre: float | None = world_positions[i]
            length: float | None = pose_lengths[i]
            age: float | None = ages[i]

            if centre is not None and length is not None and age is not None:
                strength: float = pow(min(age * 1.8, 1.0), 1.5)
                void_width: float = P.void_width * 0.5
                width: float = (void_width + length * void_width)
                edge: int = int(P.void_edge * len(array))
                Comp.draw_field(array, centre, width, strength, edge, BlendType.MAX)


    @staticmethod
    def make_patterns(W_L: np.ndarray, W_R: np.ndarray, blues: np.ndarray, pose_metrics: MultiPoseMetrics, P: CompSettings) -> None:
        W_L.fill(0.0)
        W_R.fill(0.0)
        blues.fill(0.0)

        world_positions: dict[int, float | None] = pose_metrics.world_positions
        ages: dict[int, float | None] = pose_metrics.ages
        pose_lengths: dict[int, float | None] = pose_metrics.pose_lengths
        num_player_width: float = 1.0 / max(pose_metrics.smooth_num_active_players, 1)
        pattern_width: float = P.pattern_width * 0.25 + num_player_width * P.pattern_width * 0.25

        for i in range(pose_metrics.num_players):

            if pose_metrics.is_player_present(i) and world_positions[i] is not None:
                centre: float | None = world_positions[i]
                age: float | None = ages[i]
                if centre is None or age is None:
                    continue
                age_pattern_speed: float = 0.25
                age_pattern_power: float = 0.75
                age_pattern_width: float = pattern_width * pow(min(age * age_pattern_speed, 1.0), age_pattern_power)

                white_line_count_A: float = 33  #+ (P.input_A0 * 88) + (P.input_A1 * 88)
                white_line_count_B: float = 33  #+ (P.input_B0 * 88) + (P.input_B1 * 88)
                white_line_width_A: float = 0.1 #+ (P.input_A0 * 0.7)
                white_line_width_B: float = 0.1 #+ (P.input_B0 * 0.7)
                white_line_speed_A: float = 1.23 #+ (P.input_A1 * 13.2)
                white_line_speed_B: float = 1.23 #+ (P.input_B1 * 13.2)
                white_line_sharpness: float = 1

                blend: BlendType = BlendType.MAX

                Comp.draw_waves(W_L, centre, age_pattern_width,  white_line_count_A, white_line_width_A, white_line_sharpness, white_line_speed_A, 0, blend)
                Comp.draw_waves(W_L, centre, -age_pattern_width, white_line_count_A, white_line_width_A, white_line_sharpness, white_line_speed_A, 0, blend)
                Comp.draw_waves(W_R, centre, age_pattern_width,  white_line_count_B, white_line_width_B, white_line_sharpness, white_line_speed_B, 0.5, blend)
                Comp.draw_waves(W_R, centre, -age_pattern_width, white_line_count_B, white_line_width_B, white_line_sharpness, white_line_speed_B, 0.5, blend)

                Comp.draw_waves(blues, centre, age_pattern_width,  white_line_count_A, white_line_width_A, white_line_sharpness, -white_line_speed_A, 0, blend)
                Comp.draw_waves(blues, centre, -age_pattern_width, white_line_count_A, white_line_width_A, white_line_sharpness, -white_line_speed_A, 0, blend)
                Comp.draw_waves(blues, centre, age_pattern_width,  white_line_count_B, white_line_width_B, white_line_sharpness, -white_line_speed_B, 0.5, blend)
                Comp.draw_waves(blues, centre, -age_pattern_width, white_line_count_B, white_line_width_B, white_line_sharpness, -white_line_speed_B, 0.5, blend)


    @staticmethod
    def draw_waves(array: np.ndarray, anchor: float, span: float, num_waves: float, thickness: float, sharpness: float, speed: float, phase: float,blend: BlendType) -> None:
        # optional: work with smoothsteps

        resolution: int = len(array)
        pixel_anchor: int = int(anchor * resolution)
        pixel_span: int = abs(int(span * resolution))
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

            wave_time: float = time.time() * speed + thick_time_offset + phase * TWOPI
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


        edge_size: int = int(pixel_span * 0.1)
        Comp.draw_edge(intensities, edge_size, 1.5, EdgeSide.BOTH)
        Comp.apply_circular(array, intensities, pixel_start, blend)

    @staticmethod
    def draw_field(array: np.ndarray, centre: float, width: float, strength: float, edge: int, blend: BlendType) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int = int(width * resolution)
        idx_start = int((field_centre - field_width // 2) % resolution)

        values = np.full(field_width, strength, dtype=array.dtype)

        edge_width: int = int(min(edge, field_width // 2))
        if edge_width > 0:
            Comp.draw_edge(values, edge_width, 1.5, EdgeSide.BOTH)

        Comp.apply_circular(array, values, idx_start, blend)



    @staticmethod
    def draw_edge(array: np.ndarray, edge: int, curve: float, edge_side: EdgeSide) -> None:
        if edge_side == EdgeSide.NONE or edge <= 0 or curve <= 0.0:
            return

        resolution: int = len(array)
        if edge > resolution:
            edge = resolution
        if edge == 0:
            return

        if edge_side == EdgeSide.LEFT or edge_side == EdgeSide.BOTH:
            array[:edge] *= np.linspace(0, 1, edge) ** curve
        if edge_side == EdgeSide.RIGHT or edge_side == EdgeSide.BOTH:
            array[-edge:] *= np.linspace(1, 0, edge) ** curve

    @staticmethod
    def apply_circular(array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType) -> None:
        resolution: int = len(array)
        # print(array)

        start_idx = start_idx % resolution
        end_idx: int = (start_idx + len(values)) % resolution
        if start_idx < end_idx:
            Comp.blend_values(array, values, start_idx, blend)
        else:
            Comp.blend_values(array, values[:resolution - start_idx], start_idx, blend)
            Comp.blend_values(array, values[resolution - start_idx:], 0, blend)

    @staticmethod
    def blend_values(array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType) -> None:
        resolution: int = len(array)
        # print(blend)

        end_idx = start_idx + len(values)
        if start_idx < 0:
            start_idx = 0
        if end_idx > resolution:
            end_idx = resolution

            # raise ValueError("start_idx and end_idx must be within the bounds of the array.")

        if start_idx < end_idx:
            if blend == BlendType.NONE:
                array[start_idx:end_idx] =  values
            elif blend == BlendType.ADD:
                array[start_idx:end_idx] += values
            elif blend == BlendType.SUBTRACT:
                array[start_idx:end_idx] -= values
            elif blend == BlendType.MULTIPLY:
                array[start_idx:end_idx] *= values
            elif blend == BlendType.MAX:
                array[start_idx:end_idx] = np.maximum(array[start_idx:end_idx], values)
            elif blend == BlendType.MIN:
                array[start_idx:end_idx] = np.minimum(array[start_idx:end_idx], values)
            elif blend == BlendType.NON_ZERO:
                mask = values != 0
                array[start_idx:end_idx][mask] = values[mask]

        np.clip(array, 0, 1, out=array)