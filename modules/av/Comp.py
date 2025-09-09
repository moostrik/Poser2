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
from dataclasses import dataclass

from modules.Settings import Settings
from modules.av.PoseMetrics import MultiPoseMetrics
from modules.pose.PoseDefinitions import Pose

from modules.av.DrawMethods import BlendType

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class TestParameters():
    void_width: float =  0.05


    pattern_width: float = 0.2
    pattern_sharpness: float = 1.5


    pattern_edge: float = 0.2
    input_A0: float = 0.0
    input_A1: float = 0.0
    input_B0: float = 0.0
    input_B1: float = 0.0
    show_void: bool = False

    interval: float = 0.1
    num_players: int = 1

class Comp():
    def __init__(self, settings: Settings) -> None:
        self.resolution: int = settings.light_resolution
        self.parameters: TestParameters = TestParameters(
            interval=1.0 / settings.light_rate,
            num_players=settings.max_players)

        # self.draw_methods: DrawMethods = DrawMethods()
        self.pose_metrics: MultiPoseMetrics = MultiPoseMetrics(settings)

        self.white_array: np.ndarray = np.ones((self.resolution), dtype=IMG_TYPE)
        self.blue_array: np.ndarray = np.zeros((self.resolution), dtype=IMG_TYPE)
        self.void_array: np.ndarray = np.zeros((self.resolution), dtype=IMG_TYPE)
        self.output_img: np.ndarray = np.zeros((1, self.resolution, 3), dtype=IMG_TYPE)

        self.hot_reloader = HotReloadMethods(self.__class__, True)


    # SETTERS
    def set_smoothness(self, value: float) -> None:
        self.pose_metrics.set_smoothness(value)
    def set_responsiveness(self, value: float) -> None:
        self.pose_metrics.set_responsiveness(value)
    def set_void_width(self, value: float) -> None:
        self.parameters.void_width = max(0.0, min(1.0, value))
    def set_pattern_width(self, value: float) -> None:
        self.parameters.pattern_width = max(0.0, min(1.0, value))
    def set_input_A0(self, value: float) -> None:
        self.parameters.input_A0 = max(0.0, min(1.0, value))
    def set_input_A1(self, value: float) -> None:
        self.parameters.input_A1 = max(0.0, min(1.0, value))
    def set_input_B0(self, value: float) -> None:
        self.parameters.input_B0 = max(0.0, min(1.0, value))
    def set_input_B1(self, value: float) -> None:
        self.parameters.input_B1 = max(0.0, min(1.0, value))
    def show_overlay(self, value: bool) -> None:
        self.parameters.show_void = value


    def update(self, poses: list[Pose]) -> np.ndarray:

        self.pose_metrics.add_poses(poses)
        self.pose_metrics.update()

        self.blue_array.fill(0.0)
        self.white_array.fill(1.0)
        self.make_voids(self.void_array, self.pose_metrics, self.parameters)
        self.make_patterns(self.white_array, self.blue_array, self.pose_metrics, self.parameters)

        # add overlay to second channel
        if self.parameters.show_void:
            inverted_void_array = 1.0 - self.void_array
            Comp.blend_values(self.white_array, inverted_void_array, 0, BlendType.MULTIPLY)
            Comp.blend_values(self.blue_array, inverted_void_array, 0, BlendType.MULTIPLY)
            Comp.blend_values(self.blue_array, self.void_array * 0.5, 0, BlendType.ADD)

        self.output_img[0, :, 0] = self.white_array[:]
        self.output_img[0, :, 1] = self.blue_array[:]

        return self.output_img


    def reset(self) -> None:
        self.pose_metrics.reset()

    @staticmethod
    def make_voids(array: np.ndarray, pose_metrics: MultiPoseMetrics, P: TestParameters) -> None:
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
                void_width: float = P.void_width * 0.036
                width: float = (void_width + length * void_width)
                edge: float = 20
                Comp.draw_field(array, centre, width, strength, edge, BlendType.MAX)


    @staticmethod
    def make_patterns(whites: np.ndarray, blues: np.ndarray, pose_metrics: MultiPoseMetrics, P: TestParameters) -> None:
        whites.fill(0.0)
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

                white_line_count_A: float = 44  #+ (P.input_A0 * 88) + (P.input_A1 * 88)
                white_line_count_B: float = 22  #+ (P.input_B0 * 88) + (P.input_B1 * 88)
                white_line_width_A: float = 0.3 #+ (P.input_A0 * 0.7)
                white_line_width_B: float = 0.3 #+ (P.input_B0 * 0.7)
                white_line_speed_A: float = 2 #+ (P.input_A1 * 13.2)
                white_line_speed_B: float = 2.2 #+ (P.input_B1 * 13.2)

                blend: BlendType = BlendType.MAX

                Comp.draw_lines(whites, centre, age_pattern_width, white_line_count_A, white_line_width_A, white_line_speed_A, blend, False)
                Comp.draw_lines(whites, centre, -age_pattern_width, white_line_count_A, white_line_width_A, -white_line_speed_A, blend, False)
                # Comp.draw_lines(whites, centre, age_pattern_width, white_line_count_B, white_line_width_B, white_line_speed_B, blend, False)
                # Comp.draw_lines(whites, centre, -age_pattern_width, white_line_count_B, white_line_width_B, -white_line_speed_B, blend, False)


                # Comp.draw_lines(blues, centre, age_pattern_width, white_line_count_A, white_line_width_A, -white_line_speed_A, blend, False)
                # Comp.draw_lines(blues, centre, -age_pattern_width, white_line_count_A, white_line_width_A, white_line_speed_A, blend, False)
                # Comp.draw_lines(blues, centre, age_pattern_width, white_line_count_B, white_line_width_B, -white_line_speed_B, blend, False)
                # Comp.draw_lines(blues, centre, -age_pattern_width, white_line_count_B, white_line_width_B, white_line_speed_B, blend, False)


                # blue_line_count: float = 44 + (P.input_A0 * 88) + (P.input_A1 * 88)
                # blue_line_width: float = 0.3 + (P.input_A0 * 0.7)
                # blue_line_speed: float = 6.6 + (P.input_A1 * 13.2)

                # Comp.draw_lines(blues, centre, age_pattern_width, blue_line_count, blue_line_width, -blue_line_speed, BlendType.ADD)
                # Comp.draw_lines(blues, centre, -age_pattern_width, blue_line_count, blue_line_width, blue_line_speed, BlendType.ADD)


    @staticmethod
    def draw_lines(array: np.ndarray, centre: float, width: float, line_count: float, line_width: float, speed: float, blend: BlendType, relative_to_world: bool = False) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int = int(width * resolution)
        idx_start: int = field_centre if width >= 0 else (field_centre + field_width)
        field_width = abs(field_width)

        world_positions = (np.arange(field_width) + idx_start) % resolution
        world_positions = world_positions / resolution  # Normalize to 0.0-1.0 range

        time_offset: float = time.time() * speed
        wave_phase_per_unit: float = 2 * math.pi * line_count

        if relative_to_world:
            phases = world_positions * wave_phase_per_unit - time_offset
        else:
            phase_shift: float = math.pi if speed < 0 else 0.0
            rel_positions = world_positions - centre
            phases = rel_positions * wave_phase_per_unit - time_offset + phase_shift

        values = (1.0 + np.sin(phases)) * 0.5

        if line_width == 0.0:
            return
        if line_width == 1.0:
            values.fill(1.0)
        else:
            power_factor: float = 1.0 / line_width - 1.0
            values: np.ndarray = np.power(values, power_factor)


        Comp.draw_edge(values, int(field_width * 0.1), 1.5)
        Comp.apply_circular(array, values, idx_start, blend)


    @staticmethod
    def draw_field(array: np.ndarray, centre: float, width: float, strength: float, edge: int, blend: BlendType) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int = int(width * resolution)
        idx_start = int((field_centre - field_width // 2) % resolution)

        values = np.full(field_width, strength, dtype=array.dtype)

        edge_width: int = int(min(edge, field_width // 2))
        if edge_width > 0:
            Comp.draw_edge(values, edge_width, 1.5)

        Comp.apply_circular(array, values, idx_start, blend)

    @staticmethod
    def draw_edge(array: np.ndarray, edge: int, curve: float) -> None:
        resolution: int = len(array)
        if edge > resolution:
            edge = resolution
        if edge == 0:
            return

        array[:edge] *= np.linspace(0, 1, edge) ** curve
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