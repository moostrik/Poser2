from modules.av.Definitions import *
from typing import Optional
import math
import time
from enum import Enum

from modules.Settings import Settings
from modules.av.SmoothMetrics import SmoothMetrics
from modules.pose.PoseDefinitions import Pose

from modules.utils.HotReloadMethods import HotReloadMethods

class BlendType(Enum):
    NONE = "replace"
    ADD = "add"
    MAX = "max"
    MIN = "min"


class TestParameters():
    def __init__(self) -> None:
        self.void_width: float =  0.05
        self.pattern_width: float = 0.2
        self.pattern_edge: float = 0.2
        self.input_A0: float = 0.0
        self.input_A1: float = 0.0
        self.input_B0: float = 0.0
        self.input_B1: float = 0.0
        self.show_overlay: bool = False

class Comp():
    def __init__(self, settings: Settings) -> None:
        self.resolution: int = settings.light_resolution
        self.num_players: int = settings.max_players
        self.parameters: TestParameters = TestParameters()
        self.interval: float = 1.0 / settings.light_rate

        # Pre-allocate arrays
        self.white_array: np.ndarray = np.zeros((self.resolution), dtype=IMG_TYPE)
        self.blue_array: np.ndarray = np.zeros((self.resolution), dtype=IMG_TYPE)
        self.overlay_array: np.ndarray = np.zeros((self.resolution), dtype=IMG_TYPE)
        self.output_img: np.ndarray = np.zeros((1, self.resolution, 3), dtype=IMG_TYPE)
        self.indices: np.ndarray = np.arange(self.resolution)
        
        
        
        self.smooth_metrics: dict[int, SmoothMetrics] = {}
        for i in range(self.num_players):
            self.smooth_metrics[i] = SmoothMetrics(settings)

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def update(self, poses: list[Pose]) -> np.ndarray:

        for pose in poses:
            self.smooth_metrics[pose.id].add_pose(pose)
            
        world_positions: dict[int, float] = {}
        approximate_lengths: dict[int, float] = {}        
        void_widths: dict[int, float] = {}
        for i in range(self.num_players):
            world_positions[i] = self.smooth_metrics[i].world_angle
            approximate_lengths[i] = self.smooth_metrics[i].approximate_person_length

            void_width: float = self.parameters.void_width * 0.036
            void_widths[i] = void_width + approximate_lengths[i] * void_width if approximate_lengths[i] is not None else None

        # print(approximate_lengths)

        self.make_voids(self.overlay_array, world_positions, void_widths, self.parameters)


        self.output_img[0, :, 0] = self.white_array[:]
        self.output_img[0, :, 1] = self.blue_array[:]
        # add overlay to second channel 
        if self.parameters.show_overlay:
            self.output_img[0, :, 1] = self.overlay_array[:]

        return self.output_img


    def reset(self) -> None:
        self.parameters = TestParameters()

    def set_smooth_alpha(self, value: float) -> None:
        for sm in self.smooth_metrics.values():
            sm.set_alpha(value)
    def set_void_width(self, value: float) -> None:
        self.parameters.void_width = max(0.0, min(1.0, value))
    def set_pattern_width(self, value: float) -> None:
        print(type(value), value)
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
        self.parameters.show_overlay = value

    def make_voids(self,array: np.ndarray, world_positions: dict[int, float], void_widths: dict[int, float], P: TestParameters) -> None:
        array -= self.interval * 4.0
        
        for i in range(len(world_positions)):
            if world_positions[i] is not None and void_widths[i] is not None:
                Comp.draw_field_with_edge(array, world_positions[i], void_widths[i], 1.0, 20, BlendType.MAX)

            
    @staticmethod        
    def draw_field(array: np.ndarray, centre: float, width: float, value: float, blend: BlendType) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int = int(width * resolution)
        idx_start = int((field_centre - field_width // 2) % resolution)
        
        values = np.full(field_width, value, dtype=array.dtype)
        Comp.apply_circular(array, values, idx_start, blend)
            
    @staticmethod 
    def draw_field_with_edge(array: np.ndarray, centre: float, width: float, value: float, edge: int, blend: BlendType) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int = int(width * resolution)
        idx_start = int((field_centre - field_width // 2) % resolution)
        
        edge_width: int = int(min(edge, field_width // 2))

        values = np.full(field_width, value, dtype=array.dtype)
        values[:edge_width] = np.linspace(0, value, edge_width)
        values[field_width-edge_width:] = np.linspace(value, 0, edge_width)
        
        Comp.apply_circular(array, values, idx_start, blend)
            
    @staticmethod
    def apply_circular(array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType) -> None:
        resolution: int = len(array)
        
        end_idx: int = (start_idx + len(values)) % resolution
        if start_idx < end_idx:
           Comp.blend_values(array, values, start_idx, blend)
        else:
            Comp.blend_values(array, values[:resolution - start_idx], start_idx, blend)
            Comp.blend_values(array, values[resolution - start_idx:], 0, blend)

    @staticmethod
    def blend_values(array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType) -> None:
        resolution: int = len(array)
        
        end_idx = start_idx + len(values)
        if start_idx < 0 or end_idx > resolution:
            raise ValueError("start_idx and end_idx must be within the bounds of the array.")
        
        if start_idx < end_idx:
            if blend == BlendType.NONE:
                array[start_idx:end_idx] =  values
            elif blend == BlendType.ADD:
                array[start_idx:end_idx] += values
            elif blend == BlendType.MAX:
                array[start_idx:end_idx] = np.maximum(array[start_idx:end_idx], values)
            elif blend == BlendType.MIN:
                array[start_idx:end_idx] = np.minimum(array[start_idx:end_idx], values)
                
        np.clip(array, 0, 1, out=array)

    # @staticmethod        
    # def draw_in_array_with_edge(array: np.ndarray, start_idx: int, end_idx: int, value: float, edge:float, blend: BlendType) -> None:


        # print(array)
        # with np.printoptions(threshold=np.inf, precision=3, suppress=True):
        #     print(array)
        
    # @staticmethod
    # def make_fill(array: np.ndarray, P: TestParameters) -> None:
    #     array.fill(P.strength * IMG_MP)

    # @staticmethod
    # def make_pulse(array: np.ndarray, P: TestParameters) -> None:
    #     T: float = time.time()
    #     phase_angle = T * math.pi * P.speed + P.phase
    #     value: float = (0.5 * math.sin(phase_angle) + 0.5) * P.strength * IMG_MP
    #     array.fill(value)

    # @staticmethod
    # def make_chase(array: np.ndarray, P: TestParameters, indices: np.ndarray) -> None:
    #     resolution: int = array.shape[1]
    #     adjusted_speed: float = P.speed * P.amount / 10.0
    #     wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
    #     time_offset: float = time.time() * adjusted_speed * 2 * math.pi

    #     # Vectorized
    #     phases = indices * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi
    #     array[0, :] = (0.5 * np.sin(phases) + 0.5) * P.strength * IMG_MP

    #     # # Old version for reference
    #     # for i in range(resolution):
    #     #     phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi
    #     #     value: float = 0.5 * math.sin(phase) + 0.5
    #     #     array[0, i] = value * P.strength * IMG_MP

    # @staticmethod
    # def make_lines(array: np.ndarray, P: TestParameters, indices: np.ndarray) -> None:
    #     resolution: int = array.shape[1]
    #     adjusted_speed: float = P.speed * P.amount / 10.0
    #     wave_phase_per_pixel: float = P.amount * 2 * math.pi / resolution
    #     time_offset: float = time.time() * adjusted_speed * 2 * math.pi

    #     # Vectorized
    #     phases = indices * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi + math.pi
    #     values = 0.5 * np.sin(phases) + 0.5
    #     array[0, :] = np.where(values < P.width, P.strength * IMG_MP, 0.0)

    #     # # Old version for reference
    #     # for i in range(resolution):
    #     #     phase: float = i * wave_phase_per_pixel - time_offset + P.phase * 2 * math.pi + math.pi
    #     #     value: float = 0.5 * math.sin(phase) + 0.5
    #     #     value = 1.0 if value < P.width else 0.0
    #     #     array[0, i] = value * P.strength * IMG_MP

    # @staticmethod
    # def make_random(array: np.ndarray, P: TestParameters, indices: np.ndarray) -> None:
    #     T: float = time.time() * P.speed
    #     sine_values: np.ndarray = np.sin(T + indices)
    #     array[0, :] = np.where(sine_values > 0.5, P.strength * IMG_MP, 0)

