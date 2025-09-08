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

from modules.av.DrawMethods import DrawMethods, BlendType

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class TestParameters():
    void_width: float =  0.05
    pattern_width: float = 0.2
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
        
        self.draw_methods: DrawMethods = DrawMethods()
        self.pose_metrics: MultiPoseMetrics = MultiPoseMetrics(settings)

        self.white_array: np.ndarray = np.ones((self.resolution), dtype=IMG_TYPE)
        self.blue_array: np.ndarray = np.zeros((self.resolution), dtype=IMG_TYPE)
        self.void_array: np.ndarray = np.zeros((self.resolution), dtype=IMG_TYPE)
        self.output_img: np.ndarray = np.zeros((1, self.resolution, 3), dtype=IMG_TYPE)

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def update(self, poses: list[Pose]) -> np.ndarray:
        
        self.pose_metrics.update(poses)

        self.make_voids(self.void_array, self.pose_metrics, self.draw_methods, self.parameters)

        self.blue_array.fill(0.0)
        self.white_array.fill(1.0)
        # add overlay to second channel 
        if self.parameters.show_void:
            inverted_void_array = 1.0 - self.void_array
            self.draw_methods.blend_values(self.white_array, inverted_void_array, 0, BlendType.MULTIPLY)  
            self.draw_methods.blend_values(self.blue_array, inverted_void_array, 0, BlendType.MULTIPLY)
            self.draw_methods.blend_values(self.blue_array, self.void_array * 0.5, 0, BlendType.ADD)
            
        self.output_img[0, :, 0] = self.white_array[:]
        self.output_img[0, :, 1] = self.blue_array[:]

        return self.output_img


    def reset(self) -> None:
        self.pose_metrics.reset()

    @staticmethod
    def make_voids(array: np.ndarray, pose_metrics: MultiPoseMetrics, draw: DrawMethods, P: TestParameters) -> None:
        array -= P.interval * 4.0

        world_positions: dict[int, float] = pose_metrics.world_positions
        ages: dict[int, float] = pose_metrics.ages
        pose_lengths: dict[int, float] = pose_metrics.pose_lengths
        
        for i in range(len(world_positions)):
            if world_positions[i] is not None and pose_lengths[i] is not None and ages[i] is not None:
                age = pow(min(ages[i] * 1.8, 1.0), 1.5)
                void_width: float = P.void_width * 0.036
                width = (void_width + pose_lengths[i] * void_width)
                draw.draw_field_with_edge(array, world_positions[i], width, age, 20, BlendType.MAX)


            

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