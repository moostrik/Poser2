    
import numpy as np
import time
import math
from enum import Enum
from modules.utils.HotReloadMethods import HotReloadMethods    


class BlendType(Enum):
    NONE = "replace"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    MAX = "max"
    MIN = "min"
    NON_ZERO = "non_zero"
    
class DrawMethods:
    def __init__(self) -> None:

        self.hot_reloader = HotReloadMethods(self.__class__)

    
        
    @staticmethod        
    def draw_field(array: np.ndarray, centre: float, width: float, value: float, blend: BlendType) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int = int(width * resolution)
        idx_start = int((field_centre - field_width // 2) % resolution)
        
        values = np.full(field_width, value, dtype=array.dtype)
        DrawMethods.apply_circular(array, values, idx_start, blend)
            
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
        
        DrawMethods.apply_circular(array, values, idx_start, blend)
            
    @staticmethod
    def apply_circular(array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType) -> None:
        resolution: int = len(array)
        
        end_idx: int = (start_idx + len(values)) % resolution
        if start_idx < end_idx:
           DrawMethods.blend_values(array, values, start_idx, blend)
        else:
            DrawMethods.blend_values(array, values[:resolution - start_idx], start_idx, blend)
            DrawMethods.blend_values(array, values[resolution - start_idx:], 0, blend)

    @staticmethod
    def blend_values(array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType) -> None:
        resolution: int = len(array)
        print(resolution)
        
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