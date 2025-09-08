    
import numpy as np
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
        
        end_idx = start_idx + len(values)
        if start_idx < 0 or end_idx > resolution:
            raise ValueError("start_idx and end_idx must be within the bounds of the array.")
        
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