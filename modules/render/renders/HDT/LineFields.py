
import numpy as np
import traceback

from dataclasses import dataclass
from enum import Enum
from time import time

from OpenGL.GL import * # type: ignore
from modules.render.renders.BaseRender import BaseRender, Rect
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Image import Image
from modules.pose.smooth.PoseSmoothData import PoseSmoothData, PoseJoint

from modules.utils.HotReloadMethods import HotReloadMethods

PI: float = np.pi
TWOPI: float = 2 * np.pi
HALFPI: float = np.pi / 2

class EdgeSide(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    BOTH = 3

class BlendType(Enum):
    NONE = "replace"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    MAX = "max"
    MIN = "min"
    NON_ZERO = "non_zero"

@dataclass
class LineFieldsSettings():
    line_sharpness: float = 1.5         # higher is sharper
    line_speed: float = 1.5             # higher is faster
    line_width: float = 0.1             # in normalized world width (0..1)
    line_amount: float = 20.0           # number of lines


class LineFields(BaseRender):
    def __init__(self, smooth_data: PoseSmoothData, cam_id: int) -> None:
        self.smooth_data: PoseSmoothData = smooth_data
        self.cam_id: int = cam_id
        self.fbo: Fbo = Fbo()

        self.left_image: Image = Image()
        self.rigt_image: Image = Image()

        self.Wh_L_array: np.ndarray 
        self.Wh_R_array: np.ndarray
        
        self.left_pattern_time: float = 0.0
        self.right_pattern_time: float = 0.0
        
        self.interval = 1.0 / 60.0

        self.hot_reloader = HotReloadMethods(self.__class__, True)
    
    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        
        self.left_image.allocate(1, height, GL_R32F)
        self.rigt_image.allocate(1, height, GL_R32F)
                                   
        self.Wh_L_array: np.ndarray = np.ones((height), dtype=np.float32)
        self.Wh_R_array: np.ndarray = np.ones((height), dtype=np.float32)
        
    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.left_image.deallocate()
        self.rigt_image.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)
    
    def clear(self) -> None:
        BaseRender.setView(self.fbo.width, self.fbo.height)
        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.fbo.end()
        
    def update(self) -> None:
        
        self.clear()
        if not self.smooth_data.get_is_active(self.cam_id):
            return
        # return
        
        try:
            self.make_patterns(self.Wh_L_array, self.Wh_R_array)
        except Exception as e:
            print(f"Error in LineFields.make_patterns: {e}")
            traceback.print_exc()
            return

        self.left_image.set_image(self.Wh_L_array.reshape(-1, 1))
        self.rigt_image.set_image(self.Wh_R_array.reshape(-1, 1))
        self.left_image.update()
        self.rigt_image.update()
        
        BaseRender.setView(self.fbo.width, self.fbo.height)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        self.fbo.begin()
        glColor3f(1.0, 0.0, 0.0)
        self.left_image.draw(0,0,self.fbo.width, self.fbo.height)
        glColor3f(0.0, 1.0, 1.0)
        self.rigt_image.draw(0,0,self.fbo.width, self.fbo.height)
        self.fbo.end()
        
        glColor3f(1.0, 1.0, 1.0)
        glDisable(GL_BLEND)

    def make_patterns(self, W_L: np.ndarray, W_R: np.ndarray) -> None:
        id = self.cam_id

        P: LineFieldsSettings = LineFieldsSettings()
        P.line_sharpness = 4
        P.line_speed = 0
        P.line_width = 1.0
        P.line_amount = 2.0


        resolution: int = len(W_L)
        W_L.fill(0.0)
        W_R.fill(0.0)

        age = time()
        left_elbow: float = self.smooth_data.get_smoothed_angle(id, PoseJoint.left_elbow, symmetric=True)
        left_shoulder: float = self.smooth_data.get_smoothed_angle(id, PoseJoint.left_shoulder, symmetric=True)
        rigt_elbow: float = self.smooth_data.get_smoothed_angle(id, PoseJoint.right_elbow, symmetric=True)
        rigt_shoulder: float = self.smooth_data.get_smoothed_angle(id, PoseJoint.right_shoulder, symmetric=True)

        age_pattern_speed: float = 0.25
        age_pattern_power: float = 0.75

        left_count: float = 5 + P.line_amount   * (1.0 - (np.cos(left_shoulder) + 1.0) / 2.0)
        rigt_count: float = 5 + P.line_amount   * (1.0 - (np.cos(rigt_shoulder) + 1.0) / 2.0)
        left_width: float = P.line_width        * ((np.cos(left_elbow) + 1.0) / 2.0) * ((np.cos(left_shoulder) + 1.0) / 2.0) * 0.8 + 0.2
        rigt_width: float = P.line_width        * ((np.cos(rigt_elbow) + 1.0) / 2.0) * ((np.cos(rigt_shoulder) + 1.0) / 2.0) * 0.8 + 0.2
        left_speed: float = P.line_speed        * ( np.sin(left_elbow))
        rigt_speed: float = P.line_speed        * ( np.sin(rigt_elbow))
        left_sharpness: float =  1.0 + P.line_sharpness * (abs(left_elbow) / PI)
        rigt_sharpness: float =  1.0 + P.line_sharpness * (abs(rigt_elbow) / PI)

        self.left_pattern_time += self.interval * left_speed
        self.right_pattern_time += self.interval * rigt_speed
        left_time: float = self.left_pattern_time
        rigt_time: float = self.right_pattern_time

        blend: BlendType = BlendType.MAX

        LineFields.draw_waves(W_L,   0.0, 1.0,   left_count, left_width, left_sharpness, left_time,  0.0, 0.0, 0.0, blend)
        LineFields.draw_waves(W_R,   0.0, -1.0,  rigt_count, rigt_width, rigt_sharpness, rigt_time,  0.5, 0.0, 0.0, blend)

    @staticmethod
    def draw_waves(array: np.ndarray, anchor: float, span: float, num_waves: float,
                   thickness: float, sharpness: float, time_value: float, phase: float,
                   edge_left: int, edge_right: int,blend: BlendType) -> None:
        # optional: work with smoothsteps

        resolution: int = len(array)
        pixel_anchor: int = int(anchor * resolution)
        pixel_span: int = abs(int(span * resolution))
        if pixel_span == 0 or thickness <= 0.0:
            return

        if thickness >= 1.0:
            intensities: np.ndarray = np.ones(resolution, dtype=array.dtype)
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

        # LineFields.draw_edge(intensities, edge_left, 1.5, EdgeSide.LEFT)
        # LineFields.draw_edge(intensities, edge_right, 1.5, EdgeSide.RIGHT)
        LineFields.apply_circular(array, intensities, pixel_start, blend)

    @staticmethod
    def draw_field(array: np.ndarray, centre: float, width: float, strength: float, edge: int, blend: BlendType) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int = int(width * resolution)
        idx_start = int((field_centre - field_width // 2) % resolution)

        values = np.full(field_width, strength, dtype=array.dtype)

        edge_width: int = int(min(edge, field_width // 2))
        if edge_width > 0:
            LineFields.draw_edge(values, edge_width, 1.5, EdgeSide.BOTH)

        LineFields.apply_circular(array, values, idx_start, blend)

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
            LineFields.blend_values(array, values, start_idx, blend)
        else:
            LineFields.blend_values(array, values[:resolution - start_idx], start_idx, blend)
            LineFields.blend_values(array, values[resolution - start_idx:], 0, blend)

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