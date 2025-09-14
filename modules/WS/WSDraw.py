from modules.WS.WSOutput import *
import numpy as np
from enum import Enum

from modules.WS.WSDataManager import WSDataManager

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
class WSDrawSettings():
    void_width: float =  0.05           # in normalized world width (0..1)
    void_edge: float = 0.01             # in normalized world width (0..1)
    use_void: bool = True

    pattern_width: float = 0.2          # in normalized world width (0..1)
    pattern_edge: float = 0.2           # in normalized world width (0..1)

    line_sharpness: float = 1.5         # higher is sharper
    line_speed: float = 1.5             # higher is faster
    line_width: float = 0.1             # in normalized world width (0..1)
    line_amount: float = 20.0            # number of lines


class WSDraw():
    def __init__(self, resolution: int, num_players: int, interval: float, data_manager: WSDataManager, settings: WSDrawSettings) -> None:
        self.resolution: int = resolution
        self.interval: float = interval
        self.settings: WSDrawSettings = settings
        self.num_players: int = num_players

        # self.draw_methods: DrawMethods = DrawMethods()
        self.data_manager: WSDataManager = data_manager

        self.Wh_L_array: np.ndarray = np.ones((self.resolution), dtype=WS_IMG_TYPE)
        self.Wh_R_array: np.ndarray = np.ones((self.resolution), dtype=WS_IMG_TYPE)
        self.blue_array: np.ndarray = np.ones((self.resolution), dtype=WS_IMG_TYPE)
        self.void_array: np.ndarray = np.zeros((self.resolution), dtype=WS_IMG_TYPE)

        self.output: WSOutput = WSOutput(self.resolution)

        self.left_pattern_times: dict[int, float] = {}
        self.right_pattern_times: dict[int, float] = {}
        for i in range(self.num_players):
            self.left_pattern_times[i] = 0.0
            self.right_pattern_times[i] = 0.0

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def update(self) -> None:
        self.make_voids(self.void_array, self.data_manager, self.settings, self.interval)
        self.make_patterns(self.Wh_L_array, self.Wh_R_array, self.blue_array, self.data_manager, self.settings)

        self.output.infos_0 = self.Wh_L_array[:]
        self.output.infos_1 = self.Wh_R_array[:]
        self.output.infos_2 = self.blue_array[:]
        self.output.infos_3 = 0.0

        # add overlay to second channel
        if self.settings.use_void:
            self.output.infos_3 = self.void_array[:]
            inverted_void_array = 1.0 - self.void_array
            WSDraw.blend_values(self.Wh_L_array, inverted_void_array, 0, BlendType.MULTIPLY)
            WSDraw.blend_values(self.Wh_R_array, inverted_void_array, 0, BlendType.MULTIPLY)
            WSDraw.blend_values(self.blue_array, inverted_void_array, 0, BlendType.MULTIPLY)
            WSDraw.blend_values(self.blue_array, self.void_array * 0.5, 0, BlendType.ADD)

        self.output.light_0 = self.Wh_L_array[:] + self.Wh_R_array[:]
        self.output.light_1 = self.blue_array[:]

    def reset(self) -> None:
        self.data_manager.reset()

    def get_output(self) -> WSOutput:
        return self.output


    @staticmethod
    def make_voids(array: np.ndarray, pose_metrics: WSDataManager, P: WSDrawSettings, interval: float) -> None:
        array -= interval * 4.0
        np.clip(array, 0, 1, out=array)

        world_positions: dict[int, float] = pose_metrics.world_positions
        ages: dict[int, float] = pose_metrics.ages
        pose_lengths: dict[int, float] = pose_metrics.pose_lengths

        for i in range(len(world_positions)):
            if pose_metrics.presents.get(i, False) == False:
                continue
            centre: float = (world_positions[i] + np.pi) / (2 * np.pi)
            length: float = pose_lengths[i]
            age: float = ages[i]

            strength: float = pow(min(age * 1.8, 1.0), 1.5)
            void_width: float = P.void_width * 0.5
            width: float = (void_width + length * void_width)
            edge: int = int(P.void_edge * len(array))
            WSDraw.draw_field(array, centre, width, strength, edge, BlendType.MAX)

    def make_patterns(self, W_L: np.ndarray, W_R: np.ndarray, blues: np.ndarray, pose_metrics: WSDataManager, P: WSDrawSettings) -> None:

        resolution: int = len(W_L)
        W_L.fill(0.0)
        W_R.fill(0.0)
        blues.fill(0.0)

        world_positions: dict[int, float] = pose_metrics.world_positions
        ages: dict[int, float] = pose_metrics.ages
        pose_lengths: dict[int, float] = pose_metrics.pose_lengths
        num_player_width: float = 1.0 / max(pose_metrics.smooth_num_active_players, 1)
        pattern_width: float = P.pattern_width * 0.25 + num_player_width * P.pattern_width * 0.25


        for i in range(pose_metrics.num_players):

            if pose_metrics.is_player_present(i):
                centre: float = (world_positions[i] + np.pi) / (2 * np.pi)

                age: float = ages[i]
                length: float = pose_lengths[i]
                left_elbow: float = pose_metrics.left_elbows[i]
                left_shoulder: float = pose_metrics.left_shoulders[i]
                rigt_elbow: float = pose_metrics.right_elbows[i]
                rigt_shoulder: float = pose_metrics.right_shoulders[i]

                age_pattern_speed: float = 0.25
                age_pattern_power: float = 0.75
                patt_width: float = pattern_width * pow(min(age * age_pattern_speed, 1.0), age_pattern_power)

                left_count: float = 5 + P.line_amount   * (1.0 - (np.cos(left_shoulder) + 1.0) / 2.0)
                rigt_count: float = 5 + P.line_amount   * (1.0 - (np.cos(rigt_shoulder) + 1.0) / 2.0)
                left_width: float = P.line_width        * ((np.cos(left_elbow) + 1.0) / 2.0) * ((np.cos(left_shoulder) + 1.0) / 2.0) * 0.8 + 0.2
                rigt_width: float = P.line_width        * ((np.cos(rigt_elbow) + 1.0) / 2.0) * ((np.cos(rigt_shoulder) + 1.0) / 2.0) * 0.8 + 0.2
                left_speed: float = P.line_speed        * (-np.sin(left_elbow))
                rigt_speed: float = P.line_speed        * ( np.sin(rigt_elbow))
                sharpness: float =  P.line_sharpness

                self.left_pattern_times[i] += self.interval * left_speed
                self.right_pattern_times[i] += self.interval * rigt_speed
                left_time: float = self.left_pattern_times[i]
                rigt_time: float = self.right_pattern_times[i]

                outer_edge: int = int(P.pattern_edge * resolution)
                void_width: float = (P.void_width * 0.5 + length * P.void_width * 0.5)
                inner_edge: int = int(void_width * resolution * 0.7)

                blend: BlendType = BlendType.MAX

                WSDraw.draw_waves(W_L,   centre, patt_width,  left_count, left_width, sharpness, left_time,  0,   inner_edge, outer_edge, blend)
                WSDraw.draw_waves(W_L,   centre, -patt_width, left_count, left_width, sharpness, left_time,  0,   outer_edge, inner_edge, blend)
                WSDraw.draw_waves(W_R,   centre, patt_width,  rigt_count, rigt_width, sharpness, rigt_time,  0.5, inner_edge, outer_edge, blend)
                WSDraw.draw_waves(W_R,   centre, -patt_width, rigt_count, rigt_width, sharpness, rigt_time,  0.5, outer_edge, inner_edge, blend)

                WSDraw.draw_waves(blues, centre, patt_width,  left_count, left_width, sharpness, -left_time, 0,   inner_edge, outer_edge, blend)
                WSDraw.draw_waves(blues, centre, -patt_width, left_count, left_width, sharpness, -left_time, 0,   outer_edge, inner_edge, blend)
                WSDraw.draw_waves(blues, centre, patt_width,  rigt_count, rigt_width, sharpness, -rigt_time, 0.5, inner_edge, outer_edge, blend)
                WSDraw.draw_waves(blues, centre, -patt_width, rigt_count, rigt_width, sharpness, -rigt_time, 0.5, outer_edge, inner_edge, blend)

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

        WSDraw.draw_edge(intensities, edge_left, 1.5, EdgeSide.LEFT)
        WSDraw.draw_edge(intensities, edge_right, 1.5, EdgeSide.RIGHT)
        WSDraw.apply_circular(array, intensities, pixel_start, blend)

    @staticmethod
    def draw_field(array: np.ndarray, centre: float, width: float, strength: float, edge: int, blend: BlendType) -> None:
        resolution: int = len(array)
        field_centre: int = int(centre * resolution)
        field_width: int = int(width * resolution)
        idx_start = int((field_centre - field_width // 2) % resolution)

        values = np.full(field_width, strength, dtype=array.dtype)

        edge_width: int = int(min(edge, field_width // 2))
        if edge_width > 0:
            WSDraw.draw_edge(values, edge_width, 1.5, EdgeSide.BOTH)

        WSDraw.apply_circular(array, values, idx_start, blend)

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
            WSDraw.blend_values(array, values, start_idx, blend)
        else:
            WSDraw.blend_values(array, values[:resolution - start_idx], start_idx, blend)
            WSDraw.blend_values(array, values[resolution - start_idx:], 0, blend)

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