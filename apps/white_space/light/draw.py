"""Low-level 1-D LED drawing primitives — used by all Composition implementations."""

from enum import Enum

import numpy as np

PI:     float = np.pi
TWOPI:  float = 2.0 * np.pi
HALFPI: float = np.pi / 2.0


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
# Primitive drawing functions (all operate in-place on a 1-D float32 array)
# ---------------------------------------------------------------------------

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
        thick_mode: bool  = thickness > 0.5
        thick_trim: float = (thickness - 0.5) * 2.0 if thick_mode else 1.0 - thickness * 2.0
        thick_time_offset:  float = (thickness - 0.5) * -TWOPI if thick_mode else thickness * TWOPI
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

    draw_edge(intensities, edge_left,  1.5, EdgeSide.LEFT)
    draw_edge(intensities, edge_right, 1.5, EdgeSide.RIGHT)
    apply_circular(array, intensities, pixel_start, blend)


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
        draw_edge(values, edge_width, 1.5, EdgeSide.BOTH)
    apply_circular(array, values, idx_start, blend)


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


def apply_circular(
    array: np.ndarray, values: np.ndarray, start_idx: int, blend: BlendType,
) -> None:
    resolution: int = len(array)
    start_idx = start_idx % resolution
    end_idx: int = (start_idx + len(values)) % resolution
    if start_idx < end_idx:
        blend_values(array, values, start_idx, blend)
    else:
        blend_values(array, values[:resolution - start_idx], start_idx, blend)
        blend_values(array, values[resolution - start_idx:], 0, blend)


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
