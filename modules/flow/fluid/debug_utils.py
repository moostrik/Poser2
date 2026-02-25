"""Debug utilities for fluid simulation testing."""
from __future__ import annotations

import numpy as np
from OpenGL.GL import *  # type: ignore

from modules.gl import Texture


def generate_debug_obstacle_mask(width: int, height: int) -> np.ndarray:
    """Generate a test obstacle mask with circle, triangle, cross, and diagonal line.

    Each shape occupies one quadrant of the image:
        Top-left:     Circle
        Top-right:    Triangle (pointing up)
        Bottom-left:  Cross
        Bottom-right: Diagonal line

    Args:
        width:  Mask width in pixels.
        height: Mask height in pixels.

    Returns:
        uint8 numpy array of shape (height, width), 255 = obstacle.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Coordinate grids (row=y, col=x)
    yy, xx = np.ogrid[0:height, 0:width]

    # ---- Circle (top-left quadrant) ----
    cx, cy, r = width // 4, height * 3 // 4, min(width, height) // 8
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask[dist_sq <= r * r] = 255

    # ---- Triangle (top-right quadrant) ----
    tx, ty = width * 3 // 4, height * 3 // 4
    s = min(width, height) // 7
    top_y = ty + s
    bot_y = ty - int(s * 0.6)
    for y_px in range(max(0, bot_y), min(height, top_y + 1)):
        for x_px in range(max(0, tx - s), min(width, tx + s + 1)):
            t_frac = (y_px - bot_y) / max(1, top_y - bot_y)
            half_w = s * (1.0 - t_frac)
            if abs(x_px - tx) <= half_w:
                mask[y_px, x_px] = 255

    # ---- Cross (bottom-left quadrant) ----
    cx2, cy2 = width // 4, height // 4
    arm = min(width, height) // 8
    thick = max(2, min(width, height) // 40)
    mask[max(0, cy2 - thick):min(height, cy2 + thick + 1),
         max(0, cx2 - arm):min(width, cx2 + arm + 1)] = 255
    mask[max(0, cy2 - arm):min(height, cy2 + arm + 1),
         max(0, cx2 - thick):min(width, cx2 + thick + 1)] = 255

    # ---- Diagonal line (bottom-right quadrant) ----
    lx, ly = width * 3 // 4, height // 4
    length = min(width, height) // 6
    thick_l = max(2, min(width, height) // 50)
    for t in np.linspace(-1, 1, length * 4):
        px = int(lx + t * length * 0.5)
        py = int(ly + t * length * 0.3)
        for dx in range(-thick_l, thick_l + 1):
            for dy in range(-thick_l, thick_l + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < width and 0 <= ny < height:
                    mask[ny, nx] = 255

    return mask


def upload_debug_obstacle(fluid, width: int, height: int) -> None:
    """Generate a debug obstacle mask and inject it into a fluid simulation.

    Works with both FluidFlow (2D) and FluidFlow3D -- any object with
    a ``set_obstacle(texture)`` method.

    Args:
        fluid: FluidFlow or FluidFlow3D instance.
        width:  Obstacle mask width in pixels.
        height: Obstacle mask height in pixels.
    """
    mask = generate_debug_obstacle_mask(width, height)

    temp = Texture()
    temp.allocate(width, height, GL_R8)
    glBindTexture(GL_TEXTURE_2D, temp.tex_id)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RED, GL_UNSIGNED_BYTE, mask)
    glBindTexture(GL_TEXTURE_2D, 0)

    fluid.set_obstacle(temp)
    temp.deallocate()
