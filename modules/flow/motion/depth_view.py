"""The camera's depth on the flow's view: what the 3D flow measures z from."""

from dataclasses import dataclass

from modules.gl import Texture


@dataclass
class DepthView:
    """The depth on the view the flow measures on. `texture` is metres in red, 0 where there is no
    depth, on the view; `view_span` the metres one view width spans per metre of depth, which puts z
    in view widths like x/y; `depth_step` the uv between two of the depth's own samples in the view,
    so a window over the depth steps over real samples, not copies (0 steps one flow texel)."""
    texture: Texture
    view_span: float
    depth_step: tuple[float, float]
