"""Fluid3D Layer - 3D Navier-Stokes fluid simulation with cross-camera flow inputs.

Volumetric extension of FluidLayer. Receives 2D velocity, density, and
temperature from all FlowLayers and runs a full 3D Navier-Stokes simulation
with gaussian depth injection.  The composited 2D density output is already
RGB-coloured — no post-composite colorisation step is needed.
"""

# Standard library imports
from __future__ import annotations
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import *  # type: ignore
import numpy as np

# Local application imports
from modules.gl import Texture, Style, Text
from ..layer_base import LayerBase, Blit
from modules.board import HasFrames
from modules.pose import Frame, Similarity, MotionGate, AngleMotion, Points2D, PointLandmark

from modules.settings import Field, BaseSettings, Group
from modules.flow import FluidFlow, FluidFlow3D, FluidFlowSettings

from modules.utils import HotReloadMethods

from .flow_layer import FlowLayer
from ...color_settings import ColorSettings

class FluidSimEngine(IntEnum):
    """Simulation engine selection.

    FLUID_2D: 2D Navier-Stokes (FluidFlow) — the hd_trio reference look, no depth
    FLUID_3D: 3D volumetric Navier-Stokes (FluidFlow3D)
    """
    FLUID_2D = 0
    FLUID_3D = auto()


# Map enum -> class for engine creation
_ENGINE_CLASSES: dict[FluidSimEngine, type] = {
    FluidSimEngine.FLUID_2D: FluidFlow,
    FluidSimEngine.FLUID_3D: FluidFlow3D,
}


class Fluid3DLayerSettings(BaseSettings):
    """Configuration for Fluid3DLayer (3D fluid simulation)."""
    stage: Field[int] =                     Field(0, description="Pipeline stage for pose data")
    num_players: Field[int] =               Field(3, min=1, max=8, access=Field.INIT)
    sim_engine: Field[FluidSimEngine] =     Field(FluidSimEngine.FLUID_3D)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)

    fluid_flow    = Group(FluidFlowSettings)


class Fluid3DLayer(LayerBase):
    """3D fluid simulation layer with cross-camera flow inputs.

    Receives velocity, density, and temperature from all FlowLayers and runs
    a volumetric Navier-Stokes fluid simulation:
    1. Aggregate inputs from connected FlowLayers
    2. Inject velocity, density (RGB colour baked in), temperature into the 3D volume
    3. Run 3D advection, pressure projection, viscosity, buoyancy, vorticity
    4. Composite the 3D density volume -> coloured 2D output

    The density volume stores coloured ink directly (RGBA16F GL_TEXTURE_2D_ARRAY,
    output resolution); Composite3DDensity flattens it to the final RGBA16F 2D output.

    Cross-camera support:
        Each Fluid3DLayer receives a dict of all FlowLayers, enabling
        cross-camera influence (e.g., motion from camera 0 affecting
        fluid in camera 1's output).

    Usage:
        # In RenderManager:
        flow_layers = {i: FlowLayer(...) for i in range(num_cams)}
        fluid3d = Fluid3DLayer(cam_id, board, flow_layers, colors, settings)

        # Each frame:
        fluid3d.update()
        density_texture = fluid3d.density
    """

    def __init__(self, cam_id: int, board: HasFrames, flow_layers: dict[int, FlowLayer], settings: Fluid3DLayerSettings, color_settings: ColorSettings) -> None:
        """Initialize 3D fluid layer.

        Args:
            cam_id: Camera ID for this layer's output
            board: Board for pose data
            flow_layers: Dict of all FlowLayers (cam_id -> FlowLayer)
            settings: Layer configuration
            color_settings: Per-camera colors for density colorization
        """
        self._cam_id: int = cam_id
        self._board: HasFrames = board
        self._flow_layers: dict[int, FlowLayer] = flow_layers
        self.settings: Fluid3DLayerSettings = settings or Fluid3DLayerSettings()
        self._color_settings: ColorSettings = color_settings

        # Simulation engine (swappable at runtime)
        self._fluid_flow: FluidFlow | FluidFlow3D = self._create_engine()
        self._alloc_width: int = 0
        self._alloc_height: int = 0
        self._swap_engine_pending: bool = False

        # Legend for the debug depth view (hue = layer)
        self._legend_text: Text = Text()

        # Smoothed head position for the head-coupled parallax eye
        self._eye_x: float = 0.0
        self._eye_y: float = 0.0

        # Defer engine swap to GL thread (callbacks fire from websocket thread)
        self.settings.bind(Fluid3DLayerSettings.sim_engine, lambda _: self._request_swap_engine())

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def _create_engine(self) -> FluidFlow | FluidFlow3D:
        """Instantiate the selected simulation engine with the shared settings."""
        cls = _ENGINE_CLASSES[self.settings.sim_engine]
        return cls(self.settings.fluid_flow)

    def _request_swap_engine(self) -> None:
        """Request engine swap (deferred to next update on GL thread)."""
        self._swap_engine_pending = True

    def _swap_engine(self) -> None:
        """Swap simulation engine at runtime, preserving allocation state.

        Must run on the GL thread (called from update).
        """
        was_allocated = self._alloc_width > 0
        if was_allocated:
            self._fluid_flow.deallocate()
        self._fluid_flow = self._create_engine()
        if was_allocated:
            self._fluid_flow.allocate(self._alloc_width, self._alloc_height)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Composited 2D density output (RGBA16F, colour baked in)."""
        return self._fluid_flow.density

    @property
    def density(self) -> Texture:
        """Composited 2D density (RGBA16F, 3D->2D composite output)."""
        return self._fluid_flow.density

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate 3D fluid simulation buffers.

        Args:
            width: Processing width
            height: Processing height
            internal_format: Ignored (formats determined by FluidFlow3D)
        """
        self._alloc_width = width
        self._alloc_height = height
        self._fluid_flow.allocate(width, height)
        self._legend_text.allocate(font_size=18)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._fluid_flow.deallocate()
        self._legend_text.deallocate()

    def reset(self) -> None:
        """Reset fluid simulation state."""
        self._fluid_flow.reset()

    # ========== Processing ==========

    def update(self) -> None:
        """Update 3D fluid simulation with inputs from all flow layers."""
        # Handle deferred engine swap (must run on GL thread)
        if self._swap_engine_pending:
            self._swap_engine_pending = False
            self._swap_engine()

        # Get motion data from pose
        pose: Frame | None = self._board.get_frame(self.settings.stage, self._cam_id)
        similarities: np.ndarray = (
            np.nan_to_num(pose[Similarity].values) if pose is not None and Similarity in pose
            else np.full((self.settings.num_players,), 0.0)
        )
        motion_gates: np.ndarray = (
            pose[MotionGate].values if pose is not None and MotionGate in pose
            else np.full((self.settings.num_players,), 0.0)
        )
        motion: float = pose[AngleMotion].value if pose is not None and AngleMotion in pose else 0.0
        m_s = similarities

        # self.settings.fluid_flow.density.fade_time = 18.0 - (pow(motion, 2.0) * 14.0)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Aggregate inputs from all flow layers (cross-camera)
        vel_strength: float
        den_strength: float
        for cam_id, flow_layer in self._flow_layers.items():
            if cam_id == self._cam_id:
                vel_strength = motion
                den_strength = motion
            else:
                vel_strength = m_s[cam_id]
                den_strength = m_s[cam_id]

            # Add the flow as one velocity input (3D synthesizes w from it)
            self._fluid_flow.add_velocity(flow_layer.velocity, vel_strength)

            # Add density to per-camera channel (R=cam0, G=cam1, B=cam2, A=cam3)
            channel: int = cam_id % 4
            # self._fluid_flow.add_density_channel(flow_layer.magnitude, channel, den_strength)
            self._fluid_flow.add_density(flow_layer.density, den_strength)

            # Add temperature from each flow layer
            self._fluid_flow.add_temperature(flow_layer.temperature, 1.0)

        # Head-coupled parallax eye (3D engine only)
        self._update_eye(pose)

        # Update 3D fluid simulation (handles its own reallocation internally)
        self._fluid_flow.update()

        Style.pop_style()

    def _update_eye(self, pose: Frame | None) -> None:
        """Steer the parallax eye toward the tracked head.

        Head column/row follow white_space's eye-centre convention: the eyes'
        mean, the nose when they are missing, easing back to centre when both
        are. Offsets are ±1 across the frame (vertical at half range).
        """
        if not isinstance(self._fluid_flow, FluidFlow3D):
            return
        target_x, target_y = 0.0, 0.0
        if pose is not None and Points2D in pose:
            pts = pose[Points2D].values
            head = pts[[PointLandmark.left_eye, PointLandmark.right_eye]].mean(axis=0)
            if np.isnan(head).any():
                head = pts[PointLandmark.nose]
            if not np.isnan(head).any():
                target_x = (float(head[0]) - 0.5) * 2.0
                target_y = float(head[1]) - 0.5
        # ~0.4 s ease keeps keypoint jitter out of the eye (render thread only)
        dt = 1.0 / max(1, self.settings.fluid_flow.fps)
        k = 1.0 - 0.01 ** (dt / 0.4)
        self._eye_x += (target_x - self._eye_x) * k
        self._eye_y += (target_y - self._eye_y) * k
        self._fluid_flow.set_eye(self._eye_x, self._eye_y)

    # ========== Rendering ==========

    def draw(self) -> None:
        """Draw with configured blend mode."""
        Style.push_style()
        Style.set_blend_mode(self.settings.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        if self.settings.fluid_flow.depth.debug_depth and self._legend_text.allocated:
            self._draw_depth_legend()
        Style.pop_style()

    def _draw_depth_legend(self) -> None:
        """Hue legend for the debug depth view: violet = front ... red = back."""
        vp = glGetIntegerv(GL_VIEWPORT)
        w, h = int(vp[2]), int(vp[3])
        if w <= 0 or h <= 0:
            return
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        n = 12
        swatch_w = 16
        x = 16
        y = h - 46
        self._legend_text.draw_box_text(x, y, "FRONT", (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), w, h)
        bar_x = x + 76
        for i in range(n):
            hue = 0.8 * (1.0 - i / (n - 1))  # violet (front) -> red (back), matching the shader
            h6 = hue * 6.0
            xx = 1.0 - abs(h6 % 2.0 - 1.0)
            if   h6 < 1.0: rgb = (1.0, xx, 0.0)
            elif h6 < 2.0: rgb = (xx, 1.0, 0.0)
            elif h6 < 3.0: rgb = (0.0, 1.0, xx)
            elif h6 < 4.0: rgb = (0.0, xx, 1.0)
            elif h6 < 5.0: rgb = (xx, 0.0, 1.0)
            else:          rgb = (1.0, 0.0, xx)
            self._legend_text.draw_box_text(bar_x + i * swatch_w, y, " ",
                                            (0.0, 0.0, 0.0, 0.0), (*rgb, 1.0), w, h)
        self._legend_text.draw_box_text(bar_x + n * swatch_w + 8, y, "BACK",
                                        (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), w, h)
