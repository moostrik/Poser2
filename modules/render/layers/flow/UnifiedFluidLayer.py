"""Unified Fluid Layer - Single Navier-Stokes simulation for all players.

Combines inputs from all cameras/players into one wide simulation grid,
allowing fluid to naturally flow between player regions.

Grid Layout (3 players, gap_ratio=0.5):
    |  P0  | gap |  P1  | gap |  P2  |
    |  1W  | 0.5W|  1W  | 0.5W|  1W  |
    └─────────── 4W total ───────────┘
"""

from __future__ import annotations
from enum import IntEnum, auto
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Style, Fbo
from modules.gl.shaders import Blit, BlitRegion
from modules.render.layers.LayerBase import LayerBase
from modules.DataHub import DataHub, Stage
from modules.pose.Frame import Frame

from modules.flow import Visualizer, VisualisationFieldConfig
from modules.flow.fluid import FluidFlow, FluidFlowConfig
from modules.render.shaders import DensityColorize

from .FlowLayer import FlowLayer

from modules.utils.HotReloadMethods import HotReloadMethods


class UnifiedFluidSlot(LayerBase):
    """Minimal proxy for a single slot - fulfills LayerBase interface."""

    def __init__(self, parent: "UnifiedFluidLayer", slot_id: int):
        self._parent = parent
        self._slot_id = slot_id

    @property
    def texture(self) -> Texture:
        return self._parent.get_slot_texture(self._slot_id)

    def deallocate(self) -> None:
        pass  # Parent handles deallocation

    def update(self) -> None:
        pass  # Parent handles update

    def draw(self) -> None:
        self._parent.draw_slot(self._slot_id)


class UnifiedFluidDrawMode(IntEnum):
    """Draw modes for UnifiedFluidLayer."""
    DENSITY = 0
    DENSITY_RAW = auto()
    VELOCITY = auto()
    PRESSURE = auto()


@dataclass
class UnifiedFluidLayerConfig:
    """Configuration for UnifiedFluidLayer."""
    fps: float = 60.0
    num_players: int = 3
    gap_ratio: float = 0.5  # Gap = 0.5 * slot width
    draw_mode: UnifiedFluidDrawMode = UnifiedFluidDrawMode.DENSITY
    blend_mode: Style.BlendMode = Style.BlendMode.ADD
    simulation_scale: float = 0.25

    visualisation: VisualisationFieldConfig = field(default_factory=VisualisationFieldConfig)
    fluid_flow: FluidFlowConfig = field(default_factory=FluidFlowConfig)


class UnifiedFluidLayer(LayerBase):
    """Unified fluid simulation for all players on a single wide grid."""

    def __init__(self, data_hub: DataHub, flow_layers: dict[int, FlowLayer], colors: list[tuple[float, float, float, float]], config: UnifiedFluidLayerConfig | None = None) -> None:
        self._data_hub: DataHub = data_hub
        self._flow_layers: dict[int, FlowLayer] = flow_layers
        self.config: UnifiedFluidLayerConfig = config or UnifiedFluidLayerConfig()

        self._num_slots: int = len(flow_layers)
        self._delta_time: float = 1 / self.config.fps

        # Grid layout calculation:
        # total_width = num_slots + (num_slots - 1) * gap_ratio
        # For 3 slots with gap_ratio=0.5: total = 3 + 2*0.5 = 4
        self._total_width_ratio: float = self._num_slots + (self._num_slots - 1) * self.config.gap_ratio

        # Fluid simulation
        self._fluid_flow: FluidFlow = FluidFlow(self.config.simulation_scale, self.config.fluid_flow)
        self._visualizer: Visualizer = Visualizer(self.config.visualisation)

        # Colorization
        self._colorized_fbo: Fbo = Fbo()
        self._density_colorize: DensityColorize = DensityColorize()

        # Shaders
        self._blit: Blit = Blit()
        self._blit_region: BlitRegion = BlitRegion()

        # Per-slot output FBOs
        self._slot_fbos: dict[int, Fbo] = {i: Fbo() for i in range(self._num_slots)}

        # Colors for density colorization (up to 4 channels)
        self._colors: list[tuple[float, float, float, float]] = list(colors[:4])
        while len(self._colors) < 4:
            self._colors.append((0.0, 0.0, 0.0, 0.0))

        # Dimensions (set in allocate)
        self._slot_width: int = 0  # Width of one slot in pixels
        self._slot_height: int = 0
        self._unified_width: int = 0
        self._unified_height: int = 0
        self._sim_width: int = 0
        self._sim_height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0

        hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Layout Helpers (UV coordinates) ==========

    def _get_slot_uv(self, slot_id: int) -> tuple[float, float]:
        """Get slot position and width in UV coordinates [0, 1].

        Returns:
            (x, w) tuple in UV space
        """
        # Slot width in UV = 1 / total_width_ratio
        slot_w = 1.0 / self._total_width_ratio
        # Slot stride = slot_w * (1 + gap_ratio)
        slot_stride = slot_w * (1.0 + self.config.gap_ratio)
        slot_x = slot_id * slot_stride
        return slot_x, slot_w

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Full unified visualization texture."""
        return self._visualizer.texture

    @property
    def unified_density(self) -> Texture:
        """Full unified colorized density."""
        return self._colorized_fbo.texture

    def get_slot(self, slot_id: int) -> UnifiedFluidSlot:
        """Get a slot proxy that implements the LayerBase interface."""
        return UnifiedFluidSlot(self, slot_id)

    def get_slot_texture(self, slot_id: int) -> Texture:
        """Get cropped output texture for a specific slot."""
        if slot_id in self._slot_fbos:
            return self._slot_fbos[slot_id].texture
        return self._colorized_fbo.texture

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate unified simulation.

        Args:
            width: Single slot width
            height: Slot height
            internal_format: Ignored
        """
        self._slot_width = int(width * self.config.simulation_scale)
        self._slot_height = int(height * self.config.simulation_scale)

        # Unified grid = slots + gaps
        self._sim_width = int(self._slot_width * self._total_width_ratio)
        self._sim_height = self._slot_height

        # Density at higher resolution
        self._density_width = int(width * self._total_width_ratio)
        self._density_height = height

        # Unified grid dimensions for reference
        self._unified_width = self._density_width
        self._unified_height = self._density_height

        # Allocate fluid simulation
        self._fluid_flow.allocate(self._sim_width, self._sim_height, self._density_width, self._density_height)
        self._visualizer.allocate(self._sim_width, self._sim_height)

        # Colorization FBO
        self._colorized_fbo.allocate(self._density_width, self._density_height, GL_RGBA16F)
        self._density_colorize.allocate()

        # Shaders
        self._blit.allocate()
        self._blit_region.allocate()

        # Per-slot output FBOs (no bleed for now - exact slot size)
        for i in range(self._num_slots):
            self._slot_fbos[i].allocate(width, height, GL_RGBA16F)

    def deallocate(self) -> None:
        self._fluid_flow.deallocate()
        self._visualizer.deallocate()
        self._colorized_fbo.deallocate()
        self._density_colorize.deallocate()
        self._blit.deallocate()
        self._blit_region.deallocate()
        for fbo in self._slot_fbos.values():
            fbo.deallocate()

    def reset(self) -> None:
        self._fluid_flow.reset()

    # ========== Update ==========

    def update(self) -> None:
        """Update unified fluid simulation with inputs from all flow layers."""
        import numpy as np

        # Hot-reload config overrides
        self.config.fluid_flow.vel_speed = 0.1
        self.config.fluid_flow.vel_decay = 6.0

        self.config.fluid_flow.vel_vorticity = 10
        self.config.fluid_flow.vel_vorticity_radius = 3.0
        self.config.fluid_flow.vel_viscosity = 8.0
        self.config.fluid_flow.vel_viscosity_iter = 40

        self.config.fluid_flow.den_speed = 1.1
        self.config.fluid_flow.den_decay = 12

        self.config.fluid_flow.tmp_speed = 0.33
        self.config.fluid_flow.tmp_decay = 3.0

        self.config.fluid_flow.prs_speed = 0.0
        self.config.fluid_flow.prs_decay = 0.0
        self.config.fluid_flow.prs_iterations = 40

        self.config.fluid_flow.tmp_buoyancy = 0.0
        self.config.fluid_flow.tmp_weight = -10.0

        draw_mode: UnifiedFluidDrawMode = UnifiedFluidDrawMode.VELOCITY


        # Get pose data for motion/similarity modulation (like original FluidLayer)
        motions: dict[int, float] = {}
        similarities: dict[int, np.ndarray] = {}
        motion_gates: dict[int, np.ndarray] = {}

        for cam_id in self._flow_layers.keys():
            pose: Frame | None = self._data_hub.get_pose(Stage.LERP, cam_id)
            motions[cam_id] = pose.angle_motion.value if pose else 0.0
            similarities[cam_id] = pose.similarity.values if pose else np.full((self._num_slots,), 0.0)
            motion_gates[cam_id] = pose.motion_gate.values if pose else np.full((self._num_slots,), 0.0)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # For each slot region, add inputs from ALL flow layers (cross-camera influence)
        # This matches original FluidLayer behavior where all cameras contribute to each simulation
        for slot_id in range(self._num_slots):
            slot_x, slot_w = self._get_slot_uv(slot_id)

            for cam_id, flow_layer in self._flow_layers.items():
                channel = cam_id % 4

                if cam_id == slot_id:
                    # Primary input: this is the slot's own camera
                    vel_strength = 0.1
                    den_strength = motions[cam_id]
                else:
                    # Cross-camera input: modulated by similarity and motion gate
                    m_s = similarities[slot_id][cam_id] * motion_gates[slot_id][cam_id]
                    vel_strength = 0.1 * m_s
                    den_strength = m_s

                # Add velocity to this slot region
                self._fluid_flow.add_velocity_region(
                    flow_layer.velocity,
                    slot_x, 0.0, slot_w, 1.0,
                    strength=vel_strength
                )

                # Add density to this slot region (per-channel coloring)
                self._fluid_flow.add_density_channel_region(
                    flow_layer.density,
                    channel,
                    slot_x, 0.0, slot_w, 1.0,
                    strength=den_strength
                )

        # Clamp density
        self._fluid_flow.clamp_density(0.0, 1.1)

        # Run simulation
        self._fluid_flow.update(self._delta_time)

        # Colorize density
        self._colorize_density()

        # Extract slot regions
        self._extract_slots()

        # Update visualizer
        self._visualizer.update(self._get_draw_texture())

        Style.pop_style()

    def _colorize_density(self) -> None:
        """Colorize density channels with track colors."""
        self._colorized_fbo.begin()
        self._density_colorize.use(self._fluid_flow.density, self._colors)
        self._colorized_fbo.end()

    def _extract_slots(self) -> None:
        """Extract cropped regions for each slot."""
        for slot_id, fbo in self._slot_fbos.items():
            slot_x, slot_w = self._get_slot_uv(slot_id)

            fbo.begin()
            self._blit_region.use(self._colorized_fbo.texture, slot_x, 0.0, slot_w, 1.0)
            fbo.end()

    def _get_draw_texture(self) -> Texture:
        """Get texture for visualization based on draw mode."""
        if self.config.draw_mode == UnifiedFluidDrawMode.DENSITY:
            return self._colorized_fbo.texture
        elif self.config.draw_mode == UnifiedFluidDrawMode.DENSITY_RAW:
            return self._fluid_flow.density
        elif self.config.draw_mode == UnifiedFluidDrawMode.VELOCITY:
            return self._fluid_flow.velocity
        elif self.config.draw_mode == UnifiedFluidDrawMode.PRESSURE:
            return self._fluid_flow.pressure
        return self._colorized_fbo.texture

    # ========== Drawing ==========

    def draw(self) -> None:
        """Draw full unified output."""
        Style.push_style()
        Style.set_blend_mode(self.config.blend_mode)
        if self.texture.allocated:
            self._blit.use(self.texture)
        Style.pop_style()

    def draw_slot(self, slot_id: int) -> None:
        """Draw cropped output for a specific slot."""
        Style.push_style()
        Style.set_blend_mode(self.config.blend_mode)
        texture = self.get_slot_texture(slot_id)
        if texture.allocated:
            self._blit.use(texture)
        Style.pop_style()
