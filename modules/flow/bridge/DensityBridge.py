"""Density Bridge.

Combines RGB density with velocity magnitude for fluid simulation.
Composes VelocityProcessor for velocity smoothing.

Ported from ofxFlowTools ftDensityBridgeFlow.h
"""
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Fbo
from .. import FlowBase, FlowConfigBase
from .VelocityBridge import VelocityBridge, VelocityBridgeConfig
from .shaders.DensityBridgeShader import DensityBridgeShader
from ..shaders.HSV import HSV
from ..shaders.Scale import Scale

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class DensityBridgeConfig(FlowConfigBase):
    """Configuration for density bridge.

    Velocity parameters propagate to internal VelocityProcessor.
    """
    # Velocity processing parameters
    velocity: VelocityBridgeConfig = field(default_factory=VelocityBridgeConfig)

    # Density-specific parameters
    time_scale: float = field(
        default=60.0,
        metadata={"min": 0.0, "max": 200.0, "label": "Density Time Scale",
                  "description": "Density output scaling"}
    )
    saturation: float = field(
        default=2.5,
        metadata={"min": 0.0, "max": 5.0, "label": "Saturation",
                  "description": "Color saturation boost for visualization"}
    )


class DensityBridge(FlowBase):
    """Density bridge with velocity-driven alpha channel.

    Pipeline:
        1. Receive RGB density via set_density() → input_fbo
        2. Receive velocity from optical flow via set_velocity() → VelocityProcessor
        3. VelocityProcessor applies temporal smoothing (trail)
        4. VelocityProcessor applies Gaussian blur (spatial smoothing)
        5. Combine density RGB + smoothed velocity magnitude → RGBA with alpha
        6. Apply HSV saturation adjustment
        7. Output RGBA32F density via .density property
        8. Scaled output via .density_delta property

    Data flow:
        Density RGB → set_density() → input_fbo (RGBA32F)
        Velocity → set_velocity() → VelocityProcessor (trail+blur) → .velocity
        Combine → DensityBridgeShader → output_fbo (RGBA32F)
        HSV adjust → output_fbo (in-place)
        Scale → _output_delta (RGBA32F)
    """

    def __init__(self, velocity_bridge: VelocityBridge | None = None, config: DensityBridgeConfig | None = None) -> None:
        super().__init__()

        self.config: DensityBridgeConfig = config or DensityBridgeConfig()

        if velocity_bridge is None:
            # Create and link to nested config
            self._velocity_bridge = VelocityBridge()
            self._velocity_bridge.config = self.config.velocity
        else:
            # Use provided (shared) velocity bridge
            self._velocity_bridge: VelocityBridge = velocity_bridge

        # Define internal formats
        self._input_internal_format = GL_RGBA32F   # RGB density input
        self._output_internal_format = GL_RGBA32F  # RGBA density output

        # Output delta FBO (scaled density)
        self._output_delta: Fbo = Fbo()

        # Shaders
        self._density_bridge_shader: DensityBridgeShader = DensityBridgeShader()
        self._hsv_shader: HSV = HSV()
        self._multiply_shader: Scale = Scale()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def density(self) -> Texture:
        """RGBA density output (main result)."""
        return self._output

    @property
    def density_delta(self) -> Texture:
        """Density scaled by timestep for simulation."""
        return self._output_delta

    @property
    def density_input(self) -> Texture:
        """Density scaled by timestep for simulation."""
        return self._input

    @property
    def velocity(self) -> Texture:
        """Smoothed velocity from VelocityProcessor."""
        return self._velocity_bridge.velocity

    @property
    def velocity_delta(self) -> Texture:
        """Smoothed velocity from VelocityProcessor."""
        return self._velocity_bridge.velocity_delta

    @property
    def velocity_input(self) -> Texture:
        """Velocity input to VelocityProcessor."""
        return self._velocity_bridge.velocity_input

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate density bridge FBOs.

        Args:
            width: Processing width
            height: Processing height
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        # Allocate base FBOs (RGBA format)
        super().allocate(width, height, output_width, output_height)

        # Allocate VelocityProcessor
        self._velocity_bridge.allocate(width, height, output_width, output_height)

        # Allocate output delta
        self._output_delta.allocate(width, height, self._output_internal_format)

        # Allocate shaders
        self._density_bridge_shader.allocate()
        self._hsv_shader.allocate()
        self._multiply_shader.allocate()

    def deallocate(self) -> None:
        """Release all FBO resources."""
        super().deallocate()
        self._velocity_bridge.deallocate()
        self._output_delta.deallocate()
        self._density_bridge_shader.deallocate()
        self._hsv_shader.deallocate()
        self._multiply_shader.deallocate()

    def set_density(self, density: Texture) -> None:
        """Set RGB density input (replaces previous).

        Args:
            density: RGB or RGBA density texture
        """
        self._set(density)

    def add_density(self, density: Texture, strength: float = 1.0) -> None:
        """Add RGB density input (accumulates with strength).

        Args:
            density: RGB or RGBA density texture
            strength: Blend strength multiplier
        """
        self._add(density, strength)

    def set_velocity(self, velocity: Texture) -> None:
        """Set velocity input (forwards to VelocityProcessor).

        Args:
            velocity: RG velocity texture
        """
        self._velocity_bridge._set(velocity)

    def add_velocity(self, velocity: Texture, strength: float = 1.0) -> None:
        """Add velocity input (forwards to VelocityProcessor).

        Args:
            velocity: RG velocity texture
            strength: Blend strength multiplier
        """
        self._velocity_bridge._add(velocity, strength)

    def reset(self) -> None:
        """Reset all FBOs to zero."""
        super().reset()
        self._velocity_bridge.reset()
        self._output_delta.clear(0.0, 0.0, 0.0, 0.0)

    def update(self, delta_time: float) -> None:
        """Update density bridge processing.

        Args:
            delta_time: Time since last update in seconds
        """
        if not self._allocated:
            return

        # Stage 1: Process velocity through VelocityProcessor (trail + blur)
        # Config already synced via listener
        self._velocity_bridge.update(delta_time)

        # Stage 2: Combine density RGB + velocity magnitude
        self._output_fbo.begin()
        self._density_bridge_shader.use(
            self._input,                       # Density RGB from input_fbo
            self._velocity_bridge.velocity, # Smoothed velocity (unscaled)
            1.0   # Speed multiplier for alpha
        )
        self._output_fbo.end()

        # Stage 3: Apply HSV saturation adjustment
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._hsv_shader.use(
            self._output_fbo.back_texture,
            hue=0.0,
            saturation=self.config.saturation,
            value=1.0
        )
        self._output_fbo.end()

        # Stage 4: Create density_delta (scaled output)
        timestep: float = delta_time * self.config.time_scale
        self._output_delta.begin()
        self._multiply_shader.use(self._output_fbo, timestep)
        self._output_delta.end()
