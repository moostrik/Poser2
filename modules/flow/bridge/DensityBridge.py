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
from ..shaders.MultiplyForce import MultiplyForce

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class DensityBridgeConfig(FlowConfigBase):
    """Configuration for density bridge.

    Velocity parameters propagate to internal VelocityProcessor.
    """
    # Velocity processing parameters
    velocity_trail_weight: float = field(
        default=0.3,
        metadata={"min": 0.0, "max": 0.99, "label": "Velocity Trail Weight",
                  "description": "Temporal smoothing for velocity"}
    )
    velocity_blur_radius: float = field(
        default=3.0,
        metadata={"min": 0.0, "max": 10.0, "label": "Velocity Blur Radius",
                  "description": "Spatial smoothing for velocity"}
    )
    velocity_blur_steps: int = field(
        default=1,
        metadata={"min": 0, "max": 8, "label": "Velocity Blur Steps",
                  "description": "Number of blur passes for velocity"}
    )

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

    def __init__(self, config: DensityBridgeConfig | None = None) -> None:
        super().__init__()

        self.config: DensityBridgeConfig = config or DensityBridgeConfig()

        # Define internal formats
        self.input_internal_format = GL_RGBA32F   # RGB density input
        self.output_internal_format = GL_RGBA32F  # RGBA density output

        # Compose VelocityProcessor with initial config
        self.velocity_processor = VelocityBridge(
            VelocityBridgeConfig(
                trail_weight=self.config.velocity_trail_weight,
                blur_radius=self.config.velocity_blur_radius,
                blur_steps=self.config.velocity_blur_steps
            )
        )

        # Listen for config changes to propagate to VelocityProcessor
        self.config.add_listener(self._on_config_changed)

        # Output delta FBO (scaled density)
        self._output_delta: Fbo = Fbo()

        # Shaders
        self._density_bridge_shader: DensityBridgeShader = DensityBridgeShader()
        self._hsv_shader: HSV = HSV()
        self._multiply_shader: MultiplyForce = MultiplyForce()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def _on_config_changed(self) -> None:
        """Propagate relevant config changes to VelocityProcessor."""
        self.velocity_processor.config.trail_weight = self.config.velocity_trail_weight
        self.velocity_processor.config.blur_radius = self.config.velocity_blur_radius
        self.velocity_processor.config.blur_steps = self.config.velocity_blur_steps

    @property
    def density(self) -> Texture:
        """RGBA density output (main result)."""
        return self.output

    @property
    def density_delta(self) -> Texture:
        """Density scaled by timestep for simulation."""
        return self._output_delta

    @property
    def velocity(self) -> Texture:
        """Smoothed velocity from VelocityProcessor."""
        return self.velocity_processor.velocity

    @property
    def velocity_delta(self) -> Texture:
        """Smoothed velocity from VelocityProcessor."""
        return self.velocity_processor.velocity_delta

    @property
    def velocity_input(self) -> Texture:
        """Velocity input to VelocityProcessor."""
        return self.velocity_processor.input

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
        self.velocity_processor.allocate(width, height, output_width, output_height)

        # Allocate output delta
        self._output_delta.allocate(width, height, self.output_internal_format)

        # Allocate shaders
        self._density_bridge_shader.allocate()
        self._hsv_shader.allocate()
        self._multiply_shader.allocate()

    def deallocate(self) -> None:
        """Release all FBO resources."""
        super().deallocate()
        self.velocity_processor.deallocate()
        self._output_delta.deallocate()
        self._density_bridge_shader.deallocate()
        self._hsv_shader.deallocate()
        self._multiply_shader.deallocate()

    def set_density(self, density: Texture) -> None:
        """Set RGB density input (replaces previous).

        Args:
            density: RGB or RGBA density texture
        """
        self.set(density)

    def add_density(self, density: Texture, strength: float = 1.0) -> None:
        """Add RGB density input (accumulates with strength).

        Args:
            density: RGB or RGBA density texture
            strength: Blend strength multiplier
        """
        self.add(density, strength)

    def set_velocity(self, velocity: Texture) -> None:
        """Set velocity input (forwards to VelocityProcessor).

        Args:
            velocity: RG velocity texture
        """
        self.velocity_processor.set(velocity)

    def add_velocity(self, velocity: Texture, strength: float = 1.0) -> None:
        """Add velocity input (forwards to VelocityProcessor).

        Args:
            velocity: RG velocity texture
            strength: Blend strength multiplier
        """
        self.velocity_processor.add(velocity, strength)

    def reset(self) -> None:
        """Reset all FBOs to zero."""
        super().reset()
        self.velocity_processor.reset()
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
        self.velocity_processor.update(delta_time)

        # Stage 2: Combine density RGB + velocity magnitude
        self.output_fbo.begin()
        self._density_bridge_shader.use(
            self.input,                       # Density RGB from input_fbo
            self.velocity_processor.velocity, # Smoothed velocity (unscaled)
            1.0   # Speed multiplier for alpha
        )
        self.output_fbo.end()

        # Stage 3: Apply HSV saturation adjustment
        self.output_fbo.swap()
        self.output_fbo.begin()
        self._hsv_shader.use(
            self.output_fbo.back_texture,
            hue=0.0,
            saturation=self.config.saturation,
            value=1.0
        )
        self.output_fbo.end()

        # Stage 4: Create density_delta (scaled output)
        timestep: float = delta_time * self.config.time_scale
        self._output_delta.begin()
        self._multiply_shader.use(self.output_fbo, timestep)
        self._output_delta.end()
