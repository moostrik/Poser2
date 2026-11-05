"""Fluid Layer - Navier-Stokes fluid simulation with cross-camera flow inputs."""

# Standard library imports
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import GL_RGBA16F
import numpy as np

# Local application imports
from modules.gl import Texture, Style, Fbo, Blit
from ..layer_base import LayerBase
from modules.board import HasFrames
from modules.pose import Frame, Similarity, MotionGate, AngleMotion

from modules.settings import Field, BaseSettings, Group
from modules.flow import Visualizer, VisualizationFieldSettings, Fluid2D, Fluid2DSettings
from ...shaders import DensityColorize

from modules.utils import HotReloadMethods

from .motion_2d_layer import Motion2DLayer
from ...color_settings import ColorSettings


class Fluid2DDrawMode(IntEnum):
    """Draw modes for Fluid2DLayer.

    Outputs from fluid simulation:
        VELOCITY - Advected velocity field (RG16F)
        DENSITY - Colorized density field (RGBA16F, colored by track)
        DENSITY_RAW - Raw density field before colorization (RGBA16F)
        PRESSURE - Pressure field (R32F)
        TEMPERATURE - Temperature field (R32F)
        DIVERGENCE - Velocity divergence (R32F)
        VORTICITY - Vorticity curl (R32F)
        BUOYANCY - Buoyancy force (RG32F)
        OBSTACLE - Obstacle mask (R8)
    """
    VELOCITY = 0
    DENSITY = auto()
    DENSITY_RAW = auto()
    PRESSURE = auto()
    TEMPERATURE = auto()
    DIVERGENCE = auto()
    VORTICITY = auto()
    BUOYANCY = auto()
    OBSTACLE = auto()


class Fluid2DLayerSettings(BaseSettings):
    """Configuration for Fluid2DLayer (fluid simulation)."""
    stage: Field[int] =                     Field(0, description="Pipeline stage for pose data")
    max_players: Field[int] =               Field(3, min=1, max=8, access=Field.INIT)
    similarity_scale: Field[float] =        Field(1.0, min=0.0, max=1.0, description="0 = own camera only, 1 = full cross-camera response")
    draw_mode: Field[Fluid2DDrawMode] =       Field(Fluid2DDrawMode.DENSITY)
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)

    fluid: Group[Fluid2DSettings]            = Group(Fluid2DSettings)
    visualization: Group[VisualizationFieldSettings]= Group(VisualizationFieldSettings)


class Fluid2DLayer(LayerBase):
    """Fluid simulation layer with cross-camera flow inputs.

    Receives velocity, density, and temperature from all Motion2DLayers and runs
    Navier-Stokes fluid simulation:
    1. Aggregate inputs from connected Motion2DLayers
    2. Add velocity, density, temperature to fluid
    3. Run advection, pressure projection, viscosity, buoyancy
    4. Output simulated fields

    Cross-camera support:
        Each Fluid2DLayer receives a dict of all Motion2DLayers, enabling
        cross-camera influence (e.g., motion from camera 0 affecting
        fluid in camera 1's output).

    Usage:
        # In RenderManager:
        flow_layers = {i: Motion2DLayer(...) for i in range(num_cams)}
        fluid = Fluid2DLayer(cam_id, board, flow_layers, colors, settings)

        # Each frame:
        fluid.update()
        density_texture = fluid.density
    """

    def __init__(self, cam_id: int, board: HasFrames, flow_layers: dict[int, Motion2DLayer], settings: Fluid2DLayerSettings, color_settings: ColorSettings) -> None:
        """Initialize fluid layer.

        Args:
            cam_id: Camera ID for this layer's output
            board: Board for pose data
            flow_layers: Dict of all Motion2DLayers (cam_id -> Motion2DLayer)
            color_settings: Shared color settings for density colorization
            settings: Layer configuration
        """
        self._cam_id: int = cam_id
        self._board: HasFrames = board
        self._flow_layers: dict[int, Motion2DLayer] = flow_layers
        self.settings: Fluid2DLayerSettings = settings
        self._color_settings: ColorSettings = color_settings

        self._fluid: Fluid2D = Fluid2D(self.settings.fluid)
        self._visualizer: Visualizer = Visualizer(self.settings.visualization)

        # Colorization resources (rendering concern, not simulation)
        self._colorized_fbo: Fbo = Fbo()
        self._density_colorize_shader: DensityColorize = DensityColorize()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Visualization output texture."""
        return self._visualizer.texture

    @property
    def velocity(self) -> Texture:
        """Advected velocity field (RG16F)."""
        return self._fluid.velocity

    @property
    def density(self) -> Texture:
        """Advected density field (RGBA16F)."""
        return self._fluid.density

    @property
    def pressure(self) -> Texture:
        """Pressure field (R32F)."""
        return self._fluid.pressure

    @property
    def temperature(self) -> Texture:
        """Temperature field (R32F)."""
        return self._fluid.temperature

    @property
    def divergence(self) -> Texture:
        """Velocity divergence (R32F)."""
        return self._fluid.divergence

    @property
    def vorticity(self) -> Texture:
        """Vorticity curl (R32F)."""
        return self._fluid.vorticity_curl

    @property
    def buoyancy(self) -> Texture:
        """Buoyancy force (RG32F)."""
        return self._fluid.buoyancy

    @property
    def obstacle(self) -> Texture:
        """Obstacle mask (R8)."""
        return self._fluid.obstacle

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate fluid simulation buffers.

        Args:
            width: Processing width
            height: Processing height
            internal_format: Ignored (formats determined by Fluid2D)
        """
        self._fluid.allocate(width, height)
        self._visualizer.allocate(self._fluid.sim_width, self._fluid.sim_height)

        # Colorization FBO at density (full output) resolution
        self._colorized_fbo.allocate(width, height, GL_RGBA16F)
        self._density_colorize_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._fluid.deallocate()
        self._visualizer.deallocate()
        self._colorized_fbo.deallocate()
        self._density_colorize_shader.deallocate()

    def reset(self) -> None:
        """Reset fluid simulation state."""
        self._fluid.reset()

    # ========== Processing ==========

    def update(self) -> None:
        """Update fluid simulation with inputs from all flow layers."""
        # Get motion data from pose
        pose: Frame | None = self._board.get_frame(self.settings.stage, self._cam_id)
        similarities: np.ndarray = np.nan_to_num(pose[Similarity].values) if pose is not None and Similarity in pose else np.full((self.settings.max_players,), 0.0)
        motion_gates: np.ndarray = pose[MotionGate].values if pose is not None and MotionGate in pose else np.full((self.settings.max_players,), 0.0)
        motion: float = pose[AngleMotion].value if pose is not None and AngleMotion in pose else 0.0
        # Cross-camera weight for both velocity and density below. The own-camera
        # branch uses `motion`, so a visitor's own fluid is unaffected by the scale.
        m_s = similarities * self.settings.similarity_scale

        self.settings.fluid.density.fade_time = 18.0 - (pow(motion, 2.0) * 14.0)  # More motion = faster decay

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

            # Add velocity from each flow layer
            self._fluid.add_velocity(flow_layer.velocity, vel_strength)

            # Add density to per-camera channel (R=cam0, G=cam1, B=cam2, A=cam3)
            channel: int = cam_id % 4
            self._fluid.add_density_channel(flow_layer.magnitude, channel, den_strength)

            # Add temperature from each flow layer
            self._fluid.add_temperature(flow_layer.temperature, 1.0)

        # Snapshot sim dims before update for resize detection
        prev_sw, prev_sh = self._fluid.sim_width, self._fluid.sim_height

        # Update fluid simulation (handles its own reallocation internally)
        self._fluid.update()

        # Resize visualizer if sim dims changed
        if self._fluid.sim_width != prev_sw or self._fluid.sim_height != prev_sh:
            self._visualizer.allocate(self._fluid.sim_width, self._fluid.sim_height)

        # Colorize density channels with track colors
        self._colorize_density()

        # Update visualization
        self._visualizer.process(self._get_draw_texture())

        Style.pop_style()

    # ========== Rendering ==========

    def draw(self) -> None:
        """Draw with configured blend mode."""
        Style.push_style()
        Style.set_blend_mode(self.settings.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        Style.pop_style()

    # ========== Texture Selection ==========

    def _get_draw_texture(self) -> Texture:
        """Get texture to draw based on draw_mode."""
        textures = {
            Fluid2DDrawMode.VELOCITY: self._fluid.velocity,
            Fluid2DDrawMode.DENSITY: self._colorized_fbo.texture,
            Fluid2DDrawMode.DENSITY_RAW: self._fluid.density,
            Fluid2DDrawMode.PRESSURE: self._fluid.pressure,
            Fluid2DDrawMode.TEMPERATURE: self._fluid.temperature,
            Fluid2DDrawMode.DIVERGENCE: self._fluid.divergence,
            Fluid2DDrawMode.VORTICITY: self._fluid.vorticity_curl,
            Fluid2DDrawMode.BUOYANCY: self._fluid.buoyancy,
            Fluid2DDrawMode.OBSTACLE: self._fluid.obstacle,
        }
        return textures.get(self.settings.draw_mode, self._fluid.density)

    def _colorize_density(self) -> None:
        """Colorize density channels with track colors.

        Maps each RGBA density channel to a corresponding track color and
        composites them additively.
        """
        # Build padded color list (up to 4 channels) from settings each frame
        colors: list[tuple[float, float, float, float]] = list(self._color_settings.track_color_tuples[:4])
        while len(colors) < 4:
            colors.append((0.0, 0.0, 0.0, 0.0))

        self._density_colorize_shader.reload()
        self._colorized_fbo.begin()
        self._density_colorize_shader.use(self._fluid.density, colors)
        self._colorized_fbo.end()
