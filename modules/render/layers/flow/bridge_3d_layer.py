"""The 3D bridge layer: the velocity trail or the kinetic bridge on a 3D velocity (xyz), the density
with its color and the temperature."""

# Standard library imports
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import GL_VIEWPORT, glGetIntegerv

# Local application imports
from modules.gl import GpuTimer, Texture, Style, Text
from modules.flow import (
    VelocityTrailBridge, VelocityTrailBridgeSettings,
    KineticBridge, KineticBridgeSettings, KineticStrip, KINETIC_STRIP_LANES,
    VelocityMagnitude,
    TemperatureBridge, TemperatureBridgeSettings,
    DensityBridge, DensityBridgeSettings,
)
from modules.settings import Field, Group
from modules.utils import HotReloadMethods
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput


class BridgeMethod(IntEnum):
    """How the bridge makes the velocity the fluid reads and the amount the density reads."""
    VELOCITY_TRAIL = 0  # a smoothed velocity, density where there is speed
    KINETIC = auto()  # two trails, reading the change of the motion


class Bridge3DDrawMode(IntEnum):
    """Draw modes for Bridge3DLayer. A 3D velocity (xyz) draws as white x/y arrows with a ring of radius |z|
    on the arrow's scale, red toward the camera and green away; as scalars, in two colors around grey.

    Outputs, what the fluid gets:
        VELOCITY - The active bridge's smoothed 3D velocity
        DENSITY - The colored density (RGBA16F)
        TEMPERATURE - The temperature field, in two colors

    Kinetic bridge (only while it runs):
        KINETIC_FAST / KINETIC_SLOW - The two trails (RGBA16F, xyz)
        KINETIC_ACCELERATION - fast - slow along the motion in two colors: + onset warm, - release cold
        KINETIC_DEPTH_ACCELERATION - The z of fast - slow: red toward the camera, green away
        KINETIC_TURN - fast - slow across the motion in two colors
        KINETIC_HELD_RELEASE - The peak-held release
        KINETIC_RESERVOIR - Density stored per pixel
        KINETIC_EMISSION - The density amount it emits
        KINETIC_ONSET_VELOCITY - The second velocity input (RGBA16F, xyz)
        KINETIC_CURL - The fast trail's 2D curl in two colors
        KINETIC_STRIP - Speed, onset, release and emission over the last seconds
    """
    VELOCITY = 0
    DENSITY = auto()
    TEMPERATURE = auto()
    KINETIC_FAST = auto()
    KINETIC_SLOW = auto()
    KINETIC_ACCELERATION = auto()
    KINETIC_DEPTH_ACCELERATION = auto()
    KINETIC_TURN = auto()
    KINETIC_HELD_RELEASE = auto()
    KINETIC_RESERVOIR = auto()
    KINETIC_EMISSION = auto()
    KINETIC_ONSET_VELOCITY = auto()
    KINETIC_CURL = auto()
    KINETIC_STRIP = auto()


DEPTH_DRAW_MODES: set[Bridge3DDrawMode] = {Bridge3DDrawMode.KINETIC_DEPTH_ACCELERATION}     # red toward the camera, green away
VECTOR_DRAW_MODES: set[Bridge3DDrawMode] = {       # 3D velocities, drawn as velocities whatever the texture's format
    Bridge3DDrawMode.VELOCITY, Bridge3DDrawMode.KINETIC_FAST, Bridge3DDrawMode.KINETIC_SLOW, Bridge3DDrawMode.KINETIC_ONSET_VELOCITY,
}


class Bridge3DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge3DLayer: the method and each bridge's group."""
    draw_mode: Field[Bridge3DDrawMode] =    Field(Bridge3DDrawMode.VELOCITY)
    method: Field[BridgeMethod] =           Field(BridgeMethod.VELOCITY_TRAIL, description="How the bridge makes the fluid's velocity and the density's amount: one trail, or two trails reading the change of the motion")

    trail       = Group(VelocityTrailBridgeSettings)
    kinetic     = Group(KineticBridgeSettings)
    density     = Group(DensityBridgeSettings)
    temperature = Group(TemperatureBridgeSettings)


class Bridge3DLayer(BridgeLayer):
    """The 3D bridge layer. `settings.method` picks the velocity trail or the kinetic bridge for the
    fluid's velocity and the density's amount; the density bridge colors the amount with the density's
    color, the temperature bridge reads the color where there is speed."""

    def __init__(self, settings: Bridge3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge3DLayerSettings = settings
        self._trail_bridge: VelocityTrailBridge = VelocityTrailBridge(settings.trail)
        self._kinetic_bridge: KineticBridge = KineticBridge(settings.kinetic)
        self._velocity_magnitude: VelocityMagnitude = VelocityMagnitude()
        self._density_bridge: DensityBridge = DensityBridge(settings.density)
        self._temperature_bridge: TemperatureBridge = TemperatureBridge(settings.temperature)

        self._kinetic_strip: KineticStrip = KineticStrip()
        self._strip_text: Text = Text()
        self._gpu_timer: GpuTimer = GpuTimer("GPU bridge")

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def _kinetic(self) -> bool:
        return self.settings.method == BridgeMethod.KINETIC

    @property
    def _velocity(self) -> Texture:
        """The active bridge's smoothed velocity, xyz (RGBA16F); z is 0 without depth."""
        return self._kinetic_bridge.smooth_velocity if self._kinetic else self._trail_bridge.smooth_velocity

    @property
    def output(self) -> BridgeOutput:
        kinetic: KineticBridgeSettings = self.settings.kinetic
        return BridgeOutput(
            self._velocity, self._density_bridge.density, self._temperature_bridge.temperature,
            self._kinetic_bridge.onset_velocity if self._kinetic and kinetic.onset.force > 0.0 else None,
            self._kinetic_bridge.curl if self._kinetic and kinetic.onset.spin > 0.0 else None,
        )

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        super().allocate(width, height, internal_format)
        self._density_bridge.allocate(width, height)
        self._kinetic_strip.allocate()
        self._strip_text.allocate(font_size=18)

    def deallocate(self) -> None:
        super().deallocate()
        self._density_bridge.deallocate()
        self._kinetic_strip.deallocate()
        self._strip_text.deallocate()
        self._gpu_timer.deallocate()

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._trail_bridge.allocate(width, height, output_width, output_height)
        self._kinetic_bridge.allocate(width, height)
        self._velocity_magnitude.allocate(width, height)
        self._temperature_bridge.allocate(width, height)

    def _deallocate_stages(self) -> None:
        self._trail_bridge.deallocate()
        self._kinetic_bridge.deallocate()
        self._velocity_magnitude.deallocate()
        self._temperature_bridge.deallocate()

    def _reset_stages(self) -> None:
        self._trail_bridge.reset()
        self._kinetic_bridge.reset()
        self._density_bridge.reset()
        self._kinetic_strip.reset()

    # ========== The bridges ==========

    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        self._gpu_timer.begin()
        mode: Bridge3DDrawMode = self.settings.draw_mode
        kinetic: bool = self._kinetic
        if kinetic:
            bridge: KineticBridge = self._kinetic_bridge
            spin: float = self.settings.kinetic.onset.spin
            bridge.set_velocity(input.velocity)
            bridge.update(input.dt)
            if mode == Bridge3DDrawMode.KINETIC_DEPTH_ACCELERATION:
                bridge.update_depth_acceleration()
            if spin > 0.0 or mode == Bridge3DDrawMode.KINETIC_CURL:
                bridge.update_curl(spin if spin > 0.0 else 1.0)
            if mode == Bridge3DDrawMode.KINETIC_STRIP:
                self._kinetic_strip.update(bridge.fast, bridge.acceleration, bridge.emission, self.settings.visualization.strip_scale)
        else:
            self._trail_bridge.set_velocity(input.velocity)
            self._trail_bridge.update(input.dt)

        self._velocity_magnitude.set_input(self._velocity)
        self._velocity_magnitude.update()

        # The amount the density bridge makes its alpha from: the kinetic bridge's emission, or the trail's speed
        alpha: Texture = self._kinetic_bridge.emission if kinetic else self._velocity_magnitude.magnitude
        self._density_bridge.set_color(color)
        self._density_bridge.set_alpha(alpha)
        self._density_bridge.update()

        self._temperature_bridge.set_color(color)
        self._temperature_bridge.set_mask(self._velocity_magnitude.magnitude)
        self._temperature_bridge.update()
        self._gpu_timer.end()

    def _visualize(self, input: BridgeInput) -> None:
        mode: Bridge3DDrawMode = self.settings.draw_mode
        signed: Texture | None = self._signed_texture()
        if signed is not None:
            self._visualizer.process_signed(signed, mode in DEPTH_DRAW_MODES)
        else:
            self._visualizer.process(self._draw_texture(), mode in VECTOR_DRAW_MODES)

    # ========== Rendering ==========

    def draw(self) -> None:
        super().draw()
        if self.settings.draw_mode == Bridge3DDrawMode.KINETIC_STRIP and self._strip_text.allocated:
            Style.push_style()
            self._draw_strip_labels()
            Style.pop_style()

    def _draw_strip_labels(self) -> None:
        """The name of each lane of the kinetic strip, at its top left."""
        vp = glGetIntegerv(GL_VIEWPORT)
        w, h = int(vp[2]), int(vp[3])
        if w <= 0 or h <= 0:
            return
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for lane, name in enumerate(KINETIC_STRIP_LANES):
            self._strip_text.draw_box_text(8, lane * h // len(KINETIC_STRIP_LANES) + 8, name, (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), w, h)

    # ========== Texture Selection ==========

    def _signed_texture(self) -> Texture | None:
        """The scalar field to draw in two colors, None when draw_mode is another kind of texture."""
        bridge: KineticBridge = self._kinetic_bridge
        textures: dict[Bridge3DDrawMode, Texture] = {
            Bridge3DDrawMode.TEMPERATURE: self._temperature_bridge.temperature,
            Bridge3DDrawMode.KINETIC_ACCELERATION: bridge.acceleration,
            Bridge3DDrawMode.KINETIC_DEPTH_ACCELERATION: bridge.depth_acceleration,
            Bridge3DDrawMode.KINETIC_TURN: bridge.turn,
            Bridge3DDrawMode.KINETIC_HELD_RELEASE: bridge.held_release,
            Bridge3DDrawMode.KINETIC_RESERVOIR: bridge.reservoir,
            Bridge3DDrawMode.KINETIC_EMISSION: bridge.emission,
            Bridge3DDrawMode.KINETIC_CURL: bridge.curl,
        }
        return textures.get(self.settings.draw_mode)

    def _draw_texture(self) -> Texture:
        bridge: KineticBridge = self._kinetic_bridge
        textures: dict[Bridge3DDrawMode, Texture] = {
            Bridge3DDrawMode.VELOCITY: self._velocity,
            Bridge3DDrawMode.DENSITY: self._density_bridge.density,
            Bridge3DDrawMode.KINETIC_FAST: bridge.fast,
            Bridge3DDrawMode.KINETIC_SLOW: bridge.slow,
            Bridge3DDrawMode.KINETIC_ONSET_VELOCITY: bridge.onset_velocity,
            Bridge3DDrawMode.KINETIC_STRIP: self._kinetic_strip.texture,
        }
        return textures.get(self.settings.draw_mode, self._velocity)
