"""The 3D fluid layer: `Fluid3D`, the volumetric Navier–Stokes simulation, on its injections.

The density volume stores the colored density (RGBA16F GL_TEXTURE_2D_ARRAY, output resolution);
`Composite3DDensity` flattens it to the RGBA16F 2D output, the layer's texture, with the parallax
eye the app steers by `set_eye`.
"""

# Third-party imports
from OpenGL.GL import GL_VIEWPORT, glGetIntegerv

# Local application imports
from modules.gl import Texture, Style, Text, Blit
from modules.flow import Fluid3D, Fluid3DSettings
from modules.utils import HotReloadMethods
from .fluid_layer import FluidLayer, FluidLayerSettings, FluidOutput, Injection
from .port import Port, TextureKind


class Fluid3DLayerSettings(Fluid3DSettings, FluidLayerSettings):
    """Configuration for Fluid3DLayer: the stage's settings and its fluid's own, one panel, the stage's
    first; the stage's width and height are the fluid's simulation grid."""


class Fluid3DLayer(FluidLayer):
    """The 3D fluid layer: each injection's velocity, its z as w, its onset and spin, density and
    temperature into the volume at its weight; one step per tick; the composited density as the texture."""

    def __init__(self, settings: Fluid3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Fluid3DLayerSettings = settings
        self._fluid: Fluid3D = Fluid3D(settings)
        self._legend_text: Text = Text()     # legend for the debug layers view (hue = layer)

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        """Composited 2D density output (RGBA16F, color baked in)."""
        return self._fluid.density

    @property
    def output(self) -> FluidOutput:
        return FluidOutput(self._fluid.density, textures={'density': Port(self._fluid.density, TextureKind.IMAGE)})

    def set_eye(self, x: float, y: float) -> None:
        """The parallax eye's offset (±1 across the frame, vertical at half range), read at the next composite."""
        self._fluid.set_eye(x, y)

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._fluid.allocate(width, height)
        self._legend_text.allocate(font_size=18)

    def deallocate(self) -> None:
        self._fluid.deallocate()
        self._legend_text.deallocate()
        super().deallocate()

    def reset(self) -> None:
        self._fluid.reset()

    # ========== Processing ==========

    def _inject(self, injection: Injection) -> None:
        fluid: Fluid3D = self._fluid
        # The velocity's z (blue) is the w; the onset likewise
        fluid.add_velocity(injection.velocity, injection.weight, injection.velocity, 2)
        if injection.onset is not None:
            fluid.add_velocity(injection.onset, injection.weight, injection.onset, 2)
        if injection.spin is not None:
            fluid.set_spin(injection.spin, injection.weight)
        fluid.add_density(injection.density, injection.weight)
        fluid.add_temperature(injection.temperature, injection.weight)

    def _step(self, dt: float) -> None:
        self._fluid.update(dt)

    # ========== Rendering ==========

    def draw(self) -> None:
        Style.push_style()
        Style.set_blend_mode(self.settings.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        if self.settings.volume.debug_layers and self._legend_text.allocated:
            self._draw_layer_legend()
        Style.pop_style()

    def _draw_layer_legend(self) -> None:
        """Hue legend for the debug layers view: violet = front ... red = back."""
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
