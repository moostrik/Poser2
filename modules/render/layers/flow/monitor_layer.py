"""The monitor: a layer that shows one texture of one stage, picked on the panel, drawn only while
it is shown.

`MonitorInput` is what the app hands it each tick: every stage's ports. The monitor picks the stage
and the texture by its own settings, and draws their channels, all or one, by the port's kind or by
the view asked: as an image, as a colormap centered on grey, as glyphs, as a strip of the channels'
amounts over time, or as a histogram of their values, the last two under their log axis, the colormap under its key
(`Legend`). It renders in `draw`, once per tick however many panels show it, and nothing in
`update`, at the size it is drawn at (`DrawnSize`): the viewport the app fits it into, not the
size the app allocates the layer at."""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from OpenGL.GL import GL_VIEWPORT, glGetIntegerv, glViewport

from modules.gl import Texture, Texture3D, Style, Blit, MouseButton, fit
from modules.flow import Visualizer, Strip, ALL_CHANNELS, Legend, LegendKind, DrawnSize, Port, TextureKind, VolumeCut, VolumeCube, VolumeGlyphs, Cut, Orbit
from modules.settings import Field, BaseSettings, Group, Widget
from ..layer_base import LayerBase


class FlowStage(IntEnum):
    """A stage of a flow system, as a monitor names it."""
    MOTION = 0
    BRIDGE = auto()
    FLUID = auto()


class MonitorView(IntEnum):
    """How the monitor draws the channels: by the port's kind, or as asked."""
    AUTO = 0
    IMAGE = auto()
    COLORMAP = auto()
    GLYPHS = auto()
    HISTOGRAM = auto()
    STRIP = auto()


class Channel(IntEnum):
    """The texture's channels the monitor draws: one, or all."""
    X = 0
    Y = auto()
    Z = auto()
    W = auto()
    ALL = ALL_CHANNELS


class GlyphSpacing(IntEnum):
    """The distance between glyphs, in pixels, in steps that double."""
    SPACING_4 = 4
    SPACING_8 = 8
    SPACING_16 = 16
    SPACING_32 = 32
    SPACING_64 = 64
    SPACING_128 = 128

    @property
    def label(self) -> str:
        """The panel's text for it: the pixels."""
        return str(self.value)


class MonitorAdvancedSettings(BaseSettings):
    """How the monitor draws: the channels, the scale, the glyphs' spacing, the legend and the blend."""
    channel: Field[Channel] =       Field(Channel.ALL, width=16, description="The channels drawn")
    cut: Field[Cut] =               Field(Cut.CUBE, width=24, description="How a volume is laid out: in 3D, one layer, or seen from the front, the top or the side")
    scale: Field[float] =           Field(1.0, min=0.01, max=100.0, widget=Widget.log_slider, description="Multiplier on the values")
    glyph_spacing: Field[GlyphSpacing] = Field(GlyphSpacing.SPACING_32, width=24, description="Distance between glyphs (pixels)")
    legend: Field[bool] =           Field(True, description="The strip's and histogram's log axis, the colormap's key")
    blend_mode: Field[Style.BlendMode] = Field(Style.BlendMode.ADD, width=32, description="How the monitor is drawn over the panel")


class MonitorLayerSettings(BaseSettings):
    """Configuration for MonitorLayer: what it shows, and in `advanced` how."""
    stage: Field[FlowStage] =       Field(FlowStage.BRIDGE, description="The stage whose texture is shown")
    names: Field[list[str]] =       Field([""], access=Field.READ, visible=False, description="The stage's textures")
    texture: Field[str] =           Field("output.velocity", widget=Widget.text_select, options=names, description="The texture shown")
    view: Field[MonitorView] =      Field(MonitorView.AUTO, description="By its kind, as an image, as a colormap centered on grey, as glyphs (arrows, or dots for a scalar), as a strip over time, or as a histogram")
    camera: Field[str] =            Field("", access=Field.READ, description="A volume's view: the angle and zoom (left and right drag), the layer (scroll); middle click resets")

    advanced = Group(MonitorAdvancedSettings)


@dataclass
class MonitorInput:
    """Every stage's ports, built by the app each tick; the monitor picks by its settings."""
    ports: dict[FlowStage, dict[str, Port]] = field(default_factory=dict)


_AUTO_VIEWS: dict[TextureKind, MonitorView] = {
    TextureKind.IMAGE: MonitorView.IMAGE,
    TextureKind.VELOCITY: MonitorView.GLYPHS,
    TextureKind.SCALAR: MonitorView.COLORMAP,
}


class MonitorLayer(LayerBase):
    """Shows one port of one stage. `set_input` before the render loop; `update` keeps the panel's list
    of names; `draw` renders the port once per tick and draws it."""

    def __init__(self, settings: MonitorLayerSettings) -> None:
        self.settings: MonitorLayerSettings = settings
        self._input: MonitorInput | None = None
        self._visualizer: Visualizer = Visualizer()
        self._strip: Strip = Strip()
        self._legend: Legend = Legend()
        self._legend_kind: LegendKind | None = None     # the legend of the plot drawn last, if it has one
        self._rendered: bool = False            # this tick's port is drawn into the visualizer or the strip
        self._strip_of: tuple[FlowStage, str, Channel] | None = None    # what the strip plots; anything else restarts it
        self._port: Port | None = None          # the port rendered last
        self._view: MonitorView = MonitorView.AUTO   # the view rendered last, AUTO resolved; draw shows what render drew
        self._drawn_size: DrawnSize = DrawnSize()   # the size it renders at, from the viewports it is drawn into
        self._rendered_size: tuple[int, int] = (0, 0)   # the size this tick's port was rendered at
        self._strip_stepped: bool = False           # the strip got this tick's column
        self._volume_cut: VolumeCut = VolumeCut()   # a volume laid out flat for the 2D views
        self._volume_cube: VolumeCube = VolumeCube()    # a volume in 3D, for the image and colormap views
        self._volume_glyphs: VolumeGlyphs = VolumeGlyphs()  # a volume in 3D, for the glyphs
        self._shown_3d: Texture | None = None       # what it rendered last, when it is one of the 3D views
        self._slice: float = 0.5                    # the slice's place through a volume, 0 front to 1 back
        self._layers: int = 0                       # the layers a scroll steps through, the show's density's
        self._orbit: Orbit = Orbit()                # the 3D views' camera; the mouse turns it, a middle click resets it
        self._aspect: float | None = None           # the shape of what it rendered, kept when drawn; None fills the panel
        self._allocated: bool = False

    def set_input(self, input: MonitorInput) -> None:
        """Every stage's ports, for the next draw."""
        self._input = input

    # ========== The mouse ==========

    def drag(self, dx: float, dy: float, button: MouseButton) -> None:
        """A drag of `dx`, `dy` (shares of the window, y down): the left button turns the 3D view,
        the right zooms it. Render thread, from the window's callbacks."""
        if button == MouseButton.LEFT:
            self._orbit.turn(dx, dy)
        elif button == MouseButton.RIGHT:
            self._orbit.dolly(dy)
        self._show_camera()

    def scroll(self, steps: float, layers: int) -> None:
        """A turn of the wheel: the slice one of `layers` further back per step away from the user,
        `layers` the show's density's, the same for every monitor, so they stay on one place."""
        if layers <= 0:
            return
        self._layers = layers
        layer: int = min(max(int(self._slice * layers) + round(steps), 0), layers - 1)
        self._slice = (layer + 0.5) / layers
        self._show_camera()

    def reset_view(self) -> None:
        """The front view, framed, at the middle layer."""
        self._orbit.reset()
        self._slice = 0.5
        self._show_camera()

    def _show_camera(self) -> None:
        orbit: Orbit = self._orbit
        layer: str = f", layer {int(self._slice * self._layers) + 1} of {self._layers}" if self._layers else ""
        self.settings.camera = f"yaw {orbit.yaw:.0f}°, pitch {orbit.pitch:.0f}°, zoom {orbit.zoom:.2f}{layer}"

    @property
    def texture(self) -> Texture:
        """The port as drawn last: the strip, a volume's cube, or the visualizer's output."""
        if self._view == MonitorView.STRIP:
            return self._strip.texture
        if self._shown_3d is not None:
            return self._shown_3d
        return self._visualizer.texture

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """The shaders; the size the app gives is not the monitor's: it renders at the size it is drawn at."""
        self._visualizer.allocate()
        self._strip.allocate()
        self._legend.allocate()
        self._volume_cut.allocate()
        self._volume_cube.allocate()
        self._volume_glyphs.allocate()
        self._allocated = True

    def deallocate(self) -> None:
        self._volume_cut.deallocate()
        self._volume_cube.deallocate()
        self._volume_glyphs.deallocate()
        self._visualizer.deallocate()
        self._strip.deallocate()
        self._legend.deallocate()
        self._allocated = False

    def _resize(self, width: int, height: int) -> None:
        self._visualizer.resize(width, height)
        self._strip.resize(height)
        self._legend.resize(width, height)
        self._volume_cube.resize(width, height)
        self._volume_glyphs.resize(width, height)

    # ========== Processing ==========

    def _ports(self) -> dict[str, Port]:
        return self._input.ports.get(self.settings.stage, {}) if self._input is not None else {}

    def _pick(self) -> Port | None:
        """The port the settings name, else the stage's first output, else its first port."""
        ports: dict[str, Port] = self._ports()
        port: Port | None = ports.get(self.settings.texture)
        if port is None and ports:
            outputs: list[Port] = [p for name, p in ports.items() if name.startswith('output.')]
            port = outputs[0] if outputs else next(iter(ports.values()))
        return port if port is not None and port.texture.allocated else None

    def update(self) -> None:
        """Keep the panel's list of the stage's textures; the port is rendered at the first draw."""
        self._rendered = False
        self._strip_stepped = False
        self._drawn_size.tick()
        names: list[str] = list(self._ports()) or [""]
        if names != self.settings.names:
            self.settings.names = names

    def _render(self) -> None:
        port: Port | None = self._pick()
        self._port = port
        self._legend_kind = None
        view: MonitorView = self.settings.view
        if view != MonitorView.STRIP:
            self._strip_of = None               # the strip starts over when it is shown again
        if port is None:
            return
        settings: MonitorLayerSettings = self.settings
        channel: Channel = settings.advanced.channel
        scale: float = settings.advanced.scale
        if view == MonitorView.AUTO:
            view = _AUTO_VIEWS[port.kind]
        self._view = view
        self._aspect = None
        self._shown_3d = None
        if port.volume and settings.advanced.cut == Cut.CUBE and view in (MonitorView.IMAGE, MonitorView.COLORMAP, MonitorView.GLYPHS):
            self._render_3d(port, view, scale, int(channel))
            if view != MonitorView.IMAGE:
                self._legend_kind = LegendKind.COLORMAP
                if settings.advanced.legend:
                    self._legend.update(self._legend_kind, int(channel), scale, self._visualizer.histogram_total)
            return
        texture: Texture = self._flat(port, view)
        if view == MonitorView.STRIP:
            shown: tuple[FlowStage, str, Channel] = (self.settings.stage, self.settings.texture, channel)
            if shown != self._strip_of:
                self._strip.reset()
                self._strip_of = shown
            if not self._strip_stepped:         # one column per tick, also when a larger draw renders again
                self._strip.update(texture, scale, int(channel))
                self._strip_stepped = True
            self._legend_kind = LegendKind.STRIP
        elif view == MonitorView.IMAGE:
            self._visualizer.process_image(texture, scale, int(channel))
        elif view == MonitorView.COLORMAP:
            self._visualizer.process_colormap(texture, scale, int(channel))
            self._legend_kind = LegendKind.COLORMAP
        elif view == MonitorView.HISTOGRAM:
            self._visualizer.process_histogram(port.texture if port.volume else texture, scale, int(channel))
            self._legend_kind = LegendKind.HISTOGRAM
        else:
            self._visualizer.process_glyphs(texture, scale, float(settings.advanced.glyph_spacing.value),
                                            port.kind != TextureKind.VELOCITY, int(channel))

    def _render_3d(self, port: Port, view: MonitorView, scale: float, channel: int) -> None:
        """A volume in its box from the orbit camera: for the image and the colormap its layers stacked,
        the slice's layer pulled out; for the glyphs arrows of a velocity or dots of a scalar."""
        if port.prepare is not None:
            port.prepare()
        volume = port.texture
        assert isinstance(volume, Texture3D)
        if view == MonitorView.GLYPHS:
            spacing: float = float(self.settings.advanced.glyph_spacing.value)
            self._shown_3d = self._volume_glyphs.process(volume, self._orbit, port.extent, spacing, scale,
                                                         port.kind != TextureKind.VELOCITY, channel)
        else:
            self._shown_3d = self._volume_cube.process(volume, self._orbit, self._slice, port.extent, scale, channel,
                                                       view == MonitorView.COLORMAP)

    def _flat(self, port: Port, view: MonitorView) -> Texture:
        """The 2D texture a view draws: the port's own, or a volume's flat cut. A volume is brought up
        to date first (`Port.prepare`); the strip reads its front mean, the volume's mean, and the
        histogram counts the volume itself. A cut keeps the box's proportions when drawn."""
        if not port.volume:
            assert isinstance(port.texture, Texture)
            return port.texture
        if port.prepare is not None:
            port.prepare()
        volume = port.texture
        assert isinstance(volume, Texture3D)
        if view == MonitorView.HISTOGRAM:
            return self._volume_cut.texture         # not drawn: the histogram counts the volume
        cut: Cut = Cut.FRONT if view == MonitorView.STRIP else self.settings.advanced.cut
        if not cut.flat:
            cut = Cut.SLICE                         # a view with no cube of its own draws the slice
        flat: Texture = self._volume_cut.process(volume, cut, self._slice, port.kind, port.extent)
        if view != MonitorView.STRIP:
            self._aspect = flat.width / max(flat.height, 1)
        return flat
        if self._legend_kind is not None and settings.advanced.legend:
            self._legend.update(self._legend_kind, int(channel), scale, self._visualizer.histogram_total)

    # ========== Rendering ==========

    def draw(self) -> None:
        if not self._allocated:
            return
        viewport = glGetIntegerv(GL_VIEWPORT)
        size: tuple[int, int] = self._drawn_size.see(int(viewport[2]), int(viewport[3]))
        if not self._rendered or size != self._rendered_size:
            # Rendering into the port's FBOs sets the viewport to their size; the panel's comes back after
            self._resize(*size)
            self._rendered_size = size
            Style.push_style()
            Style.set_blend_mode(Style.BlendMode.DISABLED)
            self._render()
            Style.pop_style()
            glViewport(int(viewport[0]), int(viewport[1]), int(viewport[2]), int(viewport[3]))
            self._rendered = True
        if self._port is None:
            return
        Style.push_style()
        Style.set_blend_mode(self.settings.advanced.blend_mode)
        if self.texture.allocated:
            if self._aspect is not None:        # a volume's cut, in the box's proportions inside the panel
                x, y, w, h = fit(self._aspect, 1.0, int(viewport[2]), int(viewport[3]))
                glViewport(int(viewport[0] + x), int(viewport[1] + y), int(w), int(h))
                Blit.use(self.texture)
                glViewport(int(viewport[0]), int(viewport[1]), int(viewport[2]), int(viewport[3]))
            else:
                Blit.use(self.texture)
        if self._legend_kind is not None and self.settings.advanced.legend and self._legend.texture.allocated:
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            Blit.use(self._legend.texture)
        Style.pop_style()
