"""The flat monitor: a layer that shows one texture of one stage, picked on the panel, drawn only while it is
shown. `Monitor3DLayer` extends it with volumes.

`MonitorInput` is what the app hands it each tick: every stage's ports. The monitor shows its own stage's, the
texture picked by its settings, and draws their channels, all or one, by the view asked: as a colormap by the
port's kind, as glyphs, raw as an image, as a histogram of their values, or as a strip of the channels' amounts
over time, the last two under their log axis, the colormap under its key (`Legend`). It renders in `draw`, once
per tick however many panels show it, and nothing in `update`, at the size it is drawn at (`DrawnSize`): the
viewport the app fits it into, not the size the app allocates the layer at."""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from OpenGL.GL import GL_RGBA16F, GL_VIEWPORT, glGetIntegerv, glViewport

from modules.gl import Colormap, Fbo, Texture, Style
from modules.flow import Visualizer, Strip, ALL_CHANNELS, Legend, LegendKind, DrawnSize, Port, TextureKind, REFERENCE_STEP, ZPositionView
from modules.settings import Field, BaseSettings, Group, Widget
from ...layer_base import LayerBase
from .shaders import DepthColormap, DepthDrawMode

MAIN_TINT: str = "#4d5a4f"      # the main monitor's panel: a grey with a little green
DEBUG_TINT: str = "#5e4b4b"     # the debug monitor's panel: a grey with a little red


class FlowStage(IntEnum):
    """A stage of a flow system, as a monitor names it."""
    VIEW = 0
    COLOR = auto()
    MOTION = auto()
    BRIDGE = auto()
    FLUID = auto()


class MonitorView(IntEnum):
    """How the monitor draws the channels, the most used first."""
    COLORMAP = 0            # by the port's kind: a vector by its direction, an image as stored, a scalar centered on grey, a depth in its colormap (channel Y: its confidence as grey)
    GLYPHS = auto()
    IMAGE = auto()          # the raw values as stored, for debugging
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


class Monitor2DAdvancedSettings(BaseSettings):
    """How the monitor draws: the channels, the scale, the glyphs' spacing, the legend and the blend."""
    channel: Field[Channel] =       Field(Channel.ALL, description="The channels drawn")
    scale: Field[float] =           Field(1.0, min=0.01, max=100.0, widget=Widget.log_slider, description="Multiplier on the values")
    blend_mode: Field[Style.BlendMode] = Field(Style.BlendMode.ADD, description="How the monitor is drawn over the panel")
    glyph_spacing: Field[GlyphSpacing] = Field(GlyphSpacing.SPACING_32, description="Distance between glyphs (pixels)")
    colormap: Field[Colormap] =     Field(Colormap.JET, description="A depth's colors, from its near to its far")
    legend: Field[bool] =           Field(True, description="The overlay: titles, log axes and the color key")


class Monitor2DLayerSettings(BaseSettings):
    """Configuration for Monitor2DLayer: what it shows of its stage, and in `advanced` how."""
    names: Field[list[str]] =       Field([""], access=Field.READ, visible=False, description="The stage's textures, in the order the data runs")
    texture: Field[str] =           Field("output.velocity", widget=Widget.path_select, options=names, description="The texture shown, picked a step at a time")
    view: Field[MonitorView] =      Field(MonitorView.COLORMAP, description="How the port is drawn: by its kind, glyphs, raw, histogram or strip")

    advanced = Group(Monitor2DAdvancedSettings)


@dataclass
class MonitorInput:
    """Every stage's ports, built by the app each tick; the monitor picks by its settings."""
    ports: dict[FlowStage, dict[str, Port]] = field(default_factory=dict)


class Monitor2DLayer(LayerBase):
    """Shows one flat port of its stage; a volume it leaves to `Monitor3DLayer`. `set_input` before the render
    loop; `update` keeps the panel's list of names; `draw` renders the port once per tick and draws it."""

    def __init__(self, stage: FlowStage, settings: Monitor2DLayerSettings) -> None:
        self.stage: FlowStage = stage
        self.settings: Monitor2DLayerSettings = settings
        self._depth_picture: Fbo = Fbo()                # a depth in its colormap
        self._depth_colormap: DepthColormap = DepthColormap()
        self._input: MonitorInput | None = None
        self._visualizer: Visualizer = Visualizer()
        self._strip: Strip = Strip()
        self._legend: Legend = Legend()
        self._legend_kind: LegendKind | None = None     # the legend of the plot drawn last, if it has one
        self._rendered: bool = False            # this tick's port is drawn into the visualizer or the strip
        self._strip_of: tuple[FlowStage, str, Channel] | None = None    # what the strip plots; anything else restarts it
        self._port: Port | None = None          # the port rendered last
        self._view: MonitorView = MonitorView.COLORMAP   # the view rendered last; draw shows what render drew
        self._drawn_size: DrawnSize = DrawnSize()   # the size it renders at, from the viewports it is drawn into
        self._rendered_size: tuple[int, int] = (0, 0)   # the size this tick's port was rendered at
        self._strip_stepped: bool = False           # the strip got this tick's column
        self._z_position_view: ZPositionView = ZPositionView()  # a z position drawn as itself and its weight
        self._shown: Texture | None = None          # what it rendered last when the visualizer did not: a depth, a volume
        self._allocated: bool = False

    def set_input(self, input: MonitorInput) -> None:
        """Every stage's ports, for the next draw."""
        self._input = input

    @property
    def texture(self) -> Texture:
        """The port as drawn last: the strip, a depth's colormap, or the visualizer's output."""
        if self._view == MonitorView.STRIP:
            return self._strip.texture
        if self._shown is not None:
            return self._shown
        return self._visualizer.texture

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """The shaders; the size the app gives is not the monitor's: it renders at the size it is drawn at."""
        self._visualizer.allocate()
        self._strip.allocate()
        self._legend.allocate()
        self._z_position_view.allocate()
        self._depth_colormap.allocate()
        self._allocated = True

    def deallocate(self) -> None:
        self._z_position_view.deallocate()
        self._depth_colormap.deallocate()
        self._depth_picture.deallocate()
        self._visualizer.deallocate()
        self._strip.deallocate()
        self._legend.deallocate()
        self._allocated = False

    def _resize(self, width: int, height: int) -> None:
        """Everything but the visualizer, which `_render` sizes to what it draws."""
        self._strip.resize(height)
        self._legend.resize(width, height)

    # ========== Processing ==========

    def _ports(self) -> dict[str, Port]:
        return self._input.ports.get(self.stage, {}) if self._input is not None else {}

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
        if port.prepare is not None:
            port.prepare()                      # its owner brings it up to date, or learns it is shown
        advanced: Monitor2DAdvancedSettings = self.settings.advanced
        channel: Channel = advanced.channel
        channels: int = port.channels
        if channel != Channel.ALL and channel >= channels:
            self._port = None                   # a channel the port does not hold: nothing to draw
            return
        scale: float = advanced.scale
        if port.kind == TextureKind.ACCELERATION:
            scale *= REFERENCE_STEP             # the velocity it adds in a step, in a velocity's units
        self._view = view
        self._shown = None
        by_kind: bool = view == MonitorView.COLORMAP and port.kind == TextureKind.IMAGE    # an image's colormap is itself
        if port.volume and self._render_volume(port, view, scale, channel, channels, by_kind):
            return
        texture: Texture = self._flat(port, view)
        if view == MonitorView.COLORMAP and port.kind == TextureKind.DEPTH and port.depth_range is not None:
            mode: DepthDrawMode = DepthDrawMode.CONFIDENCE if channel == Channel.Y else DepthDrawMode.COLOR
            self._render_depth(texture, port.depth_range, mode, title=self._title(port))
            return
        self._visualizer.resize(*self._rendered_size)
        if view == MonitorView.STRIP:
            shown: tuple[FlowStage, str, Channel] = (self.stage, self.settings.texture, channel)
            if shown != self._strip_of:
                self._strip.reset()
                self._strip_of = shown
            if not self._strip_stepped:         # one column per tick, also when a larger draw renders again
                self._strip.update(texture, scale, int(channel), channels)
                self._strip_stepped = True
            self._legend_kind = LegendKind.STRIP
        elif view == MonitorView.IMAGE or by_kind:
            self._visualizer.process_image(texture, scale, int(channel))
        elif view == MonitorView.COLORMAP:
            self._visualizer.process_colormap(texture, scale, int(channel), port.kind.vector)
            self._legend_kind = LegendKind.COLORMAP
        elif view == MonitorView.HISTOGRAM:
            self._visualizer.process_histogram(port.texture if port.volume else texture, scale, int(channel), channels)
            self._legend_kind = LegendKind.HISTOGRAM
        else:
            self._visualizer.process_glyphs(texture, scale, float(advanced.glyph_spacing.value),
                                            not port.kind.vector, int(channel), port.kind == TextureKind.ACCELERATION)
        if advanced.legend:
            self._legend.update(self._legend_kind, int(channel), channels, scale, self._visualizer.histogram_total, title=self._title(port))

    def _render_volume(self, port: Port, view: MonitorView, scale: float, channel: Channel, channels: int,
                       by_kind: bool) -> bool:
        """A volume: a flat monitor draws none. True when the volume is done with, as here."""
        self._port = None
        return True

    def _title(self, port: Port) -> str:
        """What is shown: the stage and the port's name, the one picked when the settings name none; an
        acceleration drawn per step says so."""
        name: str = next((key for key, value in self._ports().items() if value is port), self.settings.texture)
        per_step: str = " per step" if port.kind == TextureKind.ACCELERATION else ""
        return f"{self.stage.name.lower()} / {name}{per_step}"

    def _render_depth(self, depth: Texture, depth_range: tuple[float, float], mode: DepthDrawMode, title: str) -> None:
        """A depth in `advanced.colormap`, on a log scale from its near to its far (its red); or with `mode`
        CONFIDENCE its confidence (its green) as grey from 0 to 1."""
        width, height = self._rendered_size
        if (self._depth_picture.width, self._depth_picture.height) != (width, height):
            self._depth_picture.deallocate()
            self._depth_picture.allocate(width, height, GL_RGBA16F)
        near, far = depth_range
        self._depth_picture.begin()
        self._depth_colormap.use(depth, depth, near, far, mode, self.settings.advanced.colormap)
        self._depth_picture.end()
        self._shown = self._depth_picture
        if self.settings.advanced.legend:
            self._legend.update(None, 0, 1, 1.0, 0, title=title)

    def _flat(self, port: Port, view: MonitorView) -> Texture:
        """The 2D texture a view draws: the port's own, a z position as itself over its weight."""
        assert isinstance(port.texture, Texture)
        if port.kind == TextureKind.Z_POSITION:
            return self._z_position_view.process(port.texture)     # the z position over its weight, and the weight
        return port.texture

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
        Style.set_blend_mode(self.settings.advanced.blend_mode.premultiplied)     # every view draws its color times its coverage
        if self.texture.allocated:
            self.texture.draw()
        if self.settings.advanced.legend and self._legend.texture.allocated:
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            self._legend.texture.draw()
        Style.pop_style()
