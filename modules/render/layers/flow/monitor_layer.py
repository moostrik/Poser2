"""The monitor: a layer that shows one texture of one stage, picked on the panel, drawn only while
it is shown.

`MonitorInput` is what the app hands it each tick: every stage's ports. The monitor picks the stage
and the texture by its own settings, and draws their channels, all or one, by the port's kind or by
the view asked: as an image, as a colormap centered on grey, as arrows, as a strip of the channels'
amounts over time, or as a histogram of their values, the last two under their log axis, the colormap under its key
(`Legend`). It renders in `draw`, once per tick however many panels show it, and nothing in
`update`."""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from OpenGL.GL import GL_VIEWPORT, glGetIntegerv, glViewport

from modules.gl import Texture, Style, Blit
from modules.flow import Visualizer, HISTOGRAM_TOTAL, Strip, ALL_CHANNELS, Legend, LegendKind
from modules.settings import Field, BaseSettings, Group, Widget
from ..layer_base import LayerBase
from .port import Port, TextureKind


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
    ARROWS = auto()
    HISTOGRAM = auto()
    STRIP = auto()


class Channel(IntEnum):
    """The texture's channels the monitor draws: one, or all."""
    X = 0
    Y = auto()
    Z = auto()
    W = auto()
    ALL = ALL_CHANNELS


class ArrowSpacing(IntEnum):
    """The distance between arrows, in pixels, in steps that double."""
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
    """How the monitor draws: the channels, the scale, the arrows' spacing, the legend and the blend."""
    channel: Field[Channel] =       Field(Channel.ALL, width=16, description="The channels drawn")
    scale: Field[float] =           Field(1.0, min=0.01, max=100.0, widget=Widget.log_slider, description="Multiplier on the values")
    arrow_spacing: Field[ArrowSpacing] = Field(ArrowSpacing.SPACING_32, width=24, description="Distance between arrows (pixels)")
    legend: Field[bool] =           Field(True, description="The strip's and histogram's log axis, the colormap's key")
    blend_mode: Field[Style.BlendMode] = Field(Style.BlendMode.ADD, width=32, description="How the monitor is drawn over the panel")


class MonitorLayerSettings(BaseSettings):
    """Configuration for MonitorLayer: what it shows, and in `advanced` how."""
    stage: Field[FlowStage] =       Field(FlowStage.BRIDGE, description="The stage whose texture is shown")
    names: Field[list[str]] =       Field([""], access=Field.READ, visible=False, description="The stage's textures")
    texture: Field[str] =           Field("velocity", widget=Widget.text_select, options=names, description="The texture shown")
    view: Field[MonitorView] =      Field(MonitorView.AUTO, description="By its kind, as an image, as a colormap centered on grey, as arrows, as a strip over time, or as a histogram")

    advanced = Group(MonitorAdvancedSettings)


@dataclass
class MonitorInput:
    """Every stage's ports, built by the app each tick; the monitor picks by its settings."""
    ports: dict[FlowStage, dict[str, Port]] = field(default_factory=dict)


_AUTO_VIEWS: dict[TextureKind, MonitorView] = {
    TextureKind.IMAGE: MonitorView.IMAGE,
    TextureKind.VELOCITY: MonitorView.ARROWS,
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

    def set_input(self, input: MonitorInput) -> None:
        """Every stage's ports, for the next draw."""
        self._input = input

    @property
    def texture(self) -> Texture:
        """The port as drawn last: the strip or the visualizer's output."""
        if self._view == MonitorView.STRIP:
            return self._strip.texture
        return self._visualizer.texture

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._visualizer.allocate(width, height)
        self._strip.allocate()
        self._legend.allocate()

    def deallocate(self) -> None:
        self._visualizer.deallocate()
        self._strip.deallocate()
        self._legend.deallocate()

    # ========== Processing ==========

    def _ports(self) -> dict[str, Port]:
        return self._input.ports.get(self.settings.stage, {}) if self._input is not None else {}

    def _pick(self) -> Port | None:
        """The port the settings name, else the stage's first."""
        ports: dict[str, Port] = self._ports()
        port: Port | None = ports.get(self.settings.texture)
        if port is None and ports:
            port = next(iter(ports.values()))
        return port if port is not None and port.texture.allocated else None

    def update(self) -> None:
        """Keep the panel's list of the stage's textures; the port is rendered at the first draw."""
        self._rendered = False
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
        if view == MonitorView.STRIP:
            shown: tuple[FlowStage, str, Channel] = (self.settings.stage, self.settings.texture, channel)
            if shown != self._strip_of:
                self._strip.reset()
                self._strip_of = shown
            self._strip.update(port.texture, scale, int(channel))
            self._legend_kind = LegendKind.STRIP
        elif view == MonitorView.IMAGE:
            self._visualizer.process_image(port.texture, scale, int(channel))
        elif view == MonitorView.COLORMAP:
            self._visualizer.process_colormap(port.texture, scale, int(channel))
            self._legend_kind = LegendKind.COLORMAP
        elif view == MonitorView.HISTOGRAM:
            self._visualizer.process_histogram(port.texture, scale, int(channel))
            self._legend_kind = LegendKind.HISTOGRAM
        else:
            self._visualizer.process_arrows(port.texture, scale, float(settings.advanced.arrow_spacing.value))
        if self._legend_kind is not None and settings.advanced.legend:
            self._legend.update(self._legend_kind, int(channel), scale, HISTOGRAM_TOTAL)

    # ========== Rendering ==========

    def draw(self) -> None:
        if not self._rendered:
            # Rendering into the port's FBOs sets the viewport to their size; the panel's comes back after
            viewport = glGetIntegerv(GL_VIEWPORT)
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
            Blit.use(self.texture)
        if self._legend_kind is not None and self.settings.advanced.legend and self._legend.texture.allocated:
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            Blit.use(self._legend.texture)
        Style.pop_style()
