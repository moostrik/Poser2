"""The monitor: a layer that shows one texture of one stage, picked on the panel, drawn only while
it is shown.

`MonitorInput` is what the app hands it each tick: every stage's ports. The monitor picks the stage
and the texture by its own settings, and draws it by its kind or by the mode asked: as the picture it
is, as arrows, as a signed scalar in two colors, or as a strip of its channels' amounts over time.
It renders in `draw`, once per tick however many panels show it, and nothing in `update`."""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from OpenGL.GL import GL_VIEWPORT, glGetIntegerv, glViewport

from modules.gl import Texture, Style, Text, Blit
from modules.flow import Visualizer, VisualizationFieldSettings, Strip, STRIP_LANES
from modules.settings import Field, BaseSettings, Group, Widget
from ..layer_base import LayerBase
from .port import Port, TextureKind


class FlowStage(IntEnum):
    """A stage of a flow system, as a monitor names it."""
    MOTION = 0
    BRIDGE = auto()
    FLUID = auto()


class MonitorMode(IntEnum):
    """How the monitor draws its texture: by the texture's kind, or as asked."""
    AUTO = 0
    IMAGE = auto()
    ARROWS = auto()
    SCALAR = auto()
    STRIP = auto()


class Channel(IntEnum):
    X = 0
    Y = auto()
    Z = auto()
    W = auto()


class MonitorLayerSettings(BaseSettings):
    """Configuration for MonitorLayer: what it shows and how."""
    stage: Field[FlowStage] =       Field(FlowStage.BRIDGE, description="The stage whose texture is shown")
    names: Field[list[str]] =       Field([""], access=Field.READ, visible=False, description="The stage's textures")
    texture: Field[str] =           Field("velocity", widget=Widget.text_select, options=names, description="The texture shown")
    mode: Field[MonitorMode] =      Field(MonitorMode.AUTO, description="Drawn by its kind, or as an image, as arrows, as a signed scalar, or as a strip of its channels' amounts over time")
    channel: Field[Channel] =       Field(Channel.X, description="The channel the scalar mode shows")
    blend_mode: Field[Style.BlendMode] = Field(Style.BlendMode.ADD)

    visualization = Group(VisualizationFieldSettings)


@dataclass
class MonitorInput:
    """Every stage's ports, built by the app each tick; the monitor picks by its settings."""
    textures: dict[FlowStage, dict[str, Port]] = field(default_factory=dict)


class MonitorLayer(LayerBase):
    """Shows one port of one stage. `set_input` before the render loop; `update` keeps the panel's list
    of names; `draw` renders the port once per tick and draws it."""

    def __init__(self, settings: MonitorLayerSettings) -> None:
        self.settings: MonitorLayerSettings = settings
        self._input: MonitorInput | None = None
        self._visualizer: Visualizer = Visualizer(settings.visualization)
        self._strip: Strip = Strip()
        self._text: Text = Text()
        self._rendered: bool = False            # this tick's port is drawn into the visualizer or the strip
        self._strip_of: tuple[FlowStage, str] | None = None     # the port the strip plots; another one restarts it
        self._port: Port | None = None          # the port rendered last

    def set_input(self, input: MonitorInput) -> None:
        """Every stage's ports, for the next draw."""
        self._input = input

    @property
    def texture(self) -> Texture:
        """The port as drawn last: the strip, or the visualizer's output."""
        if self.settings.mode == MonitorMode.STRIP:
            return self._strip.texture
        return self._visualizer.texture

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._visualizer.allocate(width, height)
        self._strip.allocate()
        self._text.allocate(font_size=18)

    def deallocate(self) -> None:
        self._visualizer.deallocate()
        self._strip.deallocate()
        self._text.deallocate()

    # ========== Processing ==========

    def _ports(self) -> dict[str, Port]:
        return self._input.textures.get(self.settings.stage, {}) if self._input is not None else {}

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
        if port is None:
            return
        mode: MonitorMode = self.settings.mode
        channel: int = int(self.settings.channel)
        if mode == MonitorMode.STRIP:
            shown: tuple[FlowStage, str] = (self.settings.stage, self.settings.texture)
            if shown != self._strip_of:
                self._strip.reset()
                self._strip_of = shown
            self._strip.update(port.texture, self.settings.visualization.strip_scale)
        elif mode == MonitorMode.IMAGE or (mode == MonitorMode.AUTO and port.kind == TextureKind.IMAGE):
            self._visualizer.process(port.texture)
        elif mode == MonitorMode.ARROWS or (mode == MonitorMode.AUTO and port.kind == TextureKind.VELOCITY):
            self._visualizer.process(port.texture, True)
        else:
            self._visualizer.process_signed(port.texture, port.kind == TextureKind.DEPTH, channel)

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
        Style.set_blend_mode(self.settings.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        if self.settings.mode == MonitorMode.STRIP and self._text.allocated:
            self._draw_lane_labels()
        Style.pop_style()

    def _draw_lane_labels(self) -> None:
        """The name of each lane of the strip, at its top left."""
        vp = glGetIntegerv(GL_VIEWPORT)
        w, h = int(vp[2]), int(vp[3])
        if w <= 0 or h <= 0:
            return
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for lane, name in enumerate(STRIP_LANES):
            self._text.draw_box_text(8, lane * h // len(STRIP_LANES) + 8, name, (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.6), w, h)
