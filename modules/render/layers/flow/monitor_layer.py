"""The monitor: a layer that shows one texture of one stage, picked on the panel, drawn only while
it is shown.

`MonitorInput` is what the app hands it each tick: every stage's ports. The monitor picks the stage
and the texture by its own settings, and draws their channels, all or one, by the view asked: as a
colormap by the port's kind (a volume as the cube), as glyphs, raw as an image, as a histogram of their
values, or as a strip of the channels' amounts over time, the last two under their log axis, the
colormap under its key (`Legend`). It renders in `draw`, once per tick however many panels show it, and nothing in
`update`, at the size it is drawn at (`DrawnSize`): the viewport the app fits it into, not the
size the app allocates the layer at."""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from OpenGL.GL import GL_VIEWPORT, glGetIntegerv, glViewport

from modules.gl import Texture, Texture3D, Style, Blit, MouseButton
from modules.flow import Visualizer, Strip, ALL_CHANNELS, Legend, LegendKind, DrawnSize, Port, TextureKind, VolumeCut, VolumeCube, CubeColor, VolumeGlyphs, Cut, Orbit, PITCH_LIMIT, ZOOM_LIMITS, box_size, axis_labels, REFERENCE_STEP
from modules.settings import Field, BaseSettings, Group, Widget
from ..layer_base import LayerBase


class FlowStage(IntEnum):
    """A stage of a flow system, as a monitor names it."""
    MOTION = 0
    BRIDGE = auto()
    FLUID = auto()


class MonitorView(IntEnum):
    """How the monitor draws the channels, the most used first."""
    COLORMAP = 0            # by the port's kind: a vector by its direction, an image as stored, a scalar centered on grey
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


class OrbitSettings(BaseSettings):
    """The 3D views' camera around the box, which the mouse turns and zooms, and the panel too."""
    yaw: Field[float] =             Field(0.0, min=-180.0, max=180.0, description="Turn around the vertical axis (degrees); 0 = from the front")
    pitch: Field[float] =           Field(0.0, min=-PITCH_LIMIT, max=PITCH_LIMIT, description="Turn up or down (degrees)")
    zoom: Field[float] =            Field(1.0, min=ZOOM_LIMITS[0], max=ZOOM_LIMITS[1], widget=Widget.log_slider, description="1 frames the box")


class MonitorAdvancedSettings(BaseSettings):
    """How the monitor draws: the channels, the scale, the glyphs' spacing, the legend and the blend."""
    channel: Field[Channel] =       Field(Channel.ALL, width=16, description="The channels drawn")
    scale: Field[float] =           Field(1.0, min=0.01, max=100.0, widget=Widget.log_slider, description="Multiplier on the values; in the cube, 1 after it is full")
    opacity: Field[float] =         Field(6.0, min=0.1, max=100.0, widget=Widget.log_slider, description="The cube: how much a full sample hides what is behind it, per box width")
    depth_fade: Field[float] =      Field(0.0, min=0.0, max=10.0, description="The cube: dimming per box width behind the near side; 0 = none")
    slice: Field[bool] =            Field(False, description="The cube: the scrolled layer framed and in focus, the rest dimmed")
    slice_position: Field[float] =  Field(0.5, min=0.0, max=1.0, description="The slice's place through the volume: 0 the front, 1 the back; the wheel steps it")
    slice_focus: Field[float] =     Field(0.5, min=0.0, max=1.0, description="Opacity moved from the rest to the slice: 0 none, 1 only the slice at its amount")
    glyph_spacing: Field[GlyphSpacing] = Field(GlyphSpacing.SPACING_32, width=24, description="Distance between glyphs (pixels)")
    legend: Field[bool] =           Field(True, description="The overlay: titles, log axes, the color key, and in 3D the box, its axes and the slice's frame")
    blend_mode: Field[Style.BlendMode] = Field(Style.BlendMode.ADD, width=32, description="How the monitor is drawn over the panel")

    orbit = Group(OrbitSettings)


class MonitorLayerSettings(BaseSettings):
    """Configuration for MonitorLayer: what it shows, and in `advanced` how."""
    stage: Field[FlowStage] =       Field(FlowStage.BRIDGE, description="The stage whose texture is shown")
    names: Field[list[str]] =       Field([""], access=Field.READ, visible=False, description="The stage's textures")
    texture: Field[str] =           Field("output.boundary_velocity", widget=Widget.text_select, options=names, description="The texture shown")
    view: Field[MonitorView] =      Field(MonitorView.COLORMAP, description="By its kind (in 3D the cube), as glyphs (arrows, or dots for a scalar), raw, as a histogram, or as a strip over time")

    advanced = Group(MonitorAdvancedSettings)


@dataclass
class MonitorInput:
    """Every stage's ports, built by the app each tick; the monitor picks by its settings."""
    ports: dict[FlowStage, dict[str, Port]] = field(default_factory=dict)


BOX_DEPTH: float = 1.0      # a volume's box is drawn as deep as it is wide, for now, not at its port's extent


_CUBE_COLORS: dict[TextureKind, CubeColor] = {
    TextureKind.IMAGE: CubeColor.AS_STORED,
    TextureKind.VELOCITY: CubeColor.DIRECTION,
    TextureKind.SCALAR: CubeColor.CENTERED,
    TextureKind.ACCELERATION: CubeColor.DIRECTION,
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
        self._view: MonitorView = MonitorView.COLORMAP   # the view rendered last; draw shows what render drew
        self._drawn_size: DrawnSize = DrawnSize()   # the size it renders at, from the viewports it is drawn into
        self._rendered_size: tuple[int, int] = (0, 0)   # the size this tick's port was rendered at
        self._strip_stepped: bool = False           # the strip got this tick's column
        self._volume_cut: VolumeCut = VolumeCut()   # a volume's front, flat, for the strip
        self._volume_cube: VolumeCube = VolumeCube()    # a volume in 3D, for the colormap and image views
        self._volume_glyphs: VolumeGlyphs = VolumeGlyphs()  # a volume in 3D, for the glyphs
        self._shown_3d: Texture | None = None       # what it rendered last, when it is one of the 3D views
        self._axes: list[tuple[str, float, float]] = []     # a 3D view's axis letters and their places, for the legend
        self._orbit: Orbit = Orbit()                # the 3D views' camera, set from `advanced.orbit` at each draw
        self._allocated: bool = False

    def set_input(self, input: MonitorInput) -> None:
        """Every stage's ports, for the next draw."""
        self._input = input

    # ========== The mouse ==========

    def drag(self, dx: float, dy: float, button: MouseButton) -> None:
        """A drag of `dx`, `dy` (shares of the window, y down): the left button turns the 3D view,
        the right zooms it. Render thread, from the window's callbacks."""
        orbit: OrbitSettings = self.settings.advanced.orbit
        camera: Orbit = Orbit(orbit.yaw, orbit.pitch, orbit.zoom)
        if button == MouseButton.LEFT:
            camera.turn(dx, dy)
            orbit.yaw, orbit.pitch = camera.yaw, camera.pitch
        elif button == MouseButton.RIGHT:
            camera.dolly(dy)
            orbit.zoom = camera.zoom

    def scroll(self, steps: float, layers: int) -> None:
        """A turn of the wheel: the slice one of `layers` further back per step away from the user,
        `layers` the show's density's, the same for every monitor, so they stay on one place."""
        if layers <= 0:
            return
        advanced: MonitorAdvancedSettings = self.settings.advanced
        layer: int = min(max(int(advanced.slice_position * layers) + round(steps), 0), layers - 1)
        advanced.slice_position = (layer + 0.5) / layers

    def reset_view(self) -> None:
        """The front view, framed, at the middle layer."""
        orbit: OrbitSettings = self.settings.advanced.orbit
        orbit.yaw, orbit.pitch, orbit.zoom = 0.0, 0.0, 1.0
        self.settings.advanced.slice_position = 0.5

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
        """Everything but the visualizer, which `_render` sizes to what it draws."""
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
        if port.prepare is not None:
            port.prepare()                      # its owner brings it up to date, or learns it is shown
        settings: MonitorLayerSettings = self.settings
        channel: Channel = settings.advanced.channel
        channels: int = port.channels
        if channel != Channel.ALL and channel >= channels:
            self._port = None                   # a channel the port does not hold: nothing to draw
            return
        scale: float = settings.advanced.scale
        if port.kind == TextureKind.ACCELERATION:
            scale *= REFERENCE_STEP             # the velocity it adds in a step, in a velocity's units
        self._view = view
        self._shown_3d = None
        self._axes = []
        by_kind: bool = view == MonitorView.COLORMAP and port.kind == TextureKind.IMAGE    # an image's colormap is itself
        if port.volume and view in (MonitorView.IMAGE, MonitorView.COLORMAP, MonitorView.GLYPHS):
            self._render_3d(port, view, scale, int(channel))
            if view != MonitorView.IMAGE and not by_kind:
                self._legend_kind = LegendKind.COLORMAP
            if settings.advanced.legend:
                self._legend.update(self._legend_kind, int(channel), channels, scale, self._visualizer.histogram_total, self._axes, self._title(port))
            return
        texture: Texture = self._flat(port, view)
        self._visualizer.resize(*self._rendered_size)
        if view == MonitorView.STRIP:
            shown: tuple[FlowStage, str, Channel] = (self.settings.stage, self.settings.texture, channel)
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
            self._visualizer.process_glyphs(texture, scale, float(settings.advanced.glyph_spacing.value),
                                            not port.kind.vector, int(channel), port.kind == TextureKind.ACCELERATION)
        if settings.advanced.legend:
            self._legend.update(self._legend_kind, int(channel), channels, scale, self._visualizer.histogram_total, title=self._title(port))

    def _title(self, port: Port) -> str:
        """What is shown: the stage and the port's name, the one picked when the settings name none; an
        acceleration drawn per step says so."""
        name: str = next((key for key, value in self._ports().items() if value is port), self.settings.texture)
        per_step: str = " per step" if port.kind == TextureKind.ACCELERATION else ""
        return f"{self.settings.stage.name.lower()} / {name}{per_step}"

    def _render_3d(self, port: Port, view: MonitorView, scale: float, channel: int) -> None:
        """A volume in its box from the orbit camera: for the colormap the cube colored by the port's kind,
        for the image the cube as stored, with `slice` the scrolled layer in focus; for the glyphs arrows
        of a velocity or dots of a scalar."""
        volume = port.texture
        assert isinstance(volume, Texture3D)
        width, height = self._rendered_size
        box: tuple[float, float, float] = box_size(volume.width, volume.height, BOX_DEPTH)
        orbit: OrbitSettings = self.settings.advanced.orbit
        self._orbit = Orbit(orbit.yaw, orbit.pitch, orbit.zoom)
        self._axes = axis_labels(self._orbit.view_projection(box, width / max(height, 1)), box, width, height)
        if view == MonitorView.GLYPHS:
            spacing: float = float(self.settings.advanced.glyph_spacing.value)
            self._shown_3d = self._volume_glyphs.process(volume, self._orbit, BOX_DEPTH, spacing, scale,
                                                         not port.kind.vector, channel, port.kind == TextureKind.ACCELERATION,
                                                         self.settings.advanced.legend)
        else:
            advanced = self.settings.advanced
            color: CubeColor = _CUBE_COLORS[port.kind] if view == MonitorView.COLORMAP else CubeColor.AS_STORED
            self._shown_3d = self._volume_cube.process(volume, self._orbit, BOX_DEPTH, scale, channel, color,
                                                       advanced.opacity, advanced.depth_fade,
                                                       advanced.slice_position if advanced.slice else None, advanced.legend,
                                                       advanced.slice_focus, port.light)

    def _flat(self, port: Port, view: MonitorView) -> Texture:
        """The 2D texture a view draws: the port's own, or for a volume its front, whose mean the strip
        reads, the volume's mean; the histogram counts the volume itself."""
        if not port.volume:
            assert isinstance(port.texture, Texture)
            return port.texture
        volume = port.texture
        assert isinstance(volume, Texture3D)
        if view == MonitorView.HISTOGRAM:
            return self._volume_cut.texture         # not drawn: the histogram counts the volume
        return self._volume_cut.process(volume, Cut.FRONT, 0.5, port.kind, BOX_DEPTH)

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
            Blit.use(self.texture)
        if self.settings.advanced.legend and self._legend.texture.allocated:
            Style.set_blend_mode(Style.BlendMode.ALPHA)
            Blit.use(self._legend.texture)
        Style.pop_style()
