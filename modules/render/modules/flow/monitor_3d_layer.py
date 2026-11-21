"""The 3D monitor: the flat monitor (`Monitor2DLayer`) with volumes. A volume is drawn in its box from an orbit
camera, as the cube colored by the port's kind or as stored, with a slice in focus, or as glyphs; its front
flat for the strip. The mouse turns and zooms the camera and steps the slice."""

from modules.gl import Texture, Texture3D
from modules.window import MouseButton
from modules.flow import LegendKind, Port, TextureKind, ColorSpace, VolumeCut, VolumeCube, CubeColor, VolumeGlyphs, Cut, Orbit, \
                         PITCH_LIMIT, ZOOM_LIMITS, box_size, axis_labels
from modules.settings import Field, BaseSettings, Group, Widget
from .monitor_2d_layer import Monitor2DLayer, Monitor2DLayerSettings, Monitor2DAdvancedSettings, MonitorView, FlowStage, Channel


class OrbitSettings(BaseSettings):
    """The 3D views' camera around the box, which the mouse turns and zooms, and the panel too."""
    yaw: Field[float] =             Field(0.0, min=-180.0, max=180.0, description="Turn around the vertical axis (degrees); 0 = from the front", access=Field.READWRITE)
    pitch: Field[float] =           Field(0.0, min=-PITCH_LIMIT, max=PITCH_LIMIT, description="Turn up or down (degrees)", access=Field.READWRITE)
    zoom: Field[float] =            Field(1.0, min=ZOOM_LIMITS[0], max=ZOOM_LIMITS[1], widget=Widget.log_slider, description="1 frames the box", access=Field.READWRITE)


class Monitor3DAdvancedSettings(Monitor2DAdvancedSettings):
    """How the 3D monitor draws: the flat monitor's, and the cube's opacity, depth fade and slice, and the orbit."""
    scale: Field[float] =           Field(1.0, min=0.01, max=100.0, widget=Widget.log_slider, description="Multiplier on the values; in the cube, 1 after it is full")
    legend: Field[bool] =           Field(True, description="The overlay: titles, log axes, the color key, and in 3D the box, its axes and the slice's frame")
    opacity: Field[float] =         Field(6.0, min=0.1, max=100.0, widget=Widget.log_slider, description="The cube: how much a full sample hides what is behind it, per box width")
    depth_fade: Field[float] =      Field(0.0, min=0.0, max=10.0, description="The cube: dimming per box width behind the near side; 0 = none")
    slice: Field[bool] =            Field(False, description="The cube: the scrolled layer framed and in focus, the rest dimmed")
    slice_position: Field[float] =  Field(0.5, min=0.0, max=1.0, description="The slice's place through the volume: 0 the front, 1 the back; the wheel steps it", access=Field.READWRITE)
    slice_focus: Field[float] =     Field(0.5, min=0.0, max=1.0, description="Opacity moved from the rest to the slice: 0 none, 1 only the slice at its amount")

    orbit = Group(OrbitSettings, description="The 3D views' camera around the box")


class Monitor3DLayerSettings(Monitor2DLayerSettings):
    """Configuration for Monitor3DLayer: the flat monitor's, its `advanced` with the volume views'."""
    advanced = Group(Monitor3DAdvancedSettings, description="How the monitor draws, the volume views included")


def box_depth(port: Port) -> float:
    """How deep a volume's box is drawn, as a share of its width: its port's extent, as deep as the box it stands
    for; as deep as it is wide where the port gives none."""
    return port.extent if port.extent > 0.0 else 1.0


_CUBE_COLORS: dict[TextureKind, CubeColor] = {
    TextureKind.IMAGE: CubeColor.AS_STORED,
    TextureKind.VELOCITY: CubeColor.DIRECTION,
    TextureKind.SCALAR: CubeColor.CENTERED,
    TextureKind.ACCELERATION: CubeColor.DIRECTION,
    TextureKind.Z_POSITION: CubeColor.CENTERED,
    TextureKind.DEPTH: CubeColor.CENTERED,
}


def _image_color(port: Port) -> CubeColor:
    """How the image view colors a volume: as stored, or an image stored as Oklab turned back to RGB, as the
    composite shows it."""
    oklab: bool = port.kind == TextureKind.IMAGE and port.color_space == ColorSpace.OKLAB
    return CubeColor.FROM_OKLAB if oklab else CubeColor.AS_STORED


class Monitor3DLayer(Monitor2DLayer):
    """Shows one port of its stage, a volume too; the mouse (`drag`, `scroll`, `reset_view`) moves its 3D view."""

    def __init__(self, stage: FlowStage, settings: Monitor3DLayerSettings) -> None:
        super().__init__(stage, settings)
        self.settings: Monitor3DLayerSettings = settings
        self._volume_cut: VolumeCut = VolumeCut()   # a volume's front, flat, for the strip
        self._volume_cube: VolumeCube = VolumeCube()    # a volume in 3D, for the colormap and image views
        self._volume_glyphs: VolumeGlyphs = VolumeGlyphs()  # a volume in 3D, for the glyphs
        self._axes: list[tuple[str, float, float]] = []     # a 3D view's axis letters and their places, for the legend
        self._orbit: Orbit = Orbit()                # the 3D views' camera, set from `advanced.orbit` at each draw

    # ========== The mouse ==========

    def drag(self, dx: float, dy: float, button: MouseButton) -> None:
        """A drag of `dx`, `dy` (shares of the window, y down): the left button turns the 3D view,
        the right zooms it. Main thread, from the window's callbacks."""
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
        advanced: Monitor3DAdvancedSettings = self.settings.advanced
        layer: int = min(max(int(advanced.slice_position * layers) + round(steps), 0), layers - 1)
        advanced.slice_position = (layer + 0.5) / layers

    def reset_view(self) -> None:
        """The front view, framed, at the middle layer."""
        orbit: OrbitSettings = self.settings.advanced.orbit
        orbit.yaw, orbit.pitch, orbit.zoom = 0.0, 0.0, 1.0
        self.settings.advanced.slice_position = 0.5

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        super().allocate(width, height, internal_format)
        self._volume_cut.allocate()
        self._volume_cube.allocate()
        self._volume_glyphs.allocate()

    def deallocate(self) -> None:
        self._volume_cut.deallocate()
        self._volume_cube.deallocate()
        self._volume_glyphs.deallocate()
        super().deallocate()

    def _resize(self, width: int, height: int) -> None:
        super()._resize(width, height)
        self._volume_cube.resize(width, height)
        self._volume_glyphs.resize(width, height)

    # ========== Processing ==========

    def _render_volume(self, port: Port, view: MonitorView, scale: float, channel: Channel, channels: int,
                       by_kind: bool) -> bool:
        """A volume in 3D for the image, the colormap and the glyphs; the strip and the histogram take it flat."""
        if view not in (MonitorView.IMAGE, MonitorView.COLORMAP, MonitorView.GLYPHS):
            return False
        self._axes = []
        self._render_3d(port, view, scale, int(channel))
        if view != MonitorView.IMAGE and not by_kind:
            self._legend_kind = LegendKind.COLORMAP
        if self.settings.advanced.legend:
            self._legend.update(self._legend_kind, int(channel), channels, scale, self._visualizer.histogram_total, self._axes, self._title(port))
        return True

    def _render_3d(self, port: Port, view: MonitorView, scale: float, channel: int) -> None:
        """A volume in its box from the orbit camera: for the colormap the cube colored by the port's kind,
        for the image the cube as stored, with `slice` the scrolled layer in focus; for the glyphs arrows
        of a velocity or dots of a scalar."""
        volume = port.texture
        assert isinstance(volume, Texture3D)
        width, height = self._rendered_size
        depth: float = box_depth(port)
        box: tuple[float, float, float] = box_size(volume.width, volume.height, depth)
        advanced: Monitor3DAdvancedSettings = self.settings.advanced
        orbit: OrbitSettings = advanced.orbit
        self._orbit = Orbit(orbit.yaw, orbit.pitch, orbit.zoom)
        self._axes = axis_labels(self._orbit.view_projection(box, width / max(height, 1)), box, width, height)
        if view == MonitorView.GLYPHS:
            spacing: float = float(advanced.glyph_spacing.value)
            self._shown = self._volume_glyphs.process(volume, self._orbit, depth, spacing, scale,
                                                      not port.kind.vector, channel, port.kind == TextureKind.ACCELERATION,
                                                      advanced.legend)
        else:
            color: CubeColor = _CUBE_COLORS[port.kind] if view == MonitorView.COLORMAP else _image_color(port)
            self._shown = self._volume_cube.process(volume, self._orbit, depth, scale, channel, color,
                                                    advanced.opacity, advanced.depth_fade,
                                                    advanced.slice_position if advanced.slice else None, advanced.legend,
                                                    advanced.slice_focus, port.light)

    def _flat(self, port: Port, view: MonitorView) -> Texture:
        """The 2D texture a view draws: a flat port's own, or for a volume its front, whose mean the strip
        reads, the volume's mean; the histogram counts the volume itself."""
        if not port.volume:
            return super()._flat(port, view)
        volume = port.texture
        assert isinstance(volume, Texture3D)
        if view == MonitorView.HISTOGRAM:
            return self._volume_cut.texture         # not drawn: the histogram counts the volume
        return self._volume_cut.process(volume, Cut.FRONT, 0.5, port.kind, box_depth(port))
