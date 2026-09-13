"""White Space render — layer graph for 3-camera panoramic light installation."""

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, clear_color, Style
from modules.render.layers import LayerBase
from modules.render.layers import ImageSourceLayer, CropSourceLayer
from modules.render.layers import TrackerCompositor, PoseCompositor
from modules.render.layers import FeatureWindowLayer, FeatureFrameLayer, MTimeRenderer
from modules.render.layers import Compositor, PanoramaLayerSettings
from modules.oak import mono_frame_size
from modules.tracker import PanoramicTrackerSettings
from apps.white_space.render.layers.light_simulation_layer import LightSimulationLayer
from apps.white_space.render.layers.beam_light_simulation_layer import BeamLightSimulationLayer
from apps.white_space.render.layers.azimuth_overlay_layer import AzimuthOverlayLayer
from apps.white_space.light import FIXTURE_PROJECTION_RPM
from modules.utils.PointsAndRects import Rect, Point2f
from modules.render.composition_subdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.utils.HotReloadMethods import HotReloadMethods

from ..board import Board
from ..pose import PlayheadOffset, GhostFeature
from ..settings import Layers, RenderSettings, PlayheadFeatureSelect, CameraView

# Maps the app-local feature dropdown to the concrete app features. Kept here (not in
# settings, which stays data-only) and handed to the generic data layers via feature_map.
PLAYHEAD_FEATURE_MAP = {
    PlayheadFeatureSelect.PlayheadOffset: PlayheadOffset,
    PlayheadFeatureSelect.GhostFeature:   GhostFeature,
}

# The window's ground, behind and between the rows. Dark grey rather than black so a row's own
# black — an unlit camera, the panorama outside its elevation window — reads as content rather than
# as a hole in the window, and the gaps between rows show where one ends.
_BACKGROUND: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 1.0)

# The two rows `camera_view` switches between, and the layer whose compositing each one needs.
# A hidden row's layers are not updated at all — hiding the strip skips a whole stitch per frame.
_SWITCHED_ROWS: dict[str, Layers] = {
    'track':     Layers.tracker,
    'panoramic': Layers.cam_panorama,
}


class Render(RenderBase):
    def __init__(self, board: Board, settings: RenderSettings,
                 tracker: PanoramicTrackerSettings) -> None:
        super().__init__(settings.window)
        self.num_players: int = settings.num_players
        self.num_cams: int = settings.num_cams
        self.settings: RenderSettings = settings
        # The tracker's own geometry, live. Anything drawing the tracker's world has to use the
        # numbers the tracker used, or the display invents an error of its own.
        self.tracker_settings: PanoramicTrackerSettings = tracker
        self.board: Board = board

        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        # Row 1 — per-camera: source layers + tracker compositor
        for i in range(self.num_cams):
            self.L[Layers.cam_image][i] = ImageSourceLayer(i, board)
            self.L[Layers.cam_crop][i]  = CropSourceLayer(i, board)
            self.L[Layers.tracker][i]   = TrackerCompositor(
                i, board,
                self.L[Layers.cam_image][i].texture,
                settings.preview.tracker,
                settings.colors,
            )

        # Row 5 — per-player: pose compositor + data overlays
        # cam_image[0] texture used as fallback for non-GPU crop path (GPU crop is default)
        fallback_cam_texture = self.L[Layers.cam_image][0].texture
        for i in range(self.num_players):
            self.L[Layers.poser][i]     = PoseCompositor(
                i, board,
                fallback_cam_texture,
                settings.preview.poser,
                settings.colors,
            )
            self.L[Layers.data_W][i]    = FeatureWindowLayer(i, board, settings.data, settings.colors) # type: ignore
            self.L[Layers.data_F][i]    = FeatureFrameLayer( i, board, settings.data, settings.colors)   # type: ignore
            self.L[Layers.data_time][i] = MTimeRenderer(     i, board, settings.data_time)
            self.L[Layers.data_playhead_W][i] = FeatureWindowLayer(i, board, settings.playhead_data, settings.colors, feature_map=PLAYHEAD_FEATURE_MAP) # type: ignore
            self.L[Layers.data_playhead_F][i] = FeatureFrameLayer( i, board, settings.playhead_data, settings.colors, feature_map=PLAYHEAD_FEATURE_MAP) # type: ignore

        # Rows 2–4 — shared panoramic layers; constructed after cam layers so textures are ready.
        # The calibration strip owns row 2 by itself: it consumes all four camera images at once and
        # draws the tracker's own view of the same people over them, on one vertical scale.
        self.L[Layers.cam_panorama][0] = Compositor(
            board,
            [self.L[Layers.cam_image][i].texture for i in range(self.num_cams)],
            self.num_cams, settings.colors, tracker, settings.panorama,
        )
        self.L[Layers.ws_light][0]   = LightSimulationLayer(board)
        self.L[Layers.ws_beam][0]     = BeamLightSimulationLayer(board, settings.beam_light_sim)
        self.L[Layers.ws_azimuth][0]  = AzimuthOverlayLayer(board, settings.colors)

        self.subdivision_rows: list[SubdivisionRow] = self._build_rows()
        self._window_size: tuple[int, int] = (settings.window.width, settings.window.height)
        self._align_center: bool = False
        self.subdivision: Subdivision = make_subdivision(
            self.subdivision_rows, *self._window_size, self._align_center
        )

        # The strip's height follows the focus depth: a nearer cylinder is seen over a narrower
        # band of elevation from the centre, so dragging this reshapes the row — and reshaping means
        # GL reallocation, which only the render thread may do. The callback arrives on whichever
        # thread wrote the setting, so it raises a flag and `update()` acts on it.
        self._layout_dirty: bool = False
        settings.panorama.bind(PanoramaLayerSettings.focus_radius, self._on_layout_setting)
        RenderSettings.camera_view.bind(settings, self._on_layout_setting)

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def _build_rows(self) -> list[SubdivisionRow]:
        """The rows the window shows, top to bottom.

        `camera_view` decides which of the two camera rows are among them. Every row's height is
        its content's aspect over the shared width, so leaving one out hands its height to the
        others rather than leaving a gap — which is the point of the switch.
        """
        view: CameraView = self.settings.camera_view
        rows: list[SubdivisionRow] = []
        if view in (CameraView.CAMERAS, CameraView.BOTH):
            rows.append(self._track_row())
        if view in (CameraView.PANORAMA, CameraView.BOTH):
            rows.append(self._panorama_row())
        rows.append(SubdivisionRow(name='ws_light', columns=1,                rows=1, src_aspect_ratio=6.0,  padding=Point2f(0.0, 1.0)))
        rows.append(SubdivisionRow(name='pose',     columns=self.num_players, rows=1, src_aspect_ratio=0.75, padding=Point2f(1.0, 1.0)))
        return rows

    def _hidden_layers(self) -> set[Layers]:
        """The layers belonging to rows this layout leaves out — neither composited nor drawn."""
        return {layer for name, layer in _SWITCHED_ROWS.items() if not self.subdivision.has(name)}

    def _track_row(self) -> SubdivisionRow:
        """Row 1: one view per camera, always — the raw frames the strip below is derived from.

        Mono and landscape, as this app has always been; the frame's shape follows the configured
        sensor mode and the delivered height, so switching either reshapes the row with it.
        """
        frame_w, frame_h = mono_frame_size(self.settings.resolution, height=self.settings.frame_height)
        return SubdivisionRow(name='track', columns=self.num_cams, rows=1,
                              src_aspect_ratio=frame_w / frame_h, padding=Point2f(1.0, 1.0))

    def _panorama_row(self) -> SubdivisionRow:
        """Row 2: the whole ring as one 360° strip, under the raw frames.

        The aspect is the compositor's own — 360 degrees of azimuth over the elevations it can
        actually fill, both measured at the rig centre. It is not a preference: `tilt`, `fov` and
        `focus_radius` all move it, and the compositor is the only thing that knows how.
        """
        panorama: Compositor = self.L[Layers.cam_panorama][0]  # type: ignore[assignment]
        return SubdivisionRow(name='panoramic', columns=1, rows=1,
                              src_aspect_ratio=panorama.aspect_ratio,
                              padding=Point2f(0.0, 1.0))

    def _on_layout_setting(self, _value: object) -> None:
        self._layout_dirty = True

    def _rebuild_layout(self) -> None:
        self.subdivision_rows = self._build_rows()
        self.subdivision = make_subdivision(
            self.subdivision_rows, *self._window_size, self._align_center
        )
        self.allocate_window_renders()

    def on_main_window_resize(self, width: int, height: int) -> None:
        self._window_size = (width, height)
        self._align_center = True
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()

    def allocate(self) -> None:
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                layer.allocate(1080, 1920, GL_RGBA16F)
        self.allocate_window_renders()

    def allocate_window_renders(self) -> None:
        # Only the rows this layout holds: a hidden row keeps whatever it had and is never drawn,
        # so allocating it to the fallback rect would be wasted memory at the wrong shape.
        if self.subdivision.has('track'):
            for i in range(self.num_cams):
                w, h = self.subdivision.get_allocation_size('track', i)
                self.L[Layers.tracker][i].allocate(w, h, GL_RGBA)

        if self.subdivision.has('panoramic'):
            w, h = self.subdivision.get_allocation_size('panoramic', 0)
            self.L[Layers.cam_panorama][0].allocate(w, h, GL_RGBA)

        w, h = self.subdivision.get_allocation_size('ws_light', 0)
        self.L[Layers.ws_azimuth][0].allocate(w, h, GL_RGBA)

        for i in range(self.num_players):
            w, h = self.subdivision.get_allocation_size('pose', i)
            self.L[Layers.poser][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.settings.panorama.unbind(PanoramaLayerSettings.focus_radius, self._on_layout_setting)
        RenderSettings.camera_view.unbind(self.settings, self._on_layout_setting)
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()

    def update(self) -> None:
        self._notify_update()

        if self._layout_dirty:
            self._layout_dirty = False
            self._rebuild_layout()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        hidden: set[Layers] = self._hidden_layers()
        for layer_type, cam_dict in self.L.items():
            if layer_type in hidden:
                continue
            for layer in cam_dict.values():
                layer.update()

    def _viewport(self, height: int, rect: Rect) -> None:
        glViewport(
            int(rect.x),
            int(height - rect.y - rect.height),
            int(rect.width),
            int(rect.height),
        )

    def draw_main(self, width: int, height: int) -> None:
        clear_color(*_BACKGROUND)
        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        # Row 1 — one tracker compositor per camera: the raw frames. Absent under PANORAMA.
        if self.subdivision.has('track'):
            for i in range(self.num_cams):
                self._viewport(height, self.subdivision.get_rect('track', i))
                self.L[Layers.tracker][i].draw()

        # Row 2 — the whole ring as one 360° strip, image and tracker data on one vertical scale.
        # Absent under CAMERAS.
        if self.subdivision.has('panoramic'):
            self._viewport(height, self.subdivision.get_rect('panoramic', 0))
            self.L[Layers.cam_panorama][0].draw()

        # Row 3 - WS light strip: the ring, or the bar's lights while the fixture is in beam mode —
        # the same rule the fixture applies to the same command (the frame's target rpm).
        output = self.board.get_composition_output()
        beam_mode = output is not None and output.motor_command.target_rpm < FIXTURE_PROJECTION_RPM
        self._viewport(height, self.subdivision.get_rect('ws_light', 0))
        self.L[Layers.ws_beam if beam_mode else Layers.ws_light][0].draw()
        if self.settings.azimuth_overlay:
            self.L[Layers.ws_azimuth][0].draw()

        # Row 4 - pose cutouts with data overlays, one viewport per player
        for i in range(self.num_players):
            self._viewport(height, self.subdivision.get_rect('pose', i))
            self.L[Layers.poser][i].draw()
            self.L[Layers.data_W][i].draw()
            self.L[Layers.data_F][i].draw()
            self.L[Layers.data_time][i].draw()
            self.L[Layers.data_playhead_W][i].draw()
            self.L[Layers.data_playhead_F][i].draw()

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        pass
