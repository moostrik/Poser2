"""White Space render — layer graph for 3-camera panoramic light installation."""

from collections.abc import Sequence
from enum import IntEnum

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, clear_color, Style
from modules.camera import RigCameraSettings, CameraCheckSettings, mono_frame_size, delivered_height
from modules.tracker import PanoramicTrackerSettings
from apps.white_space.render.layers.light_simulation_layer import LightSimulationLayer
from apps.white_space.render.layers.beam_light_simulation_layer import BeamLightSimulationLayer
from apps.white_space.render.layers.azimuth_overlay_layer import AzimuthOverlayLayer
from apps.white_space.render.layers.pose_figure_layer import PoseFigureLayer
from apps.white_space.light import FIXTURE_PROJECTION_RPM
from modules.pose import BaseScalarFeature, nodes
from modules.utils import Color, Rect, Point2f
from modules.render import layers, make_subdivision, SubdivisionRow, Subdivision

from ..board import RenderBoard
from ..pose import PlayheadOffset, dummy_id
from ..settings import Layers, RenderSettings, PlayheadFeatureSelect, CameraView, Layout, Stage
from .pose_slots import POSE_SLOTS, pose_slots

# Maps the app-local feature enum to the concrete app feature. Kept here (not in settings,
# which stays data-only) and handed to the generic data layers via feature_map.
PLAYHEAD_FEATURE_MAP: dict[IntEnum, type[BaseScalarFeature]] = {
    PlayheadFeatureSelect.PlayheadOffset: PlayheadOffset,
}

# The window's ground, behind and between the rows. Dark grey rather than black so a row's own
# black — an unlit camera, the panorama outside its elevation window — reads as content rather than
# as a hole in the window, and the gaps between rows show where one ends.
_BACKGROUND: tuple[float, float, float, float] = (0.15, 0.15, 0.15, 1.0)

# The rows a layout may leave out (`camera_view` one of the two camera rows, POSE both), and the
# layers whose compositing each one needs. A hidden row's layers are not updated at all — hiding
# the strip skips a whole stitch per frame.
_SWITCHED_ROWS: dict[str, tuple[Layers, ...]] = {
    'track':     (Layers.tracker,),
    'panoramic': (Layers.cam_panorama,),
    'ws_light':  (Layers.light_simulation, Layers.beam_light_simulation, Layers.azimuth_overlay, Layers.pose_figures),
}


class WhiteSpaceRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings,
                 tracker: PanoramicTrackerSettings,
                 cameras: Sequence[RigCameraSettings], camera_check: CameraCheckSettings,
                 angle_extractor: nodes.AngleExtractorSettings) -> None:
        super().__init__(settings.window)
        self.max_players: int = settings.max_players
        self.num_cams: int = settings.num_cams
        self.dummy_id: int = dummy_id(self.max_players)
        self._pose_slots: list[int | None] = [None] * POSE_SLOTS     # POSE layout: which pose is in which slot
        self.settings: RenderSettings = settings
        # The tracker's own geometry, live. Anything drawing the tracker's world has to use the
        # numbers the tracker used, or the display invents an error of its own.
        self.tracker_settings: PanoramicTrackerSettings = tracker
        self.board: RenderBoard = board

        self.L: dict[Layers, dict[int, layers.LayerBase]] = {layer: {} for layer in Layers}

        # Row 1 — per-camera: source layers + tracker compositor
        for i in range(self.num_cams):
            self.L[Layers.cam_image][i] = layers.ImageSourceLayer(i, board)
            self.L[Layers.cam_crop][i]  = layers.CropSourceLayer(i, board)
            self.L[Layers.tracker][i]   = layers.TrackerCompositor(
                i, board,
                self.L[Layers.cam_image][i].texture,
                settings.preview.tracker,
                settings.colors,
            )
            self.L[Layers.cam_readings][i] = layers.CameraReadingsLayer(cameras[i], camera_check)

        # Row 5 — per-player: pose compositor + data overlays
        # cam_image[0] texture used as fallback for non-GPU crop path (GPU crop is default)
        fallback_cam_texture = self.L[Layers.cam_image][0].texture
        for i in range(self.max_players):
            self.L[Layers.pose][i]     = layers.PoseCompositor(
                i, board,
                fallback_cam_texture,
                settings.preview.pose,
                settings.colors,
            )
            self.L[Layers.data_W][i]    = layers.FeatureWindowLayer(i, board, settings.data, settings.colors)
            self.L[Layers.data_F][i]    = layers.FeatureFrameLayer( i, board, settings.data, settings.colors)
            self.L[Layers.data_time][i] = layers.MotionTimeRenderer(i, board, settings.data_time)
            self.L[Layers.data_playhead_W][i] = layers.FeatureWindowLayer(i, board, settings.playhead_data, settings.colors, feature_map=PLAYHEAD_FEATURE_MAP)
            self.L[Layers.data_playhead_F][i] = layers.FeatureFrameLayer( i, board, settings.playhead_data, settings.colors, feature_map=PLAYHEAD_FEATURE_MAP)

        # The dummy's column, last in the same row: its skeleton at LERP, the only stage it is in
        # (the pose compositor overlays CLEAN, SMOOTH and PREDICT over a crop it has neither of),
        # under the same data overlays as a player's. White: it is not a player.
        d = self.dummy_id
        self.L[Layers.dummy_pose][d] = layers.PoseLineLayer(d, board, layers.PoseLineLayerSettings(
            stage=int(Stage.LERP),
            line_width=settings.preview.pose.line_width,
            line_smooth=settings.preview.pose.line_smooth,
            use_scores=True, use_bbox=False, color=Color(1.0, 1.0, 1.0),
        ))
        self.L[Layers.data_W][d]    = layers.FeatureWindowLayer(d, board, settings.data, settings.colors)
        self.L[Layers.data_F][d]    = layers.FeatureFrameLayer( d, board, settings.data, settings.colors)
        self.L[Layers.data_time][d] = layers.MotionTimeRenderer(d, board, settings.data_time)
        self.L[Layers.data_playhead_W][d] = layers.FeatureWindowLayer(d, board, settings.playhead_data, settings.colors, feature_map=PLAYHEAD_FEATURE_MAP)
        self.L[Layers.data_playhead_F][d] = layers.FeatureFrameLayer( d, board, settings.playhead_data, settings.colors, feature_map=PLAYHEAD_FEATURE_MAP)

        # Rows 2–4 — shared panoramic layers; constructed after cam layers so textures are ready.
        # The calibration strip owns row 2 by itself: it consumes all four camera images at once and
        # draws the tracker's own view of the same people over them, on one vertical scale.
        self.L[Layers.cam_panorama][0] = layers.Compositor(
            board,
            [self.L[Layers.cam_image][i].texture for i in range(self.num_cams)],
            self.num_cams, settings.colors, tracker, settings.panorama,
        )
        self.L[Layers.light_simulation][0]      = LightSimulationLayer(board)
        self.L[Layers.beam_light_simulation][0] = BeamLightSimulationLayer(board, settings.beam_light_sim)
        self.L[Layers.azimuth_overlay][0]       = AzimuthOverlayLayer(board, settings.colors)
        self.L[Layers.pose_figures][0]          = PoseFigureLayer(board, settings.colors, angle_extractor)

        self.subdivision_rows: list[SubdivisionRow] = self._build_rows()
        self._window_size: tuple[int, int] = (settings.window.width, settings.window.height)
        self._align_center: bool = False
        self.subdivision: Subdivision = make_subdivision(
            self.subdivision_rows, *self._window_size, self._align_center
        )

        # The strip's height follows the focus depth: a nearer cylinder is seen over a narrower
        # band of elevation from the center, so dragging this reshapes the row — and reshaping means
        # GL reallocation, which only the render thread may do. The callback arrives on whichever
        # thread wrote the setting, so it raises a flag and `update()` acts on it.
        self._layout_dirty: bool = False
        settings.panorama.bind(layers.PanoramaLayerSettings.focus_radius, self._on_layout_setting)
        RenderSettings.camera_view.bind(settings, self._on_layout_setting)
        RenderSettings.layout.bind(settings, self._on_layout_setting)


    def _build_rows(self) -> list[SubdivisionRow]:
        """The rows the window shows, top to bottom.

        POSE: the three longest-present poses, large, over the projection. CAMERA: the camera
        rows `camera_view` picks, the projection and the pose row. Every row's height is its
        content's aspect over the shared width, so leaving one out hands its height to the others
        rather than leaving a gap — which is the point of the switches.
        """
        projection = SubdivisionRow(name='ws_light', columns=1, rows=1, src_aspect_ratio=6.0, padding=Point2f(0.0, 1.0))
        if self.settings.layout == Layout.POSE:
            slots = SubdivisionRow(name='slots', columns=POSE_SLOTS, rows=1, src_aspect_ratio=0.75, padding=Point2f(1.0, 1.0))
            return [slots, projection]
        view: CameraView = self.settings.camera_view
        rows: list[SubdivisionRow] = []
        if view in (CameraView.CAMERAS, CameraView.BOTH):
            rows.append(self._track_row())
        if view in (CameraView.PANORAMA, CameraView.BOTH):
            rows.append(self._panorama_row())
        rows.append(projection)
        # The pose row: one column per player, and the dummy's last (index max_players, its id).
        rows.append(SubdivisionRow(name='pose', columns=self.max_players + 1, rows=1, src_aspect_ratio=0.75, padding=Point2f(1.0, 1.0)))
        return rows

    def _hidden_layers(self) -> set[Layers]:
        """The layers belonging to rows this layout leaves out — neither composited nor drawn."""
        return {layer for name, layers in _SWITCHED_ROWS.items() if not self.subdivision.has(name) for layer in layers}

    def _skeleton(self, track_id: int) -> layers.LayerBase:
        """A pose's crop-and-skeleton layer: a player's compositor, or the dummy's lines."""
        return self.L[Layers.dummy_pose if track_id == self.dummy_id else Layers.pose][track_id]

    def _track_row(self) -> SubdivisionRow:
        """Row 1: one view per camera, always — the raw frames the strip below is derived from.

        Mono and landscape; the frame's shape is derived from the tracker's camera fields as the
        camera derives it, so a change of sensor mode or tilt reshapes the row with it.
        """
        T = self.tracker_settings
        rows = delivered_height(False, T.resolution, T.fov, T.tilt, T.lens_fov, (T.lens_center_x, T.lens_center_y))
        frame_w, frame_h = mono_frame_size(T.resolution, height=rows)
        return SubdivisionRow(name='track', columns=self.num_cams, rows=1,
                              src_aspect_ratio=frame_w / frame_h, padding=Point2f(1.0, 1.0))

    def _panorama_row(self) -> SubdivisionRow:
        """Row 2: the whole rig as one 360° strip, under the raw frames.

        The aspect is the compositor's own — 360 degrees of azimuth over the elevations it can
        actually fill, both measured at the rig center. It is not a preference: `tilt`, `fov` and
        `focus_radius` all move it, and the compositor is the only thing that knows how.
        """
        panorama: layers.Compositor = self.L[Layers.cam_panorama][0]  # type: ignore[assignment]
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
                self.L[Layers.cam_readings][i].allocate(w, h, GL_RGBA)

        if self.subdivision.has('panoramic'):
            w, h = self.subdivision.get_allocation_size('panoramic', 0)
            self.L[Layers.cam_panorama][0].allocate(w, h, GL_RGBA)

        if self.subdivision.has('ws_light'):
            w, h = self.subdivision.get_allocation_size('ws_light', 0)
            self.L[Layers.azimuth_overlay][0].allocate(w, h, GL_RGBA)
            self.L[Layers.pose_figures][0].allocate(w, h, GL_RGBA)

        if self.subdivision.has('pose'):
            for i in range(self.max_players):
                w, h = self.subdivision.get_allocation_size('pose', i)
                self.L[Layers.pose][i].allocate(w, h, GL_RGBA)
            w, h = self.subdivision.get_allocation_size('pose', self.max_players)
            self.L[Layers.dummy_pose][self.dummy_id].allocate(w, h, GL_RGBA)

        # POSE layout: any pose may land in a slot, so every skeleton layer takes the slot's size.
        if self.subdivision.has('slots'):
            w, h = self.subdivision.get_allocation_size('slots', 0)
            for track_id in range(self.max_players + 1):
                self._skeleton(track_id).allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        self.settings.panorama.unbind(layers.PanoramaLayerSettings.focus_radius, self._on_layout_setting)
        RenderSettings.camera_view.unbind(self.settings, self._on_layout_setting)
        RenderSettings.layout.unbind(self.settings, self._on_layout_setting)
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

        # Every row is drawn if the layout holds it, so one pass serves both layouts.

        # POSE layout, top — the three longest-present poses, large, each as its pose-row cell
        # would show it; a pose keeps its slot while it is among them.
        if self.subdivision.has('slots'):
            self._pose_slots = pose_slots(self._pose_slots, self.board.get_frames(int(Stage.LERP)))
            for slot, track_id in enumerate(self._pose_slots):
                if track_id is None:
                    continue
                self._viewport(height, self.subdivision.get_rect('slots', slot))
                self._skeleton(track_id).draw()
                self._draw_data_overlays(track_id)

        # Row 1 — one tracker compositor per camera: the raw frames, with that camera's frame rate,
        # tilt and roll error over them. Absent under PANORAMA.
        if self.subdivision.has('track'):
            for i in range(self.num_cams):
                self._viewport(height, self.subdivision.get_rect('track', i))
                self.L[Layers.tracker][i].draw()
                self.L[Layers.cam_readings][i].draw()

        # Row 2 — the whole rig as one 360° strip, image and tracker data on one vertical scale.
        # Absent under CAMERAS.
        if self.subdivision.has('panoramic'):
            self._viewport(height, self.subdivision.get_rect('panoramic', 0))
            self.L[Layers.cam_panorama][0].draw()

        # Row 3 (POSE layout: the bottom) - WS light: the projection, or the bar's lights while the
        # fixture is in beam mode — the same rule the fixture applies to the same command (the
        # frame's target rpm).
        if self.subdivision.has('ws_light'):
            light_frame = self.board.get_light_frame()
            beam_mode = light_frame is not None and light_frame.motor_command.target_rpm < FIXTURE_PROJECTION_RPM
            self._viewport(height, self.subdivision.get_rect('ws_light', 0))
            self.L[Layers.beam_light_simulation if beam_mode else Layers.light_simulation][0].draw()
            if self.settings.azimuth_overlay:
                self.L[Layers.azimuth_overlay][0].draw()
            if self.settings.pose_figures:
                self.L[Layers.pose_figures][0].draw()

        # Row 4 - pose cutouts with data overlays, one viewport per player, the dummy's last
        if self.subdivision.has('pose'):
            for track_id in range(self.max_players + 1):
                self._viewport(height, self.subdivision.get_rect('pose', track_id))
                self._skeleton(track_id).draw()
                self._draw_data_overlays(track_id)

    def _draw_data_overlays(self, track_id: int) -> None:
        self.L[Layers.data_W][track_id].draw()
        self.L[Layers.data_F][track_id].draw()
        self.L[Layers.data_time][track_id].draw()
        self.L[Layers.data_playhead_W][track_id].draw()
        self.L[Layers.data_playhead_F][track_id].draw()

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        pass
