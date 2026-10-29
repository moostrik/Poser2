"""HD Trio render — layer graph for 3-camera fluid installation."""

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, Shader, Style, clear_color, Texture, MonitorId, WindowSettings
from modules.camera import RigCameraSettings, CameraCheckSettings
from modules.render import layers as ls, make_subdivision, SubdivisionRow, Subdivision
from modules.utils import Rect, Point2f, HotReloadMethods

from .render_board import RenderBoard
from .settings import Layers, RenderSettings, ShowStage, Stage, IntroSequenceSettings
from . import render_stages
from .render_stages import STAGES, StageLayer
from .intro_sequence import IntroSequencePlayer, SequenceDataProxy, FixedColorProxy
from pathlib import Path


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_mask,
    Layers.cam_foreground,
    Layers.cam_crop,

    Layers.center_geometry,
    Layers.center_mask,
    Layers.center_foreground,
    Layers.center_pose,

    Layers.color_mask,
    Layers.flow,
    Layers.fluid,
    # Layers.composite, -> updated via stage system
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.tracker,
    Layers.cam_readings,
]

LARGE_LAYERS: list[Layers] = [
    Layers.flow,
    Layers.fluid,
    Layers.composite,
]


class HDTrioRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings,
                 cameras: list[RigCameraSettings], camera_check: CameraCheckSettings, data_path: str = "") -> None:
        super().__init__(settings.window)
        self.max_players: int = settings.max_players
        self.num_cams: int = settings.num_cams
        self.settings: RenderSettings = settings

        self.board: RenderBoard = board

        self._update_layers: list[Layers] = UPDATE_LAYERS
        self._interface_layers: list[Layers] = INTERFACE_LAYERS
        self._preview_layers: list[Layers] = settings.layer.select.preview
        self._draw_layers: list[Layers] = settings.layer.select.final

        self.L: dict[Layers, dict[int, ls.LayerBase]] = {layer: {} for layer in Layers}
        self._stages: dict[ShowStage, list[StageLayer]] = {stage: [] for stage in STAGES}

        # Set data.b overrides
        settings.data.b.line_width = 6.0
        settings.data.b.line_smooth = 6.0
        settings.data.b.use_history_color = True

        # intro sequence overlay — shared objects must exist before the per-camera loop
        self._intro_proxy = SequenceDataProxy()
        self._intro_player = IntroSequencePlayer(settings.intro_sequence, self.num_cams, data_path=data_path)
        self._intro_color_proxy = FixedColorProxy(settings.intro_sequence)
        self._intro_geometries: dict[int, ls.CenterGeometry] = {}

        flows: dict[int, ls.FlowLayer] = {}
        cmt: dict[int, Texture] = {}
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.board)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.board)
            cam_foreground = self.L[Layers.cam_foreground][i] = ls.ForegroundSourceLayer(i, self.board)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.board)

            tracker_compositor = self.L[Layers.tracker][i] = ls.TrackerCompositor(  i, self.board,      cam_image.texture,          settings.preview.tracker,   settings.colors)
            pose_compositor = self.L[Layers.pose][i] =      ls.PoseCompositor(      i, self.board,      cam_image.texture,          settings.preview.pose,      settings.colors)
            self.L[Layers.cam_readings][i] =                 ls.CameraReadingsLayer( cameras[i], camera_check)

            center_geometry = self.L[Layers.center_geometry][i] = ls.CenterGeometry( i, self.board,                                  settings.center.geometry)
            center_mask =   self.L[Layers.center_mask][i] = ls.CenterMaskLayer(     i, center_geometry, cam_mask.texture,           settings.center.mask)
            cmt[i] = center_mask.texture
            center_cam =    self.L[Layers.center_cam][i] =  ls.CenterCamLayer(      i, center_geometry, cam_image.texture,  cmt[i], settings.center.cam)
            center_foreground = self.L[Layers.center_foreground][i] = ls.CenterForegroundLayer(i, center_geometry, cam_foreground.texture, cmt[i], settings.center.foreground, settings.colors)
            center_pose =   self.L[Layers.center_pose][i] = ls.CenterPoseLayer(     i, center_geometry,                             settings.center.pose,       settings.colors)

            intro_geometry = self._intro_geometries[i] =     ls.CenterGeometry(      i, self._intro_proxy,                          settings.center.geometry)  # type: ignore[arg-type]
            self.L[Layers.intro_pose][i] =                   ls.CenterPoseLayer(     i, intro_geometry,                             settings.intro_sequence.pose, self._intro_color_proxy)  # type: ignore[arg-type]

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.board,      center_foreground.texture, cmt, settings.center.color,  settings.colors)
            flows[i] =      self.L[Layers.flow][i] =        ls.FlowLayer(           i, self.board,      cam_mask,  center_mask.texture, center_foreground.texture, settings.flow)
            fluid =         self.L[Layers.fluid][i] =       ls.FluidLayer(          i, self.board,      flows,                      settings.fluid,             settings.colors)

            lut =           self.L[Layers.composite][i] =   ls.CompositeLayer(                          [fluid, ms_mask],           settings.layer.composite)

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_time][i] = ls.MotionTimeRenderer(i, self.board)

            cam_layers = {layer: cam_dict[i] for layer, cam_dict in self.L.items() if i in cam_dict}
            for stage, cls in STAGES.items():
                self._stages[stage].append(cls(i, board, settings, cam_layers))

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='track',        columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='preview',      columns=self.max_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.window.width, settings.window.height, False)

        # Propagate window fps to fluid simulation settings
        def _propagate_fps(fps: int) -> None:
            settings.fluid.fluid_flow.fps = fps
        settings.window.bind(WindowSettings.avg_fps, _propagate_fps)

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)
        self.stage_reloader = HotReloadMethods(StageLayer, True, True)
        self.stage_reloader.add_file_changed_callback(self._rebuild_stages)

        self._prev_stage: ShowStage | None = None
        self._active_stages: list[StageLayer] = []

    def _rebuild_stages(self) -> None:
        """Re-instantiate stage objects after hot-reload of stages.py."""
        self._stages = {
            stage: [
                cls(i, self.board, self.settings, {layer: cam_dict[i] for layer, cam_dict in self.L.items() if i in cam_dict})
                for i in range(self.num_cams)
            ]
            for stage, cls in render_stages.STAGES.items()
        }
        self._prev_stage = None
        self._active_stages = []

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()

    def allocate(self) -> None:
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                if layer_type in LARGE_LAYERS:
                    layer.allocate(1080 * 2, 1920 * 2, GL_RGBA16F)
                else:
                    layer.allocate(1080, 1920, GL_RGBA16F)
        self.allocate_window_renders()
        Shader.enable_hot_reload()

    def allocate_window_renders(self) -> None:
        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size('track', i)
            for layer_type in INTERFACE_LAYERS:
                self.L[layer_type][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()

    def update(self) -> None:
        self._notify_update()

        self._update_layers = UPDATE_LAYERS
        self._draw_layers = self.settings.layer.select.final
        self._preview_layers = self.settings.layer.select.preview

        # Stage transitions — enter/exit before layer updates so settings writes take effect
        seq = self.board.get_sequencer_state()
        stage = ShowStage(seq.stage)
        progress = seq.stage_progress
        prev_stage = self._prev_stage
        if stage != prev_stage:
            for s in self._active_stages:
                s.exit()
            self._active_stages = self._stages.get(stage, [])
            for s in self._active_stages:
                s.enter()
            self._prev_stage = stage

        # Layer updates
        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        seen: set[Layers] = set()
        for layer_type in self._update_layers + self._interface_layers + self._draw_layers + self._preview_layers:
            if layer_type not in seen:
                seen.add(layer_type)
                for layer in self.L[layer_type].values():
                    layer.update()

        # Intro sequence overlay — tick player while the white example is on screen
        _INTRO_STAGES = (ShowStage.WHITE_POSE, ShowStage.FLUID, ShowStage.PRACTICE, ShowStage.ENJOY_IN)
        if stage in _INTRO_STAGES:
            if prev_stage not in _INTRO_STAGES:  # first frame of intro sequence
                self._intro_player.start()
            frames = self._intro_player.update()
            self._intro_proxy.set_frames(frames)
            for i in range(self.num_cams):
                self._intro_geometries[i].update()
                self.L[Layers.intro_pose][i].update()
        else:
            if self._intro_player.active:
                self._intro_player.stop()
            self._intro_proxy.clear()

        # Stage composition — after layer updates so textures are fresh
        for s in self._active_stages:
            s.update(progress)

    def draw_main(self, width: int, height: int) -> None:
        clear_color()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        for i in range(self.num_cams):
            track_rect: Rect = self.subdivision.get_rect('track', i)
            glViewport(int(track_rect.x), int(height - track_rect.y - track_rect.height), int(track_rect.width), int(track_rect.height))
            for layer_type in self._interface_layers:
                self.L[layer_type][i].draw()

        for i in range(self.num_cams):
            preview_rect: Rect = self.subdivision.get_rect('preview', i)
            glViewport(int(preview_rect.x), int(height - preview_rect.y - preview_rect.height), int(preview_rect.width), int(preview_rect.height))
            for layer_type in self._preview_layers:
                self.L[layer_type][i].draw()

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        clear_color()

        try:
            cam_id: int = self.settings.window.secondary_list.index(MonitorId(monitor_id))
        except ValueError:
            return

        if cam_id >= self.num_cams:
            return

        Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        for layer_type in self._draw_layers:
            if cam_id in self.L[layer_type]:
                self.L[layer_type][cam_id].draw()

        Style.pop_style()
