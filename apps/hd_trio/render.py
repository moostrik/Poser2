"""HD Trio render — layer graph for 3-camera fluid installation."""

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, Shader, Style, clear_color, Texture, MonitorId, WindowSettings
from modules.render.layers import LayerBase
from modules.render import layers as ls, make_subdivision, SubdivisionRow, Subdivision
from modules.utils import Rect, Point2f, HotReloadMethods

from .render_board import RenderBoard
from .settings import Layers, RenderSettings, ShowStage, Stage
from . import render_stages
from .render_stages import STAGES, StageLayer
from .intro_sequence import IntroSequencePlayer, SequenceDataProxy, FixedColorProxy


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_mask,
    Layers.cam_frg,
    Layers.cam_crop,

    Layers.centre_geom,
    Layers.centre_mask,
    Layers.centre_frg,
    Layers.centre_pose,

    Layers.color_mask,
    Layers.flow,
    Layers.fluid,
    # Layers.composite, -> updated via stage system
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.poser,
]

LARGE_LAYERS: list[Layers] = [
    Layers.flow,
    Layers.fluid,
    Layers.composite,
]


class HDTrioRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings) -> None:
        super().__init__(settings.window)
        self.num_players: int = settings.num_players
        self.num_cams: int = settings.num_cams
        self.settings: RenderSettings = settings

        self.board: RenderBoard = board

        self._update_layers: list[Layers] = UPDATE_LAYERS
        self._interface_layers: list[Layers] = INTERFACE_LAYERS
        self._preview_layers: list[Layers] = settings.layer.select.preview
        self._draw_layers: list[Layers] = settings.layer.select.final

        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}
        self._stages: dict[ShowStage, list[StageLayer]] = {stage: [] for stage in STAGES}

        # Set data.b overrides
        settings.data.b.line_width = 6.0
        settings.data.b.line_smooth = 6.0
        settings.data.b.use_history_color = True

        flows: dict[int, ls.FlowLayer] = {}
        cmt: dict[int, Texture] = {}
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.board)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.board)
            cam_frg =       self.L[Layers.cam_frg][i]=      ls.FrgSourceLayer(      i, self.board)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.board)

            cam_comp =      self.L[Layers.poser][i] =       ls.TrackerCompositor(   i, self.board,      cam_image.texture,          settings.preview.tracker,   settings.colors)
            track_comp =    self.L[Layers.tracker][i] =     ls.PoseCompositor(      i, self.board,      cam_image.texture,          settings.preview.poser,     settings.colors)

            centre_gmtry=   self.L[Layers.centre_geom][i] = ls.CentreGeometry(      i, self.board,                                  settings.centre.geometry)
            centre_mask =   self.L[Layers.centre_mask][i] = ls.CentreMaskLayer(     i, centre_gmtry,    cam_mask.texture,           settings.centre.mask)
            cmt[i] = centre_mask.texture
            centre_cam =    self.L[Layers.centre_cam][i] =  ls.CentreCamLayer(      i, centre_gmtry,    cam_image.texture,  cmt[i], settings.centre.cam)
            centre_frg =    self.L[Layers.centre_frg][i] =  ls.CentreFrgLayer(      i, centre_gmtry,    cam_frg.texture,    cmt[i], settings.centre.frg,        settings.colors)
            centre_pose =   self.L[Layers.centre_pose][i] = ls.CentrePoseLayer(     i, centre_gmtry,                                settings.centre.pose,       settings.colors)

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.board,      centre_frg.texture, cmt,    settings.centre.color,      settings.colors)
            flows[i] =      self.L[Layers.flow][i] =        ls.FlowLayer(           i, self.board,      cam_mask,  centre_mask.texture, centre_frg.texture,     settings.flow)
            fluid =         self.L[Layers.fluid][i] =       ls.FluidLayer(          i, self.board,      flows,                      settings.fluid,             settings.colors)

            lut =           self.L[Layers.composite][i] =   ls.CompositeLayer(                          [fluid, ms_mask],           settings.layer.lut)

            self.L[Layers.data_A_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_A_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.a, settings.colors)
            self.L[Layers.data_B_W][i]  = ls.FeatureWindowLayer(i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_B_F][i]  = ls.FeatureFrameLayer( i, self.board, settings.data.b, settings.colors)
            self.L[Layers.data_time][i] = ls.MTimeRenderer(     i, self.board)

            cam_layers = {layer: cam_dict[i] for layer, cam_dict in self.L.items() if i in cam_dict}
            for stage, cls in STAGES.items():
                self._stages[stage].append(cls(i, board, settings, cam_layers))

        # composition
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='track',        columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='preview',      columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(self.subdivision_rows, settings.window.width, settings.window.height, False)

        # Propagate window fps to fluid simulation configs
        def _propagate_fps(fps: int) -> None:
            settings.fluid.fluid_flow.fps = fps
        settings.window.bind(WindowSettings.avg_fps, _propagate_fps)

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)
        self.stage_reloader = HotReloadMethods(StageLayer, True, True)
        self.stage_reloader.add_file_changed_callback(self._rebuild_stages)

        self._prev_stage: ShowStage | None = None
        self._active_stages: list[StageLayer] = []

        # intro sequence overlay
        self._intro_proxy = SequenceDataProxy()
        self._intro_player = IntroSequencePlayer(settings.intro_sequence, self.num_cams)
        self._intro_color_proxy = FixedColorProxy(settings.intro_sequence)
        self._intro_geoms: dict[int, ls.CentreGeometry] = {}
        for i in range(self.num_cams):
            self._intro_geoms[i] = ls.CentreGeometry(i, self._intro_proxy, settings.centre.geometry)  # type: ignore[arg-type]
            self.L[Layers.intro_pose][i] = ls.CentrePoseLayer(i, self._intro_geoms[i], settings.intro_sequence.pose, self._intro_color_proxy)  # type: ignore[arg-type]

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
            self.L[Layers.poser][i].allocate(w, h, GL_RGBA)

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
        seq = self.board.get_sequence()
        stage = ShowStage(seq.stage)
        progress = seq.stage_progress
        if stage != self._prev_stage:
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

        # Intro sequence overlay — tick player during INTRO stages
        _INTRO_STAGES = (ShowStage.INTRO_IN, ShowStage.INTRO, ShowStage.INTRO_OUT)
        if stage in _INTRO_STAGES:
            if not self._intro_player.active:
                self._intro_player.start()
            frames = self._intro_player.update()
            self._intro_proxy.update(frames)
            for i in range(self.num_cams):
                self._intro_geoms[i].update()
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
            camera_id: int = self.settings.window.secondary_list.index(MonitorId(monitor_id))
        except ValueError:
            return

        if camera_id >= self.num_cams:
            return

        Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ADD)

        for layer_type in self._draw_layers:
            if camera_id in self.L[layer_type]:
                self.L[layer_type][camera_id].draw()

        Style.pop_style()
