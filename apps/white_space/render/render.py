"""White Space render — layer graph for 3-camera panoramic light installation."""

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, clear_color, Style
from modules.render.layers import LayerBase
from modules.render.layers import ImageSourceLayer, MaskSourceLayer, CropSourceLayer
from modules.render.layers import TrackerCompositor, PoseCompositor
from modules.render.layers import FeatureWindowLayer, FeatureFrameLayer, MTimeRenderer
from modules.render.layers.generic.PanoramicTrackerLayer import PanoramicTrackerLayer
from apps.white_space.render.layers.light_simulation_layer import LightSimulationLayer
from modules.utils.PointsAndRects import Rect, Point2f
from modules.render.composition_subdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.utils.HotReloadMethods import HotReloadMethods

from .board import RenderBoard
from ..settings import Layers, RenderSettings


class WhiteSpaceRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings) -> None:
        super().__init__(settings.window)
        self.num_players: int = settings.num_players
        self.num_cams: int = settings.num_cams
        self.settings: RenderSettings = settings
        self.board: RenderBoard = board

        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        # Row 1 — per-camera: source layers + tracker compositor
        for i in range(self.num_cams):
            self.L[Layers.cam_image][i] = ImageSourceLayer(i, board)
            self.L[Layers.cam_mask][i]  = MaskSourceLayer(i, board)
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
            self.L[Layers.data_W][i]    = FeatureWindowLayer(i, board, settings.data, settings.colors)
            self.L[Layers.data_F][i]    = FeatureFrameLayer( i, board, settings.data, settings.colors)
            self.L[Layers.data_time][i] = MTimeRenderer(     i, board)

        # Rows 2–4 — shared panoramic layers; constructed after cam layers so textures are ready
        self.L[Layers.ws_tracker][0] = PanoramicTrackerLayer(board, self.num_cams, settings.colors)
        self.L[Layers.ws_light][0]   = LightSimulationLayer(board)

        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='track',      columns=self.num_cams,    rows=1, src_aspect_ratio=16/9, padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='panoramic',  columns=1,                rows=1, src_aspect_ratio=10.0, padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name='ws_light',   columns=1,                rows=1, src_aspect_ratio=3.0, padding=Point2f(0.0, 1.0)),
            SubdivisionRow(name='pose',       columns=self.num_players, rows=1, src_aspect_ratio=0.75, padding=Point2f(1.0, 1.0)),
        ]
        self.subdivision: Subdivision = make_subdivision(
            self.subdivision_rows, settings.window.width, settings.window.height, False
        )

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def on_main_window_resize(self, width: int, height: int) -> None:
        self.subdivision = make_subdivision(self.subdivision_rows, width, height, True)
        self.allocate_window_renders()

    def allocate(self) -> None:
        for layer_type, cam_dict in self.L.items():
            for layer in cam_dict.values():
                layer.allocate(1080, 1920, GL_RGBA16F)
        self.allocate_window_renders()

    def allocate_window_renders(self) -> None:
        for i in range(self.num_cams):
            w, h = self.subdivision.get_allocation_size('track', i)
            self.L[Layers.tracker][i].allocate(w, h, GL_RGBA)

        w, h = self.subdivision.get_allocation_size('panoramic', 0)
        self.L[Layers.ws_tracker][0].allocate(w, h, GL_RGBA)

        for i in range(self.num_players):
            w, h = self.subdivision.get_allocation_size('pose', i)
            self.L[Layers.poser][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()

    def update(self) -> None:
        self._notify_update()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        for cam_dict in self.L.values():
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
        clear_color()
        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        # Row 1 — tracker compositor, one viewport per camera
        for i in range(self.num_cams):
            self._viewport(height, self.subdivision.get_rect('track', i))
            self.L[Layers.tracker][i].draw()

        # Row 2 — panoramic tracker standin (single wide viewport)
        self._viewport(height, self.subdivision.get_rect('panoramic', 0))
        self.L[Layers.ws_tracker][0].draw()

        # Row 3 - WS light strip
        self._viewport(height, self.subdivision.get_rect('ws_light', 0))
        self.L[Layers.ws_light][0].draw()

        # Row 4 - pose cutouts with data overlays, one viewport per player
        for i in range(self.num_players):
            self._viewport(height, self.subdivision.get_rect('pose', i))
            self.L[Layers.poser][i].draw()
            self.L[Layers.data_W][i].draw()
            self.L[Layers.data_F][i].draw()
            self.L[Layers.data_time][i].draw()

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        pass
