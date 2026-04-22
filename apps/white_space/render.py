"""White Space render — layer graph for 3-camera panoramic light installation."""

from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, clear_color, Style
from modules.render.layers import LayerBase
from modules.render.layers import ImageSourceLayer, MaskSourceLayer, CropSourceLayer
from modules.render.layers import TrackerCompositor, PoseCompositor
from modules.render.layers import FeatureWindowLayer, FeatureFrameLayer, MTimeRenderer
from modules.render.layers.WS.TrackerPanoramicLayer import TrackerPanoramicLayer
from modules.render.layers.WS.WSLightLayer import WSLightLayer
from modules.render.layers.WS.WSLinesLayer import WSLinesLayer
from modules.utils.PointsAndRects import Rect, Point2f
from modules.render.composition_subdivider import make_subdivision, SubdivisionRow, Subdivision
from modules.utils.HotReloadMethods import HotReloadMethods
from modules.gl.WindowManager import WindowSettings

from .render_board import RenderBoard
from .settings import Layers, RenderSettings, ShowStage


UPDATE_LAYERS: list[Layers] = [
    Layers.cam_image,
    Layers.cam_mask,
    Layers.cam_crop,
    Layers.ws_light,
    Layers.ws_lines,
    Layers.ws_tracker,
    Layers.data_W,
    Layers.data_F,
    Layers.data_time,
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.tracker,
    Layers.poser,
]


class WhiteSpaceRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings) -> None:
        super().__init__(settings.window)
        self.num_players: int = settings.num_players
        self.num_cams: int = settings.num_cams
        self.settings: RenderSettings = settings
        self.board: RenderBoard = board

        self._update_layers: list[Layers] = UPDATE_LAYERS
        self._interface_layers: list[Layers] = INTERFACE_LAYERS
        self._preview_layers: list[Layers] = settings.layers_select.preview
        self._draw_layers: list[Layers] = settings.layers_select.final

        self.L: dict[Layers, dict[int, LayerBase]] = {layer: {} for layer in Layers}

        # WS panoramic layers — shared (not per-cam)
        self.L[Layers.ws_tracker][0] = TrackerPanoramicLayer(board, self.num_cams)
        self.L[Layers.ws_light][0]   = WSLightLayer(board)
        self.L[Layers.ws_lines][0]   = WSLinesLayer(board)

        for i in range(self.num_cams):
            self.L[Layers.cam_image][i] = ImageSourceLayer(i, board)
            self.L[Layers.cam_mask][i]  = MaskSourceLayer(i, board)
            self.L[Layers.cam_crop][i]  = CropSourceLayer(i, board)

            self.L[Layers.tracker][i] = TrackerCompositor(
                i, board,
                self.L[Layers.cam_image][i].texture,
                settings.preview.tracker,
                settings.colors,
            )
            self.L[Layers.poser][i] = PoseCompositor(
                i, board,
                self.L[Layers.cam_image][i].texture,
                settings.preview.poser,
                settings.colors,
            )

            self.L[Layers.data_W][i]    = FeatureWindowLayer(i, board, settings.data, settings.colors)
            self.L[Layers.data_F][i]    = FeatureFrameLayer( i, board, settings.data, settings.colors)
            self.L[Layers.data_time][i] = MTimeRenderer(     i, board)

        # Subdivision: top row = camera strips (square aspect), bottom row = WS output
        self.subdivision_rows: list[SubdivisionRow] = [
            SubdivisionRow(name='track',   columns=self.num_cams,    rows=1, src_aspect_ratio=1.0,  padding=Point2f(1.0, 1.0)),
            SubdivisionRow(name='preview', columns=self.num_players, rows=1, src_aspect_ratio=9/16, padding=Point2f(1.0, 1.0)),
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
            if i in self.L[Layers.tracker]:
                w, h = self.subdivision.get_allocation_size('track', i)
                self.L[Layers.tracker][i].allocate(w, h, GL_RGBA)

    def deallocate(self) -> None:
        for cam_dict in self.L.values():
            for layer in cam_dict.values():
                layer.deallocate()

    def update(self) -> None:
        self._notify_update()

        self._draw_layers    = self.settings.layers_select.final
        self._preview_layers = self.settings.layers_select.preview

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        seen: set[Layers] = set()
        for layer_type in self._update_layers + self._interface_layers + self._draw_layers + self._preview_layers:
            if layer_type not in seen:
                seen.add(layer_type)
                for layer in self.L[layer_type].values():
                    layer.update()

    def draw_main(self, width: int, height: int) -> None:
        clear_color()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.ALPHA)

        for i in range(self.num_cams):
            track_rect: Rect = self.subdivision.get_rect('track', i)
            glViewport(
                int(track_rect.x),
                int(height - track_rect.y - track_rect.height),
                int(track_rect.width),
                int(track_rect.height),
            )
            for layer_type in self._interface_layers:
                if i in self.L[layer_type]:
                    self.L[layer_type][i].draw()

        for i in range(self.num_players):
            preview_rect: Rect = self.subdivision.get_rect('preview', i)
            glViewport(
                int(preview_rect.x),
                int(height - preview_rect.y - preview_rect.height),
                int(preview_rect.width),
                int(preview_rect.height),
            )
            for layer_type in self._preview_layers:
                if i in self.L[layer_type]:
                    self.L[layer_type][i].draw()
                elif 0 in self.L[layer_type]:
                    # Shared layers (ws_light, ws_lines, ws_tracker) stored at index 0
                    self.L[layer_type][0].draw()

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        pass
