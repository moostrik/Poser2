"""HD Trio render — layer graph for 3-camera fluid installation."""

from collections.abc import Sequence
from typing import cast

import numpy as np
from OpenGL.GL import GL_RGBA16F, GL_RGBA, glViewport

from modules.gl import RenderBase, Shader, Style, clear_color, Texture, MonitorId
from modules.board import HasFrames
from modules.camera import RigCameraSettings, CameraCheckSettings
from modules.flow import hold_tick
from modules.pose import Frame, Similarity, AngleMotion
from modules.render import ColorSettings, layers as ls, shaders, make_subdivision, SubdivisionRow, Subdivision
from modules.utils import Rect, Point2f, Color, HotReloadMethods

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
    Layers.motion,
    Layers.bridge,
    Layers.fluid,
    Layers.monitor_a,
    Layers.monitor_b,
    # Layers.composite, -> updated via stage system
]

INTERFACE_LAYERS: list[Layers] = [
    Layers.tracker,
    Layers.cam_readings,
]

LARGE_LAYERS: list[Layers] = [
    Layers.motion,
    Layers.bridge,
    Layers.fluid,
    Layers.monitor_a,
    Layers.monitor_b,
    Layers.composite,
]

FADE_TIME_STILL: float = 18.0   # s: the density's fade while the player stands still
FADE_TIME_RANGE: float = 14.0   # s: how much faster it fades at full motion
BORDER: tuple[float, float, float, float] = (0.4, 0.4, 0.4, 1.0)    # the border around each panel in the main window, so a black layer shows
BORDER_WIDTH: float = 2.0       # px


class HDTrioRender(RenderBase):
    def __init__(self, board: RenderBoard, settings: RenderSettings,
                 cameras: Sequence[RigCameraSettings], camera_check: CameraCheckSettings, data_path: str = "") -> None:
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

        cmt: dict[int, Texture] = {}
        for i in range(self.num_cams):
            cam_image =     self.L[Layers.cam_image][i] =   ls.ImageSourceLayer(    i, self.board)
            cam_mask =      self.L[Layers.cam_mask][i] =    ls.MaskSourceLayer(     i, self.board)
            cam_foreground = self.L[Layers.cam_foreground][i] = ls.ForegroundSourceLayer(i, self.board)
            cam_crop =      self.L[Layers.cam_crop][i] =    ls.CropSourceLayer(     i, self.board)

            tracker_compositor = self.L[Layers.tracker][i] = ls.TrackerCompositor(  i, self.board,      cam_image.texture,          settings.preview.tracker,   settings.colors)
            pose_compositor = self.L[Layers.pose][i] =      ls.PoseCompositor(      i, self.board,      cam_image.texture, cam_crop, settings.preview.pose,      settings.colors)
            self.L[Layers.cam_readings][i] =                 ls.CameraReadingsLayer( cameras[i], camera_check)

            center_geometry = self.L[Layers.center_geometry][i] = ls.CenterGeometry( i, self.board,      cam_image.texture,          settings.center.geometry)
            center_mask =   self.L[Layers.center_mask][i] = ls.CenterMaskLayer(     i, center_geometry, cam_mask,               ls.SourceSpace.CROP, settings.center.mask)
            cmt[i] = center_mask.texture
            center_cam =    self.L[Layers.center_cam][i] =  ls.CenterCamLayer(      i, center_geometry, cam_image,          cmt[i], settings.center.cam)
            center_foreground = self.L[Layers.center_foreground][i] = ls.CenterForegroundLayer(i, center_geometry, cam_foreground, cmt[i], ls.SourceSpace.CROP, False, settings.center.foreground, settings.colors)
            center_pose =   self.L[Layers.center_pose][i] = ls.CenterPoseLayer(     i, center_geometry,                             settings.center.pose,       settings.colors)

            intro_geometry = self._intro_geometries[i] =     ls.CenterGeometry(      i, cast(HasFrames, self._intro_proxy), cam_image.texture, settings.center.geometry)
            self.L[Layers.intro_pose][i] =                   ls.CenterPoseLayer(     i, intro_geometry,                             settings.intro_sequence.pose, cast(ColorSettings, self._intro_color_proxy))

            ms_mask =       self.L[Layers.color_mask][i] =  ls.MSColorMaskLayer(    i, self.board,      center_foreground.texture, cmt, settings.center.color,  settings.colors)
            self.L[Layers.motion][i] =                      ls.Motion2DLayer(       settings.motion)
            self.L[Layers.bridge][i] =                      ls.Bridge2DLayer(       settings.bridge)
            fluid =         self.L[Layers.fluid][i] =       ls.Fluid2DLayer(        settings.fluid)
            self.L[Layers.monitor_a][i] =                   ls.MonitorLayer(        settings.monitor_a)
            self.L[Layers.monitor_b][i] =                   ls.MonitorLayer(        settings.monitor_b)

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

        # The length of each tick, the flow's dt, held to twice the mean tick (`hold_tick`)
        self._last_tick: float = 0.0
        self._tick_mean: float | None = None
        self._dt: float = 1.0 / 60.0
        self.add_update_callback(self._measure_tick)

        # The border around each panel in the main window
        self._border: shaders.DrawRectangleOutline = shaders.DrawRectangleOutline()

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
        self._border.allocate()
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
        self._border.deallocate()

    def _measure_tick(self, now: float) -> None:
        if self._last_tick > 0.0:
            self._dt, self._tick_mean = hold_tick(now - self._last_tick, self._tick_mean)
        self._last_tick = now

    # ========== The flow's wiring ==========

    def _wire_flow(self) -> None:
        """Build each stage's input: a motion input per camera on its centered mask, with the capture
        time of the frame the mask was made from; a bridge input with the centered foreground as the
        picture behind the density and the temperature; a fluid input per screen with one injection
        per camera's bridge in the camera's track color, the own camera's at the player's motion,
        another's at the cross-camera similarity times `similarity_scale`."""
        dt: float = self._dt
        colors: list[Color] = self.settings.colors.track_colors
        bridges: dict[int, ls.Bridge2DLayer] = {i: cast(ls.Bridge2DLayer, layer) for i, layer in self.L[Layers.bridge].items()}

        for i, _motion in self.L[Layers.motion].items():
            motion = cast(ls.Motion2DLayer, _motion)
            if i not in bridges:
                continue
            mask_view: ls.LayerBase = self.L[Layers.center_mask][i]
            motion.set_input(ls.MotionInput(mask_view.texture, mask_view.timestamp if isinstance(mask_view, ls.HasCaptureTime) else None, dt))
            bridges[i].set_input(ls.BridgeInput(motion.output.velocity, dt, self.L[Layers.center_foreground][i].texture))

        for i, _fluid in self.L[Layers.fluid].items():
            fluid = cast(ls.Fluid2DLayer, _fluid)
            pose: Frame | None = self.board.get_frame(self.settings.stage, i)
            motion_value: float = float(pose[AngleMotion].value) if pose is not None and AngleMotion in pose else 0.0
            similarities: np.ndarray = (np.nan_to_num(pose[Similarity].values) if pose is not None and Similarity in pose
                                        else np.zeros(self.max_players)) * self.settings.similarity_scale
            # More motion, a faster fade
            self.settings.fluid.density.fade_time = FADE_TIME_STILL - motion_value ** 2 * FADE_TIME_RANGE
            injections: list[ls.Injection] = []
            for j, bridge in bridges.items():
                output: ls.BridgeOutput = bridge.output
                weight: float = motion_value if j == i else (float(similarities[j]) if j < len(similarities) else 0.0)
                color: Color = colors[j % len(colors)]
                injections.append(ls.Injection(output.velocity, output.density, output.temperature, weight=weight, color=(color.r, color.g, color.b)))
            fluid.set_input(ls.FluidInput(dt, injections))

        for i, _motion in self.L[Layers.motion].items():
            if i not in bridges or i not in self.L[Layers.fluid]:
                continue
            ports: dict[ls.FlowStage, dict[str, ls.Port]] = {
                ls.FlowStage.MOTION: cast(ls.Motion2DLayer, _motion).ports,
                ls.FlowStage.BRIDGE: bridges[i].ports,
                ls.FlowStage.FLUID: cast(ls.Fluid2DLayer, self.L[Layers.fluid][i]).ports,
            }
            for key in (Layers.monitor_a, Layers.monitor_b):
                if i in self.L[key]:
                    cast(ls.MonitorLayer, self.L[key][i]).set_input(ls.MonitorInput(ports))

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

        self._wire_flow()

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
            self._draw_border(track_rect.width, track_rect.height)

        for i in range(self.num_cams):
            preview_rect: Rect = self.subdivision.get_rect('preview', i)
            glViewport(int(preview_rect.x), int(height - preview_rect.y - preview_rect.height), int(preview_rect.width), int(preview_rect.height))
            for layer_type in self._preview_layers:
                self.L[layer_type][i].draw()
            self._draw_border(preview_rect.width, preview_rect.height)

    def _draw_border(self, width: float, height: float) -> None:
        """A border around the current viewport, `width` by `height` pixels."""
        if width <= 0 or height <= 0:
            return
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        self._border.use(0.0, 0.0, 1.0, 1.0, *BORDER, BORDER_WIDTH / width, BORDER_WIDTH / height)
        Style.pop_style()

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
