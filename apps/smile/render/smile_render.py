import time

import glfw
import numpy as np
from OpenGL.GL import GL_RGBA32F, glViewport

from .screen_grab import ScreenGrab
from .. import sphere
from ..board import OnlineCell, RenderBoard
from modules.gl import Fbo, Shader, BlendState, Texture, clear_color
from modules.render import LayerBase
from modules.utils import fit
from modules.window import RenderBase
from ..player import WallsSettings
from .inputs import Inputs
from .layers import (BloomLayer, CameraLayer, CreditLayer, CreditView, DiffusionLayer, FlashLayer, RevealLayer,
                     StatusLayer, WallCompositor, caption_shown, copy, fade_step, new_line_due)
from ..sequence import GrinPlayer, SequenceType, Sequencer
from .settings import RenderSettings, WindowsSettings
from .stages import STAGES, Layers, Stage

# 32 bit: the trail (0.995) adds steps too small for half floats
COMPOSITION_FORMAT = GL_RGBA32F


class SmileRender(RenderBase):
    """The installation display. Runs the sequencer every frame (it is the app clock), then lets the
    stage of the current sequence state render the layers, and fits its output in the window.
    The grin frames come from the grin player, which is asked every frame for the one that is due.
    The two walls of the sphere (`walls`, `WallCompositor`) are composited after it, each shown in the
    window of its own: window 1 is wall 0, screens 1-24, window 2 wall 1;
    the online cell of one of them shows the output without the QR code during an online turn.
    What the app's threads make it reads from the board at its tick (`RenderBoard`): the camera's frame, the faces,
    the capture, the generator's images, the status line and warm-up, and the online cell.
    Keys: Enter starts a sequence; F fullscreen, Esc quit (the window manager)."""

    def __init__(self, board: RenderBoard, settings: RenderSettings, windows: WindowsSettings, walls_settings: WallsSettings,
                 sequencer: Sequencer, grin_player: GrinPlayer, width: int, height: int) -> None:
        super().__init__(windows)
        self.board: RenderBoard = board
        self.windows: WindowsSettings = windows
        self.settings: RenderSettings = settings
        self.sequencer: Sequencer = sequencer
        self.grin_player: GrinPlayer = grin_player
        self._plays: int = grin_player.plays
        self.width: int = width
        self.height: int = height

        self.inputs: Inputs = Inputs(board)
        camera = CameraLayer(self.inputs.camera, board, settings)
        self.layers: Layers = Layers(
            camera=camera,
            flash=FlashLayer(camera, self.inputs.white),
            reveal=RevealLayer(self.inputs.capture_depth, self.inputs.white),
            diffusion=DiffusionLayer(self.inputs.diffusion),
            bloom=BloomLayer(self.inputs.diffusion_depth),
        )
        # the credit: the QR code always, the caption over the grin and the result
        self._credit: CreditLayer = CreditLayer(settings)
        self._credit_alpha: float = 0.0
        self._credit_time: float = time.perf_counter()
        self._credit_sequence: SequenceType = SequenceType.NONE
        # the operator's status line, drawn over the output while there is something to say
        self._status: StatusLayer = StatusLayer()
        self._all_layers: list[LayerBase] = list(vars(self.layers).values()) + [self._credit, self._status]

        # one instance per stage class, shared by the sequence types it covers
        stages: dict[type[Stage], Stage] = {cls: cls(self.inputs, self.layers, settings) for cls in set(STAGES.values())}
        self._stages: dict[SequenceType, Stage] = {seq: stages[cls] for seq, cls in STAGES.items()}
        self._stage: Stage | None = None

        self._output: Texture = self.layers.camera.texture
        self._freeze_fbo: Fbo = Fbo()
        self._frozen: bool = False
        # the screen for the phones of SMILE Online: switched on and read by modules/online
        self.grab: ScreenGrab = ScreenGrab()
        self._cell_view: Texture = self._output      # the output as the online cell shows it: no QR code, no status line
        # the walls of the sphere; which cell of which one is SMILE Online's is on the board
        self.walls: list[WallCompositor] = [WallCompositor(settings, walls_settings, [str(i) for i in range(sphere.CELLS)])
                                            for _ in range(sphere.WALLS)]

        self.window_manager.input.add_key_callback(self._on_key)

    # MAIN THREAD
    def allocate(self) -> None:
        for layer in self._all_layers:
            layer.allocate(self.width, self.height, COMPOSITION_FORMAT)
        self._freeze_fbo.allocate(self.width, self.height, COMPOSITION_FORMAT)
        for wall in self.walls:
            wall.allocate()
        Shader.enable_hot_reload()

    def deallocate(self) -> None:
        for layer in self._all_layers:
            layer.deallocate()
        self._freeze_fbo.deallocate()
        self.grab.deallocate()
        self.inputs.deallocate()
        for wall in self.walls:
            wall.deallocate()

    def update(self) -> None:
        # the frame's tick: the one clock read of the frame, handed to all that steps with it
        now: float = time.perf_counter()
        self._play_grin(now)
        # sequencer first: images handed over with a state change (the depth with CONTROL_0)
        # are then uploaded before the stage uses them
        message = self.sequencer.update(now)
        self._notify_update()
        self.inputs.update()

        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)
        for layer in self._all_layers:
            layer.update()

        stage: Stage = self._stages[message.sequence_type]
        if stage is not self._stage:
            self._stage = stage
            stage.enter()
        stage.update(message)

        self._output = stage.output
        online: OnlineCell | None = self.board.get_online_cell()
        self._render_credit(now, message.sequence_type, self.grab.due(now), online)
        status: str | None = self.board.get_status()
        if status:
            self._status.render(self._output, status)
            self._output = self._status.texture
        if self.settings.freeze:
            if not self._frozen:
                copy(self._output, self._freeze_fbo)
            self._output = self._freeze_fbo
        self._frozen = self.settings.freeze
        self._update_walls(online)

    def _render_credit(self, now: float, sequence_type: SequenceType, grab: bool, online: OnlineCell | None) -> None:
        """The credit over the output at the tick `now`, with two copies for SMILE Online: the phone's
        (`CreditLayer.clean`), rendered when `grab` takes it (the turn's last frame, `grab.clean`, without the caption:
        the phone fades it out itself and keeps that frame), and the online cell's (`CreditLayer.cell`), rendered while
        that cell shows it."""
        if new_line_due(sequence_type, self._credit_sequence):
            self._credit.new_visitor()
        self._credit_sequence = sequence_type
        shown: bool = caption_shown(sequence_type, self.settings.credit_camera)
        self._credit_alpha = fade_step(self._credit_alpha, shown, now - self._credit_time)
        self._credit_time = now
        if self.settings.credit:
            if grab:
                self._credit.render(self._output, 0.0 if self.grab.clean else self._credit_alpha, CreditView.PHONE)
                self.grab.grab(self._credit.clean)
            if online is not None and online.mode == 'screen':
                self._credit.render(self._output, self._credit_alpha, CreditView.CELL)
            self._cell_view = self._credit.cell
            self._credit.render(self._output, self._credit_alpha)
            self._output = self._credit.texture
        else:
            self._cell_view = self._output
            if grab:
                self.grab.grab(self._output)

    def _update_walls(self, online: OnlineCell | None) -> None:
        """The walls after the main screen, so the online cell shows this frame; only the walls that have a
        window shown are composited."""
        windows = self.windows.windows
        for index, wall in enumerate(self.walls):
            if online is not None and online.wall == index:
                wall.set_online(str(online.cell), online.mode, self._cell_view if online.mode == 'screen' else None, online.page)
            else:
                wall.set_online(None)
            if index + 1 < len(windows) and windows[index + 1].enabled:
                wall.update()

    def _play_grin(self, now: float) -> None:
        """Show the grin frame that is due at the tick `now`. Its first frame arrives during DIFFUSION and blends in as
        a generation step; DIFF_LOOP starts when the grin player starts playing."""
        frame: np.ndarray | None = self.grin_player.update(now)
        if frame is not None:
            self.inputs.diffusion.set_image(frame)
        started: bool = self.grin_player.plays != self._plays
        if started:
            self._plays = self.grin_player.plays
            self.sequencer.set_sequence(SequenceType.DIFF_LOOP)
        if started or (frame is not None and self.grin_player.playing):
            self.sequencer.set_diffusion_progress(self.grin_player.progress)

    def draw(self, window: int, width: int, height: int) -> None:
        """Window 0 the output, windows 1-2 the walls."""
        clear_color(0, 0, 0, 1)
        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)
        if window != 0:
            if window - 1 < len(self.walls):
                self._draw_fit(self.walls[window - 1].texture, width, height)
            return
        self._draw_fit(self._output, width, height)
        if self.settings.show_camera and self.inputs.camera.texture.allocated:
            self._draw_fit(self.inputs.camera.texture, width, height)

    def on_window_resize(self, window: int, width: int, height: int) -> None:
        pass

    @staticmethod
    def _draw_fit(texture: Texture, width: int, height: int) -> None:
        if not texture.allocated:
            return
        x, y, w, h = fit(texture.width, texture.height, width, height)
        glViewport(int(x), int(y), int(w), int(h))
        texture.draw()

    def _on_key(self, window: int, key: int, action: int, mods: int) -> None:
        if action == glfw.PRESS and key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self.sequencer.start_sequence()
