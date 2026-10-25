import time

import glfw
import numpy as np
from OpenGL.GL import GL_RGBA32F, glViewport  # type: ignore

from modules.render.GrinPlayer import GrinPlayer
from modules.comfy.messages import GenMessage, GenState
from modules.face.FaceFinder import FaceFrame
from modules.gl import Blit, Fbo, RenderBase, Shader, Style, Texture, clear_color
from modules.gl.Utils import fit
from modules.render.inputs import Inputs
from modules.render.layers import CameraLayer, CreditLayer, DestructLayer, DiffusionLayer, FlashLayer, LayerBase, RevealLayer, StatusLayer
from modules.render.layers.CreditLayer import caption_shown, fade_step, new_line_due
from modules.render.layers.passes import copy
from modules.render.Sequencer import SequenceType, Sequencer
from modules.render.settings import RenderSettings
from modules.render.stages import STAGES, Layers, Stage

# 32 bit: the trail (0.995) adds steps too small for half floats
COMPOSITION_FORMAT = GL_RGBA32F


class SmileRender(RenderBase):
    """The installation display. Runs the sequencer every frame (it is the app clock), then lets the
    stage of the current sequence state render the layers, and fits its output in the window.
    The grin frames come from the player, which is asked every frame for the one that is due.
    Keys: Enter starts a sequence; f fullscreen, w borderless, Esc quit (the window manager)."""

    def __init__(self, settings: RenderSettings, sequencer: Sequencer, player: GrinPlayer, width: int, height: int) -> None:
        super().__init__(settings.window)
        self.settings: RenderSettings = settings
        self.sequencer: Sequencer = sequencer
        self.player: GrinPlayer = player
        self._plays: int = player.plays
        self.width: int = width
        self.height: int = height

        self.inputs: Inputs = Inputs()
        camera = CameraLayer(self.inputs.camera, self.inputs.faces, settings)
        self.layers: Layers = Layers(
            camera=camera,
            flash=FlashLayer(camera, self.inputs.white),
            reveal=RevealLayer(self.inputs.capture_depth, self.inputs.white),
            diffusion=DiffusionLayer(self.inputs.diffusion),
            destruct=DestructLayer(self.inputs.diffusion_depth),
        )
        # the artist's credit: the QR code always, the caption over the grin and the result
        self._credit: CreditLayer = CreditLayer(settings)
        self._credit_alpha: float = 0.0
        self._credit_time: float = time.time()
        self._credit_sequence: SequenceType = SequenceType.NONE
        # the operator's status line, drawn over the output while there is something to say
        self._status: StatusLayer = StatusLayer()
        self._status_text: str | None = None
        self._all_layers: list[LayerBase] = list(vars(self.layers).values()) + [self._credit, self._status]

        # one instance per stage class, shared by the sequence types it covers
        stages: dict[type[Stage], Stage] = {cls: cls(self.inputs, self.layers, settings) for cls in set(STAGES.values())}
        self._stages: dict[SequenceType, Stage] = {seq: stages[cls] for seq, cls in STAGES.items()}
        self._stage: Stage | None = None

        self._output: Texture = self.layers.camera.texture
        self._freeze_fbo: Fbo = Fbo()
        self._frozen: bool = False

        self.window_manager.add_keyboard_callback(self._on_key)

    # RENDER THREAD
    def allocate(self) -> None:
        for layer in self._all_layers:
            layer.allocate(self.width, self.height, COMPOSITION_FORMAT)
        self._freeze_fbo.allocate(self.width, self.height, COMPOSITION_FORMAT)
        Shader.enable_hot_reload()

    def deallocate(self) -> None:
        for layer in self._all_layers:
            layer.deallocate()
        self._freeze_fbo.deallocate()
        self.inputs.deallocate()

    def update(self) -> None:
        self._play_grin()
        # sequencer first: images handed over with a state change (the depth with CONTROL_0)
        # are then uploaded before the stage uses them
        message = self.sequencer.update()
        self._notify_update()
        self.inputs.update()

        Style.reset_state()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        for layer in self._all_layers:
            layer.update()

        stage: Stage = self._stages[message.sequence_type]
        if stage is not self._stage:
            self._stage = stage
            stage.enter()
        stage.update(message)

        self._output = stage.output
        self._render_credit(message.sequence_type)
        status: str | None = self._status_text
        if status:
            self._status.render(self._output, status)
            self._output = self._status.texture
        if self.settings.freeze:
            if not self._frozen:
                copy(self._output, self._freeze_fbo)
            self._output = self._freeze_fbo
        self._frozen = self.settings.freeze

    def _render_credit(self, sequence_type: SequenceType) -> None:
        now: float = time.time()
        if new_line_due(sequence_type, self._credit_sequence):
            self._credit.new_visitor()
        self._credit_sequence = sequence_type
        shown: bool = caption_shown(sequence_type, self.settings.credit_camera)
        self._credit_alpha = fade_step(self._credit_alpha, shown, now - self._credit_time)
        self._credit_time = now
        if self.settings.credit:
            self._credit.render(self._output, self._credit_alpha)
            self._output = self._credit.texture

    def _play_grin(self) -> None:
        """Show the grin frame that is due. Its first frame arrives during DIFFUSION and blends in as
        a generation step; DIFF_LOOP starts when the player starts playing."""
        frame: np.ndarray | None = self.player.update(time.time())
        if frame is not None:
            self.inputs.diffusion.set_image(frame)
        started: bool = self.player.plays != self._plays
        if started:
            self._plays = self.player.plays
            self.sequencer.set_sequence(SequenceType.DIFF_LOOP)
        if started or (frame is not None and self.player.playing):
            self.sequencer.set_diffusion_progress(self.player.progress)

    def draw_main(self, width: int, height: int) -> None:
        clear_color(0, 0, 0, 1)
        Style.reset_state()
        self._draw_fit(self._output, width, height)
        if self.settings.show_camera and self.inputs.camera.allocated:
            self._draw_fit(self.inputs.camera, width, height)

    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None:
        pass

    def on_main_window_resize(self, width: int, height: int) -> None:
        pass

    @staticmethod
    def _draw_fit(texture: Texture, width: int, height: int) -> None:
        if not texture.allocated:
            return
        x, y, w, h = fit(texture.width, texture.height, width, height)
        glViewport(int(x), int(y), int(w), int(h))
        Blit.use(texture)

    def _on_key(self, key: int, action: int, mods: int) -> None:
        if action == glfw.PRESS and key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self.sequencer.start_sequence()

    # ANY THREAD
    def set_cam_image(self, image: np.ndarray) -> None:
        self.inputs.camera.set_image(image)

    def set_cap_image(self, image: np.ndarray) -> None:
        """A new visitor's capture. It is the first diffusion image too, until the generator sends
        one: without that, a failed generation would blend the previous visitor's last frame in."""
        self.inputs.capture.set_image(image)
        self.inputs.diffusion.set_image(image)

    def set_status(self, text: str | None) -> None:
        """A line for the operator in the corner of the image; None for no line."""
        self._status_text = text

    def set_warmup(self, step: int | None) -> None:
        """The step the generator's warm-up has reached: the camera is in pixels, smaller per step; None is ready."""
        self.layers.camera.warmup = step

    def set_faces(self, frame: FaceFrame) -> None:
        self.inputs.faces.set(frame)

    def set_diffusion_message(self, message: GenMessage) -> None:
        # the image first, so it is uploaded no later than the frame the new state starts
        if message.state == GenState.CAM_DEPTH:
            if message.image is not None:
                self.inputs.capture_depth.set_image(message.image)
            self.sequencer.start_reveal()
        elif message.state == GenState.DIFFUSION:
            if message.image is not None:
                self.inputs.diffusion.set_image(message.image)
            self.sequencer.set_sequence(SequenceType.DIFFUSION)
            self.sequencer.set_diffusion_progress(message.progress)
        elif message.state == GenState.IDLE:
            if message.image is not None:
                self.inputs.diffusion_depth.set_image(message.image)
            self.sequencer.finish()
