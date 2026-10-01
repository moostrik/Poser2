import time
from threading import Lock

import glfw
import numpy as np
from OpenGL.GL import GL_RGBA32F, glViewport  # type: ignore

from modules.render.GrinPlayer import GrinPlayer
from modules.comfy.messages import GenMessage, GenState
from modules.face.FaceFinder import FaceFrame
from modules.gl import Blit, Fbo, RenderBase, Shader, Style, Texture, clear_color
from modules.gl.Utils import fit
from modules.render.inputs import Inputs
from modules.render.layers import CameraLayer, DestructLayer, DiffusionLayer, FaceOverlayLayer, FlashLayer, LayerBase, RevealLayer, StatusLayer
from modules.render.layers.face_marks import build_marks
from modules.render.layers.passes import copy
from modules.render.Sequencer import SequenceType, Sequencer
from modules.render.settings import RenderSettings
from modules.render.stages import STAGES, Layers, Stage

# 32 bit: the trail (0.995) adds steps too small for half floats
COMPOSITION_FORMAT = GL_RGBA32F

# the face overlay shows no faces older than this: the detector only runs in the camera states
FACES_STALE: float = 0.5


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
        camera = CameraLayer(self.inputs.camera)
        self.layers: Layers = Layers(
            camera=camera,
            flash=FlashLayer(camera, self.inputs.white),
            reveal=RevealLayer(self.inputs.capture_depth, self.inputs.white),
            diffusion=DiffusionLayer(self.inputs.diffusion),
            destruct=DestructLayer(self.inputs.diffusion_depth),
        )
        # the operator's face overlay is no stage layer: drawn over the output when show_faces is on
        self._faces: FaceOverlayLayer = FaceOverlayLayer()
        # the operator's status line, drawn over the output while there is something to say
        self._status: StatusLayer = StatusLayer()
        self._status_text: str | None = None
        self._all_layers: list[LayerBase] = list(vars(self.layers).values()) + [self._faces, self._status]
        self._faces_mutex: Lock = Lock()
        self._face_frame: FaceFrame | None = None

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
        if self.settings.show_faces:
            self._render_faces(self._output)
            self._output = self._faces.texture
        status: str | None = self._status_text
        if status:
            self._status.render(self._output, status)
            self._output = self._status.texture
        if self.settings.freeze:
            if not self._frozen:
                copy(self._output, self._freeze_fbo)
            self._output = self._freeze_fbo
        self._frozen = self.settings.freeze

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

    def _render_faces(self, source: Texture) -> None:
        with self._faces_mutex:
            frame: FaceFrame | None = self._face_frame
        if frame is None or time.time() - frame.time > FACES_STALE:
            self._faces.set_marks([])
        else:
            self._faces.set_marks(build_marks(frame))
        self._faces.render(source)

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
        self.inputs.capture.set_image(image)

    def set_status(self, text: str | None) -> None:
        """A line for the operator in the corner of the image; None for no line."""
        self._status_text = text

    def set_faces(self, frame: FaceFrame) -> None:
        with self._faces_mutex:
            self._face_frame = frame

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
            self.sequencer.set_sequence(SequenceType.TIMEOUT)
