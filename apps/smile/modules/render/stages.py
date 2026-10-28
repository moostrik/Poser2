"""What the render does in each sequence state: which layers render, with which values, and which
layer is shown. One stage covers one or more SequenceTypes; enter() runs when the stage changes."""

from dataclasses import dataclass

from modules.gl import Texture
from modules.render.inputs import Inputs
from modules.render.layers import CameraLayer, DestructLayer, DiffusionLayer, FlashLayer, RevealLayer
from modules.render.Sequencer import CAPTURE_AT, SeqenceMessage, SequenceType
from modules.render.settings import RenderSettings


@dataclass
class Layers:
    camera: CameraLayer
    flash: FlashLayer
    reveal: RevealLayer
    diffusion: DiffusionLayer
    destruct: DestructLayer


class Stage:
    def __init__(self, inputs: Inputs, layers: Layers, settings: RenderSettings) -> None:
        self.inputs: Inputs = inputs
        self.layers: Layers = layers
        self.settings: RenderSettings = settings

    def enter(self) -> None:
        """Called on the first frame of this stage."""

    def update(self, message: SeqenceMessage) -> None:
        """Called every frame of this stage."""

    @property
    def output(self) -> Texture:
        raise NotImplementedError


class CameraStage(Stage):
    """Black and white camera; the chosen face's box turns to colour as its charge fills."""

    def update(self, message: SeqenceMessage) -> None:
        # value_1 is the eased charge: 0 while nobody qualifies
        self.layers.camera.render(message.value_1)

    @property
    def output(self) -> Texture:
        return self.layers.camera.texture


class FlashStage(Stage):
    """Flash to white; the camera freezes at the capture. The face's marks fade out before it, so
    the frozen frame has none: their saturated colour would outlast the image in the flash.
    At the capture the diffusion layer starts over from it: whatever follows the flash (the flash
    out, or the result at once when the generator fails) has nothing of the previous visitor."""

    # the marks are gone at this part of the time before the capture, so the last frames have none
    MARKS_GONE: float = 0.8

    def update(self, message: SeqenceMessage) -> None:
        if message.sequence_type == SequenceType.FLASH:
            marks: float = max(0.0, 1.0 - message.progress / (CAPTURE_AT * self.MARKS_GONE))
            self.layers.camera.render(1.0, marks)
        elif message.sequence_type == SequenceType.CAPTURE:
            self.layers.diffusion.seed(self.inputs.capture)     # uploaded this frame
        self.layers.flash.render(message.value_1)

    @property
    def output(self) -> Texture:
        return self.layers.flash.texture


class RevealStage(Stage):
    """The capture (the diffusion layer, seeded at the capture) appears from white by its depth,
    while ComfyUI starts generating."""

    def update(self, message: SeqenceMessage) -> None:
        if message.sequence_type == SequenceType.CONTROL_0 and message.initial:
            self.layers.reveal.set_mask()
        self.layers.reveal.render(self.layers.diffusion.texture, 1.0 - message.value_1)

    @property
    def output(self) -> Texture:
        return self.layers.reveal.texture


class DiffusionStage(Stage):
    """Generated images over the capture, then the grin, then the result holds."""

    def update(self, message: SeqenceMessage) -> None:
        s: RenderSettings = self.settings
        if message.sequence_type == SequenceType.DIFF_LOOP:
            trail: float = s.loop_trail
        else:
            trail = s.trail if s.trail_active else 0.0
        self.layers.diffusion.render(trail)

    @property
    def output(self) -> Texture:
        return self.layers.diffusion.texture


class DestructStage(Stage):
    """The result blooms, blurs and dissolves by its depth into the black and white camera."""

    def update(self, message: SeqenceMessage) -> None:
        s: RenderSettings = self.settings
        layers: Layers = self.layers
        layers.camera.render(0.0)
        if message.sequence_type == SequenceType.BLOOM:
            if message.initial:
                layers.destruct.set_mask()
            layers.destruct.render(layers.diffusion.texture, layers.camera.texture,
                                   message.value_1 * s.destroy_bloom, message.value_2)
        else:
            layers.diffusion.blur(s.destroy_blur)
            layers.destruct.render(layers.diffusion.texture, layers.camera.texture,
                                   message.value_2 * s.destroy_bloom, message.value_1)

    @property
    def output(self) -> Texture:
        return self.layers.destruct.texture


STAGES: dict[SequenceType, type[Stage]] = {
    SequenceType.NONE:          DiffusionStage,
    SequenceType.BLOOM:         DestructStage,
    SequenceType.FADE:          DestructStage,
    SequenceType.CAMERA_SHOW:   CameraStage,
    SequenceType.CAMERA_DETECT: CameraStage,
    SequenceType.FLASH:         FlashStage,
    SequenceType.CAPTURE:       FlashStage,
    SequenceType.ASK:           FlashStage,
    SequenceType.CONTROL_0:     RevealStage,
    SequenceType.CONTROL_1:     RevealStage,
    SequenceType.CONTROL_2:     RevealStage,
    SequenceType.DIFFUSION:     DiffusionStage,
    SequenceType.DIFF_LOOP:     DiffusionStage,
    SequenceType.TIMEOUT:       DiffusionStage,
    SequenceType.RESET:         DiffusionStage,
}
