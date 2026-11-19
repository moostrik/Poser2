"""ViewMask: where the player is in the view: the segmentation's mask placed in the view's window."""

# Third-party imports
import torch
from OpenGL.GL import GL_R16F

# Local application imports
from modules.flow import Port, TextureKind
from modules.gl import Fbo, Style, Tensor, Texture
from modules.inference import segmentation
from modules.utils import Rect


class ViewMask:
    """`place` draws a segmentation mask into the view at the box its crop was cut at, 0 around it; none, 0 all over.
    Its own texture."""

    def __init__(self) -> None:
        self._mask: Fbo = Fbo()
        self._source: Tensor = Tensor()     # the segmentation's mask, taken into GL

    @property
    def mask(self) -> Texture:
        return self._mask

    @property
    def ports(self) -> dict[str, Port]:
        return {'mask': Port(self._mask, TextureKind.SCALAR)}

    def allocate(self, width: int, height: int) -> None:
        self._mask.allocate(width, height, GL_R16F)
        self._mask.clear()

    def deallocate(self) -> None:
        self._mask.deallocate()
        if self._source.allocated:
            self._source.deallocate()

    def clear(self) -> None:
        self._mask.clear()

    def place(self, mask: segmentation.Image | None, window: Rect) -> None:
        """`mask` at its box, in the view whose window on the camera image is `window` (uv, rows down)."""
        self._mask.clear()
        if mask is None or mask.box is None:
            return
        # Handed on written; its memory is kept from being reused until this stream's copy is done
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        mask.mask.record_stream(stream)
        self._source.set_tensor(mask.mask)
        self._source.update()
        box: Rect = mask.box
        placed: Rect = Rect((box.x - window.x) / window.width, (box.y - window.y) / window.height,
                            box.width / window.width, box.height / window.height)
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._mask.begin()
        self._source.draw(dst=placed)
        self._mask.end()
        Style.pop_style()
