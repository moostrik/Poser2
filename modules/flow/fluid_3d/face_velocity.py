"""The 3D fluid's velocity on the cells' faces (the MAC grid, `faces.glsl`)."""

from OpenGL.GL import GL_CLAMP_TO_EDGE, GL_RGBA16F

from modules.gl import Texture3D, SwapTexture3D


class FaceVelocity:
    """The faces in one RGBA16F volume a face longer on every axis: a texel's x, y and z the faces on its cell's
    left, bottom and front, w unused. Timed in the app against three R16F volumes, one a component, it was the
    faster in every pass: one texel read gives a cell's three faces where a pass reads them together."""

    def __init__(self, swap: bool) -> None:
        self.volume: Texture3D = SwapTexture3D(wrap=GL_CLAMP_TO_EDGE) if swap else Texture3D(wrap=GL_CLAMP_TO_EDGE)

    @property
    def allocated(self) -> bool:
        return self.volume.allocated

    def allocate(self, width: int, height: int, depth: int) -> None:
        """The faces of `width` × `height` × `depth` cells."""
        self.volume.allocate(width + 1, height + 1, depth + 1, GL_RGBA16F)

    def deallocate(self) -> None:
        self.volume.deallocate()

    def clear(self) -> None:
        if not self.volume.allocated:
            return
        if isinstance(self.volume, SwapTexture3D):
            self.volume.clear_all()
        else:
            self.volume.clear()

    def swap(self) -> None:
        if isinstance(self.volume, SwapTexture3D):
            self.volume.swap()

    @property
    def back(self) -> Texture3D:
        """A swap's other buffer."""
        assert isinstance(self.volume, SwapTexture3D)
        return self.volume.back
