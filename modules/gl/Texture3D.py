"""Texture3D - 3D texture (GL_TEXTURE_3D) for volumetric fluid simulation.

Supports trilinear filtering between depth layers and per-field wrap modes
for implicit boundary conditions (no explicit obstacle border needed).

Usage:
    velocity_tex = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0,0,0,0))
    velocity_tex.allocate(256, 192, 32, GL_RGBA16F)  # u,v,w + spare

    pressure_tex = Texture3D(wrap=GL_CLAMP_TO_EDGE)   # Neumann BC
    pressure_tex.allocate(256, 192, 32, GL_R16F)
"""
from OpenGL.GL import *  # type: ignore
import numpy as np

from .Texture import get_format, get_data_type, get_channel_count, get_numpy_dtype


class Texture3D:
    """3D OpenGL texture (GL_TEXTURE_3D) for volumetric data.

    Wrap modes control boundary conditions:
        GL_CLAMP_TO_BORDER + (0,0,0,0) → Dirichlet / no-slip (velocity, density)
        GL_CLAMP_TO_EDGE              → Neumann / zero-gradient (pressure, temperature)
        GL_CLAMP_TO_BORDER + (1,0,0,0) → out-of-bounds = obstacle (obstacle mask)
    """

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Initialize 3D texture with configuration.

        Args:
            interpolation: Filter mode for min/mag (GL_LINEAR for trilinear, GL_NEAREST)
            wrap: Wrap mode for all 3 axes (S, T, R)
            border_color: RGBA color when wrap is GL_CLAMP_TO_BORDER
        """
        self.allocated: bool = False
        self.width: int = 0
        self.height: int = 0
        self.depth: int = 0
        self.internal_format: Constant = GL_NONE
        self.format: Constant = GL_NONE
        self.data_type: Constant = GL_NONE
        self.tex_id: int = 0
        self._interpolation: int = interpolation
        self._wrap: int = wrap
        self._border_color: tuple[float, float, float, float] = border_color

    @property
    def texture(self) -> 'Texture3D':
        """Convenience property for interface compatibility."""
        return self

    def allocate(self, width: int, height: int, depth: int, internal_format) -> None:
        """Allocate 3D texture with specified dimensions and format.

        Args:
            width: Texture width in texels
            height: Texture height in texels
            depth: Number of depth layers
            internal_format: OpenGL internal format (e.g., GL_RGBA16F, GL_R16F)
        """
        data_type = get_data_type(internal_format)
        if data_type == GL_NONE:
            return

        self.width = width
        self.height = height
        self.depth = depth
        self.internal_format = internal_format
        self.format = get_format(internal_format)
        self.data_type = data_type
        self.tex_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_3D, self.tex_id)

        # Wrap mode on all 3 axes
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, self._wrap)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, self._wrap)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, self._wrap)

        # Filtering (GL_LINEAR = trilinear between layers)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, self._interpolation)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, self._interpolation)

        # Border color (used when wrap = GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, self._border_color)

        # Grayscale swizzle for single-channel formats
        if self.format == GL_RED:
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_SWIZZLE_R, GL_RED)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_SWIZZLE_G, GL_RED)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_SWIZZLE_B, GL_RED)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_SWIZZLE_A, GL_ONE)

        # Allocate storage
        glTexImage3D(GL_TEXTURE_3D, 0, self.internal_format,
                     self.width, self.height, self.depth, 0,
                     self.format, self.data_type, None)

        glBindTexture(GL_TEXTURE_3D, 0)
        self.allocated = True
        self.clear()

    def deallocate(self) -> None:
        """Release texture resources."""
        if not self.allocated:
            return
        self.allocated = False
        self.width = 0
        self.height = 0
        self.depth = 0
        self.internal_format: Constant = GL_NONE
        self.format: Constant = GL_NONE
        self.data_type: Constant = GL_NONE
        glDeleteTextures(1, [self.tex_id])
        self.tex_id = 0

    def bind(self) -> None:
        """Bind as GL_TEXTURE_3D."""
        glBindTexture(GL_TEXTURE_3D, self.tex_id)

    def unbind(self) -> None:
        """Unbind GL_TEXTURE_3D."""
        glBindTexture(GL_TEXTURE_3D, 0)

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """Clear entire 3D texture to specified color (OpenGL 4.4+).

        Args:
            r, g, b, a: Clear color components [0.0, 1.0]
        """
        if not self.allocated:
            return

        channels = get_channel_count(self.format)
        if channels is None:
            return

        if self.data_type == GL_UNSIGNED_BYTE:
            clear_data = np.array([r * 255, g * 255, b * 255, a * 255][:channels], dtype=np.uint8)
        elif self.data_type == GL_HALF_FLOAT:
            clear_data = np.array([r, g, b, a][:channels], dtype=np.float16)
        else:  # GL_FLOAT
            clear_data = np.array([r, g, b, a][:channels], dtype=np.float32)

        glClearTexImage(self.tex_id, 0, self.format, self.data_type, clear_data)

    def read_to_numpy(self, flip: bool = False) -> np.ndarray | None:
        """Read 3D texture data back to CPU as NumPy array.

        Returns:
            NumPy array with shape (depth, height, width [, channels]) or None
        """
        if not self.allocated:
            return None

        dtype = get_numpy_dtype(self.data_type, self.internal_format)
        if dtype is None:
            return None

        channels = get_channel_count(self.format)
        if channels is None:
            return None

        if channels == 1:
            pixels = np.zeros((self.depth, self.height, self.width), dtype=dtype)
        else:
            pixels = np.zeros((self.depth, self.height, self.width, channels), dtype=dtype)

        self.bind()
        glGetTexImage(GL_TEXTURE_3D, 0, self.format, self.data_type, pixels)
        self.unbind()

        if flip:
            pixels = pixels[:, ::-1, :]  # Flip Y per layer
        return pixels


class SwapTexture3D:
    """Double-buffered 3D texture for ping-pong compute operations.

    No FBO needed — all reads/writes done via imageLoad/imageStore in compute shaders.
    """

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        self._textures: list[Texture3D] = [
            Texture3D(interpolation, wrap, border_color),
            Texture3D(interpolation, wrap, border_color)
        ]
        self._swap_state: int = 0

    @property
    def allocated(self) -> bool:
        return self._textures[0].allocated and self._textures[1].allocated

    @property
    def width(self) -> int:
        return self._textures[self._swap_state].width

    @property
    def height(self) -> int:
        return self._textures[self._swap_state].height

    @property
    def depth(self) -> int:
        return self._textures[self._swap_state].depth

    @property
    def internal_format(self) -> Constant:
        return self._textures[self._swap_state].internal_format

    @property
    def format(self) -> Constant:
        return self._textures[self._swap_state].format

    @property
    def data_type(self) -> Constant:
        return self._textures[self._swap_state].data_type

    @property
    def tex_id(self) -> int:
        return self._textures[self._swap_state].tex_id

    @property
    def texture(self) -> Texture3D:
        """Current write target."""
        return self._textures[self._swap_state]

    @property
    def back_texture(self) -> Texture3D:
        """Previous buffer for reading."""
        return self._textures[1 - self._swap_state]

    def allocate(self, width: int, height: int, depth: int, internal_format) -> None:
        """Allocate both 3D texture buffers."""
        self._textures[0].allocate(width, height, depth, internal_format)
        self._textures[1].allocate(width, height, depth, internal_format)

    def deallocate(self) -> None:
        """Deallocate both buffers."""
        self._textures[0].deallocate()
        self._textures[1].deallocate()

    def swap(self) -> None:
        """Swap front/back buffers."""
        self._swap_state = 1 - self._swap_state

    def bind(self) -> None:
        """Bind current texture."""
        self._textures[self._swap_state].bind()

    def unbind(self) -> None:
        """Unbind current texture."""
        self._textures[self._swap_state].unbind()

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """Clear current buffer."""
        self._textures[self._swap_state].clear(r, g, b, a)

    def clear_back(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """Clear back buffer."""
        self._textures[1 - self._swap_state].clear(r, g, b, a)

    def clear_all(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """Clear both buffers."""
        self._textures[0].clear(r, g, b, a)
        self._textures[1].clear(r, g, b, a)
