"""Texture2DArray - 2D texture array (GL_TEXTURE_2D_ARRAY) for volumetric data.

Each layer is a standard 2D texture with optimal 2D tiling/swizzle layout,
avoiding the 3D tile pathology that causes performance cliffs at certain
GL_TEXTURE_3D volume sizes.

Unlike GL_TEXTURE_3D, sampler2DArray does NOT trilinearly interpolate
between layers — callers must handle Z interpolation manually in GLSL.

Usage:
    density_tex = Texture2DArray(wrap=GL_CLAMP_TO_BORDER, border_color=(0,0,0,0))
    density_tex.allocate(3840, 2160, 28, GL_RGBA16F)
"""
from OpenGL.GL import *  # type: ignore
import numpy as np

from .Texture import get_format, get_data_type, get_channel_count, get_numpy_dtype


class Texture2DArray:
    """2D array OpenGL texture (GL_TEXTURE_2D_ARRAY) for volumetric data.

    Same interface as Texture3D but backed by GL_TEXTURE_2D_ARRAY for
    optimal per-layer 2D memory tiling.

    Wrap modes control boundary conditions (S/T only — R axis is layer index):
        GL_CLAMP_TO_BORDER + (0,0,0,0) → Dirichlet / no-slip (density)
        GL_CLAMP_TO_EDGE              → Neumann / zero-gradient
    """

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Initialize 2D array texture with configuration.

        Args:
            interpolation: Filter mode for min/mag (GL_LINEAR for bilinear, GL_NEAREST)
            wrap: Wrap mode for S and T axes
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
    def texture(self) -> 'Texture2DArray':
        """Convenience property for interface compatibility."""
        return self

    def allocate(self, width: int, height: int, depth: int, internal_format) -> None:
        """Allocate 2D array texture with specified dimensions and format.

        Args:
            width: Texture width in texels
            height: Texture height in texels
            depth: Number of array layers
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

        glBindTexture(GL_TEXTURE_2D_ARRAY, self.tex_id)

        # Wrap mode on S and T axes only (R is layer index, not spatial)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, self._wrap)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, self._wrap)

        # Filtering (GL_LINEAR = bilinear within each layer)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, self._interpolation)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, self._interpolation)

        # Border color (used when wrap = GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, self._border_color)

        # Grayscale swizzle for single-channel formats
        if self.format == GL_RED:
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_SWIZZLE_R, GL_RED)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_SWIZZLE_G, GL_RED)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_SWIZZLE_B, GL_RED)
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_SWIZZLE_A, GL_ONE)

        # Allocate storage (same call as 3D — target determines interpretation)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, self.internal_format,
                     self.width, self.height, self.depth, 0,
                     self.format, self.data_type, None)

        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
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
        """Bind as GL_TEXTURE_2D_ARRAY."""
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.tex_id)

    def unbind(self) -> None:
        """Unbind GL_TEXTURE_2D_ARRAY."""
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """Clear entire 2D array texture to specified color (OpenGL 4.4+).

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
        """Read 2D array texture data back to CPU as NumPy array.

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
        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, self.format, self.data_type, pixels)
        self.unbind()

        if flip:
            pixels = pixels[:, ::-1, :]  # Flip Y per layer
        return pixels


class SwapTexture2DArray:
    """Double-buffered 2D array texture for ping-pong compute operations.

    Same interface as SwapTexture3D but backed by GL_TEXTURE_2D_ARRAY.
    """

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        self._textures: list[Texture2DArray] = [
            Texture2DArray(interpolation, wrap, border_color),
            Texture2DArray(interpolation, wrap, border_color)
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
    def texture(self) -> Texture2DArray:
        """Current write target."""
        return self._textures[self._swap_state]

    @property
    def back_texture(self) -> Texture2DArray:
        """Previous buffer for reading."""
        return self._textures[1 - self._swap_state]

    def allocate(self, width: int, height: int, depth: int, internal_format) -> None:
        """Allocate both 2D array texture buffers."""
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
