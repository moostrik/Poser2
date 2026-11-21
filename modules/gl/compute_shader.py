"""ComputeShader - Base class for OpenGL compute shaders.

Extends Shader with compute-specific functionality:
- .comp file discovery and compilation
- glDispatchCompute with automatic workgroup calculation
- Image binding helpers for imageLoad/imageStore
- Memory barriers for synchronization

Usage:
    class MyCompute(ComputeShader):
        def use(self, input_tex: Texture, output_tex: Texture):
            glUseProgram(self.shader_program)
            self.bind_image_read(0, input_tex)
            self.bind_image_write(1, output_tex)
            glUniform1f(self.get_uniform_loc("uParam"), 1.0)
            self.dispatch(output_tex.width, output_tex.height)
"""
from __future__ import annotations

import math
from pathlib import Path

from OpenGL.GL import GL_ALL_BARRIER_BITS, GL_COMPUTE_SHADER, GL_FALSE, GL_MAX_COMPUTE_WORK_GROUP_COUNT, GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, GL_MAX_COMPUTE_WORK_GROUP_SIZE, GL_READ_ONLY, GL_READ_WRITE, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_SHADER_STORAGE_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, GL_TRUE, GL_WRITE_ONLY, glBindImageTexture, glDispatchCompute, glGetIntegeri_v, glGetIntegerv, glMemoryBarrier

from .shader import Shader
from .texture import Texture
from .texture_3d import Texture3D
from .texture_2d_array import Texture2DArray


class ComputeShader(Shader):
    """Base class for compute shaders with dispatch and image binding support."""

    # Shader file suffix for compute shaders
    COMPUTE_SUFFIX = '.comp'

    # Default workgroup size (can be overridden per-shader)
    WORKGROUP_SIZE_X: int = 16
    WORKGROUP_SIZE_Y: int = 16
    WORKGROUP_SIZE_Z: int = 1

    @classmethod
    def _stages(cls) -> list[tuple[int, Path]]:
        """The one compute stage: the class's `.comp`."""
        return [(int(GL_COMPUTE_SHADER), cls._source_dir() / f"{cls._source_stem()}{cls.COMPUTE_SUFFIX}")]

    # ========== Dispatch Methods ==========

    def dispatch(self, width: int, height: int, depth: int = 1) -> None:
        """Dispatch compute shader with automatic workgroup calculation.

        Args:
            width: Total width in pixels/elements
            height: Total height in pixels/elements
            depth: Total depth (default 1 for 2D)

        Automatically calculates workgroup counts based on WORKGROUP_SIZE_*.
        Issues memory barrier after dispatch for image access synchronization.
        """
        num_groups_x = math.ceil(width / self.WORKGROUP_SIZE_X)
        num_groups_y = math.ceil(height / self.WORKGROUP_SIZE_Y)
        num_groups_z = math.ceil(depth / self.WORKGROUP_SIZE_Z)

        glDispatchCompute(num_groups_x, num_groups_y, num_groups_z)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def dispatch_groups(self, num_groups_x: int, num_groups_y: int = 1, num_groups_z: int = 1) -> None:
        """Dispatch compute shader with explicit workgroup counts.

        Use when you need precise control over dispatch dimensions.
        """
        glDispatchCompute(num_groups_x, num_groups_y, num_groups_z)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # ========== Image Binding Methods ==========

    def bind_image(self, unit: int, texture: Texture, access: int,
                   internal_format: int | None = None) -> None:
        """Bind texture as image for imageLoad/imageStore.

        Args:
            unit: Image unit (0, 1, 2, ...)
            texture: Texture to bind
            access: GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE
            internal_format: Override format (defaults to texture's format)
        """
        fmt = internal_format if internal_format is not None else texture.internal_format
        glBindImageTexture(unit, texture.tex_id, 0, GL_FALSE, 0, access, fmt)

    def bind_image_read(self, unit: int, texture: Texture,
                        internal_format: int | None = None) -> None:
        """Bind texture as read-only image."""
        self.bind_image(unit, texture, GL_READ_ONLY, internal_format)

    def bind_image_write(self, unit: int, texture: Texture,
                         internal_format: int | None = None) -> None:
        """Bind texture as write-only image."""
        self.bind_image(unit, texture, GL_WRITE_ONLY, internal_format)

    def bind_image_readwrite(self, unit: int, texture: Texture,
                             internal_format: int | None = None) -> None:
        """Bind texture as read-write image (for in-place operations)."""
        self.bind_image(unit, texture, GL_READ_WRITE, internal_format)

    # ========== 3D Image Binding Methods ==========

    def bind_image_3d(self, unit: int, texture: Texture3D, access: int,
                      internal_format: int | None = None) -> None:
        """Bind 3D texture as layered image for imageLoad/imageStore.

        Args:
            unit: Image unit (0, 1, 2, ...)
            texture: Texture3D to bind
            access: GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE
            internal_format: Override format (defaults to texture's format)
        """
        fmt = internal_format if internal_format is not None else texture.internal_format
        glBindImageTexture(unit, texture.tex_id, 0, GL_TRUE, 0, access, fmt)

    def bind_image_3d_read(self, unit: int, texture: Texture3D,
                           internal_format: int | None = None) -> None:
        """Bind 3D texture as read-only layered image."""
        self.bind_image_3d(unit, texture, GL_READ_ONLY, internal_format)

    def bind_image_3d_write(self, unit: int, texture: Texture3D,
                            internal_format: int | None = None) -> None:
        """Bind 3D texture as write-only layered image."""
        self.bind_image_3d(unit, texture, GL_WRITE_ONLY, internal_format)

    def bind_image_3d_readwrite(self, unit: int, texture: Texture3D,
                                internal_format: int | None = None) -> None:
        """Bind 3D texture as read-write layered image."""
        self.bind_image_3d(unit, texture, GL_READ_WRITE, internal_format)


    def bind_image_2d_array(self, unit: int, texture: Texture2DArray, access: int,
                            internal_format: int | None = None) -> None:
        """Bind 2D array texture as layered image for imageLoad/imageStore.

        Args:
            unit: Image unit (0, 1, 2, ...)
            texture: Texture2DArray to bind
            access: GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE
            internal_format: Override format (defaults to texture's format)
        """
        fmt = internal_format if internal_format is not None else texture.internal_format
        glBindImageTexture(unit, texture.tex_id, 0, GL_TRUE, 0, access, fmt)

    def bind_image_2d_array_read(self, unit: int, texture: Texture2DArray,
                                 internal_format: int | None = None) -> None:
        """Bind 2D array texture as read-only layered image."""
        self.bind_image_2d_array(unit, texture, GL_READ_ONLY, internal_format)

    def bind_image_2d_array_write(self, unit: int, texture: Texture2DArray,
                                  internal_format: int | None = None) -> None:
        """Bind 2D array texture as write-only layered image."""
        self.bind_image_2d_array(unit, texture, GL_WRITE_ONLY, internal_format)

    def bind_image_2d_array_readwrite(self, unit: int, texture: Texture2DArray,
                                      internal_format: int | None = None) -> None:
        """Bind 2D array texture as read-write layered image."""
        self.bind_image_2d_array(unit, texture, GL_READ_WRITE, internal_format)

    # ========== Memory Barriers ==========

    @staticmethod
    def barrier_image() -> None:
        """Memory barrier for image load/store operations."""
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    @staticmethod
    def barrier_texture_fetch() -> None:
        """Memory barrier for texture sampling after image writes."""
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT)

    @staticmethod
    def barrier_buffer() -> None:
        """Memory barrier for buffer object access."""
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    @staticmethod
    def barrier_all() -> None:
        """Full memory barrier (use sparingly - expensive)."""
        glMemoryBarrier(GL_ALL_BARRIER_BITS)


# ========== Utility Functions ==========

def get_max_workgroup_size() -> tuple[int, int, int]:
    """Query maximum workgroup size supported by GPU."""
    max_x = glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0)[0]
    max_y = glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1)[0]
    max_z = glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2)[0]
    return (max_x, max_y, max_z)


def get_max_workgroup_count() -> tuple[int, int, int]:
    """Query maximum workgroup count per dimension."""
    max_x = glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0)[0]
    max_y = glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1)[0]
    max_z = glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2)[0]
    return (max_x, max_y, max_z)


def get_max_workgroup_invocations() -> int:
    """Query maximum total invocations per workgroup (x * y * z limit)."""
    return glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
