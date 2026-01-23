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

import logging
import math
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from .Shader import Shader
from .Texture import Texture


class ComputeShader(Shader):
    """Base class for compute shaders with dispatch and image binding support."""

    # Shader file suffix for compute shaders
    COMPUTE_SUFFIX = '.comp'

    # Default workgroup size (can be overridden per-shader)
    WORKGROUP_SIZE_X: int = 16
    WORKGROUP_SIZE_Y: int = 16
    WORKGROUP_SIZE_Z: int = 1

    def __init__(self) -> None:
        """Initialize compute shader and discover .comp file."""
        # Skip if already initialized (singleton pattern from parent)
        if hasattr(self, '_initialized'):
            return

        # Don't call parent __init__ yet - we need to set up compute-specific paths first
        self._initialized = True
        self.allocated: bool = False
        self.shader_name: str = self.__class__.__name__
        self.shader_program: int | None = None
        self.need_reload: bool = False
        self._ref_count: int = 0
        self._uniform_cache: dict[str, int] = {}

        # Discover compute shader file
        import inspect
        shader_name_normalized = self.shader_name.lower()
        self.shader_dir = Path(inspect.getfile(self.__class__)).parent

        compute_path = self.shader_dir / f"{shader_name_normalized}{self.COMPUTE_SUFFIX}"
        self.compute_file_path: Path | None = compute_path if compute_path.exists() else None

        # Clear parent's vertex/fragment paths (not used for compute)
        self.vertex_file_path = None
        self.fragment_file_path = None

        # Import threading for reload lock
        import threading
        self._reload_lock = threading.Lock()

    def allocate(self) -> None:
        """Compile compute shader and create OpenGL program."""
        self._ref_count += 1

        if self.allocated:
            return

        # Register for hot-reload
        with Shader._observer_lock:
            if self not in Shader._monitored_shaders:
                Shader._monitored_shaders.append(self)

            if Shader._hot_reload_enabled and self.shader_dir:
                Shader._watch_directory(self.shader_dir)

        # Compile compute shader
        self.allocated = self._compile_compute_shader(verbose=True)

    def _compile_compute_shader(self, verbose: bool = False) -> bool:
        """Load and compile compute shader. Returns True if successful."""
        # Load compute shader source
        compute_source: str = ''
        if self.compute_file_path:
            compute_source = self.read_shader_source(str(self.compute_file_path))

        if not compute_source:
            logging.error(f"{self.shader_name}: No compute shader found at {self.compute_file_path}")
            return False

        try:
            # Compile compute shader
            compute_shader = glCreateShader(GL_COMPUTE_SHADER)
            glShaderSource(compute_shader, compute_source)
            glCompileShader(compute_shader)

            if not glGetShaderiv(compute_shader, GL_COMPILE_STATUS):
                error = glGetShaderInfoLog(compute_shader).decode()
                logging.error(f"{self.shader_name} COMPUTE SHADER ERROR:\n{error}")
                glDeleteShader(compute_shader)
                return False

            # Link program
            new_program = glCreateProgram()
            glAttachShader(new_program, compute_shader)
            glLinkProgram(new_program)

            if not glGetProgramiv(new_program, GL_LINK_STATUS):
                error = glGetProgramInfoLog(new_program).decode()
                logging.error(f"{self.shader_name} LINK ERROR:\n{error}")
                glDeleteShader(compute_shader)
                glDeleteProgram(new_program)
                return False

            # Clean up shader object
            glDeleteShader(compute_shader)

            # Replace old program if exists
            if self.shader_program is not None:
                glDeleteProgram(self.shader_program)
            self.shader_program = new_program

            # Clear uniform cache
            self._uniform_cache.clear()

            if verbose:
                logging.info(f"{self.shader_name} compute shader loaded successfully")
            return True

        except Exception as e:
            logging.error(f"{self.shader_name} UNEXPECTED ERROR: {e}")
            return False

    def reload(self) -> bool:
        """Reload compute shader if marked for reload."""
        if not self.need_reload:
            return False

        with self._reload_lock:
            self.need_reload = False
            compiled = self._compile_compute_shader(verbose=True)
            print(f"Compute shader {self.shader_name} reload {'succeeded' if compiled else 'failed'}")
            return True

    def _on_file_changed(self, file_path: Path) -> None:
        """Mark shader for reload when .comp file changes."""
        if self.compute_file_path and file_path == self.compute_file_path:
            self.need_reload = True

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

    # ========== Texture Sampling (for hybrid read) ==========

    def bind_texture(self, unit: int, texture: Texture, uniform_name: str) -> None:
        """Bind texture for sampler access (traditional texture() calls).

        Useful when you need filtered sampling alongside image access.

        Args:
            unit: Texture unit (0, 1, 2, ...)
            texture: Texture to bind
            uniform_name: Name of sampler uniform in shader
        """
        glActiveTexture(int(GL_TEXTURE0) + unit)
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)
        glUniform1i(self.get_uniform_loc(uniform_name), unit)

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
