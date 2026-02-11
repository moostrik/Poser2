from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture
import numpy as np
from pathlib import Path


class Lut(Shader):
    """
    LUT (Look-Up Table) shader for color grading.
    Supports loading .cube format 3D LUTs.
    """

    def __init__(self) -> None:
        super().__init__()
        # Only initialize LUT-specific attributes once
        if hasattr(self, '_lut_initialized'):
            return
        self._lut_initialized = True

        self._lut_tex_id: int = 0
        self._lut_size: int = 0
        self._lut_title: str = ""
        self._lut_loaded: bool = False
        self._domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def lut_loaded(self) -> bool:
        """Check if a LUT is currently loaded."""
        return self._lut_loaded

    @property
    def lut_size(self) -> int:
        """Get the size of the currently loaded LUT (e.g., 33 for 33x33x33)."""
        return self._lut_size

    @property
    def lut_title(self) -> str:
        """Get the title from the loaded .cube file."""
        return self._lut_title

    def load_cube(self, filepath: str) -> bool:
        """
        Load a .cube LUT file and create a 3D texture.

        Args:
            filepath: Path to the .cube file

        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(filepath)
        if not path.exists():
            print(f"Lut: File not found: {filepath}")
            return False

        if path.suffix.lower() != '.cube':
            print(f"Lut: Unsupported file format: {path.suffix}")
            return False

        try:
            lut_data, size, title, domain_min, domain_max = self._parse_cube_file(path)
        except Exception as e:
            print(f"Lut: Failed to parse .cube file: {e}")
            return False

        # Clean up existing LUT texture if one was loaded
        self._deallocate_lut_texture()

        # Create 3D texture
        if not self._create_lut_texture(lut_data, size):
            return False

        self._lut_size = size
        self._lut_title = title
        self._domain_min = domain_min
        self._domain_max = domain_max
        self._lut_loaded = True

        print(f"Lut: Loaded '{title}' ({size}x{size}x{size})")
        return True

    def _parse_cube_file(self, path: Path) -> tuple[np.ndarray, int, str, tuple, tuple]:
        """
        Parse a .cube LUT file.

        Returns:
            Tuple of (lut_data, size, title, domain_min, domain_max)
        """
        size = 0
        title = path.stem
        domain_min = (0.0, 0.0, 0.0)
        domain_max = (1.0, 1.0, 1.0)
        rgb_values: list[tuple[float, float, float]] = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse header keywords
                if line.startswith('TITLE'):
                    # Extract title from quotes if present
                    parts = line.split('"')
                    if len(parts) >= 2:
                        title = parts[1]
                    else:
                        title = line.split(None, 1)[1] if len(line.split()) > 1 else title
                    continue

                if line.startswith('LUT_3D_SIZE'):
                    parts = line.split()
                    if len(parts) >= 2:
                        size = int(parts[1])
                    continue

                if line.startswith('DOMAIN_MIN'):
                    parts = line.split()
                    if len(parts) >= 4:
                        domain_min = (float(parts[1]), float(parts[2]), float(parts[3]))
                    continue

                if line.startswith('DOMAIN_MAX'):
                    parts = line.split()
                    if len(parts) >= 4:
                        domain_max = (float(parts[1]), float(parts[2]), float(parts[3]))
                    continue

                # Skip other keywords (LUT_1D_SIZE, etc.)
                if any(line.startswith(kw) for kw in ['LUT_1D_SIZE', 'LUT_1D_INPUT_RANGE', 'LUT_3D_INPUT_RANGE']):
                    continue

                # Try to parse as RGB values
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                        rgb_values.append((r, g, b))
                    except ValueError:
                        continue

        if size == 0:
            raise ValueError("LUT_3D_SIZE not found in .cube file")

        expected_entries = size * size * size
        if len(rgb_values) != expected_entries:
            raise ValueError(f"Expected {expected_entries} LUT entries, found {len(rgb_values)}")

        # Convert to numpy array with shape (size, size, size, 3)
        # .cube format stores data in R-major order: R varies fastest, then G, then B
        lut_array = np.array(rgb_values, dtype=np.float32).reshape(size, size, size, 3)

        return lut_array, size, title, domain_min, domain_max

    def _create_lut_texture(self, lut_data: np.ndarray, size: int) -> bool:
        """Create a 3D texture from LUT data."""
        try:
            self._lut_tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_3D, self._lut_tex_id)

            # Set texture parameters - use linear interpolation for smooth color transitions
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # Upload 3D texture data
            glTexImage3D(
                GL_TEXTURE_3D,
                0,  # mipmap level
                GL_RGB32F,  # internal format
                size, size, size,  # dimensions
                0,  # border
                GL_RGB,  # format
                GL_FLOAT,  # type
                lut_data
            )

            glBindTexture(GL_TEXTURE_3D, 0)
            return True

        except Exception as e:
            print(f"Lut: Failed to create 3D texture: {e}")
            if self._lut_tex_id:
                glDeleteTextures(1, [self._lut_tex_id])
                self._lut_tex_id = 0
            return False

    def _deallocate_lut_texture(self) -> None:
        """Clean up the LUT 3D texture."""
        if self._lut_tex_id:
            glDeleteTextures(1, [self._lut_tex_id])
            self._lut_tex_id = 0
        self._lut_loaded = False
        self._lut_size = 0

    def deallocate(self) -> None:
        """Clean up OpenGL resources including the LUT texture."""
        self._deallocate_lut_texture()
        super().deallocate()

    def use(self, tex0: Texture, strength: float = 1.0) -> None:
        """
        Apply the loaded LUT to an input texture.

        Args:
            tex0: Input texture to apply the LUT to
            strength: Blend strength between original (0.0) and LUT result (1.0)
        """
        if not self.allocated or not self.shader_program:
            print("Lut shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("Lut shader: input texture not allocated.")
            return
        if not self._lut_loaded:
            print("Lut shader: no LUT loaded. Call load_cube() first.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture to unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Bind LUT 3D texture to unit 1
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, self._lut_tex_id)

        # Configure shader uniforms
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("lutTex"), 1)
        glUniform1f(self.get_uniform_loc("strength"), strength)
        glUniform3f(self.get_uniform_loc("domainMin"), *self._domain_min)
        glUniform3f(self.get_uniform_loc("domainMax"), *self._domain_max)

        # Render
        draw_quad()

        # Unbind 3D texture (optional but clean)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, 0)
        glActiveTexture(GL_TEXTURE0)
