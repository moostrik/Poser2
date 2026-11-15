"""BlendShaderMode numbers its modes as blend_modes.glsl does."""

import re
import unittest
from pathlib import Path

from modules.gl.shaders import BlendShaderMode

GLSL: Path = Path(__file__).resolve().parents[1] / 'shaders' / 'blend_modes.glsl'


class TestBlendShaderMode(unittest.TestCase):

    def test_every_mode_has_the_glsl_number(self) -> None:
        constants: dict[str, int] = {name: int(value) for name, value in
                                     re.findall(r'const int BLEND_(\w+) = (\d+);', GLSL.read_text(encoding='utf-8'))}
        self.assertEqual(constants, {mode.name: mode.value for mode in BlendShaderMode})


if __name__ == '__main__':
    unittest.main()
