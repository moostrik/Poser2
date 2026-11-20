"""`clear_value` is the clear color glClearTexImage gets: none for zero, else one array per value, kept."""

import unittest

import numpy as np
from OpenGL.GL import GL_FLOAT, GL_HALF_FLOAT, GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT

from modules.gl.texture import clear_value


class ClearValueTest(unittest.TestCase):
    def test_zero_passes_no_memory(self) -> None:
        self.assertIsNone(clear_value(GL_FLOAT, 4, (0.0, 0.0, 0.0, 0.0)))

    def test_a_color_is_in_the_textures_type(self) -> None:
        color = (1.0, 0.5, 0.0, 1.0)
        byte = clear_value(GL_UNSIGNED_BYTE, 4, color)
        assert byte is not None
        self.assertEqual((byte.dtype, byte.tolist()), (np.dtype(np.uint8), [255, 127, 0, 255]))
        short = clear_value(GL_UNSIGNED_SHORT, 2, color)
        assert short is not None
        self.assertEqual((short.dtype, short.tolist()), (np.dtype(np.uint16), [65535, 32767]))
        half = clear_value(GL_HALF_FLOAT, 1, color)
        assert half is not None
        self.assertEqual(half.dtype, np.dtype(np.float16))
        floats = clear_value(GL_FLOAT, 3, color)
        assert floats is not None
        self.assertEqual((floats.dtype, floats.tolist()), (np.dtype(np.float32), [1.0, 0.5, 0.0]))

    def test_one_value_is_one_array_kept_for_good(self) -> None:
        # the driver may read the color after the call returns: the array is never freed or replaced
        self.assertIs(clear_value(GL_FLOAT, 4, (0.25, 0.0, 0.0, 1.0)), clear_value(GL_FLOAT, 4, (0.25, 0.0, 0.0, 1.0)))


if __name__ == '__main__':
    unittest.main()
