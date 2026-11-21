"""The three texture kinds clear and read through one base, each on its own target and shape, and a ping-pong pair
clears the buffer it is asked to (GL mocked)."""

import sys
import unittest
from unittest import mock

from OpenGL.GL import GL_FLOAT, GL_R16F, GL_RED, GL_RGBA, GL_RGBA16F, GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_3D

from modules.gl import SwapFbo, SwapTexture3D, Texture, Texture2DArray, Texture3D

BASE = sys.modules['modules.gl.texture_base']


def allocated(texture, tex_id: int, internal_format: int, fmt: int, depth: int = 0):
    """A texture as `allocate` leaves it, without GL: 4 × 2 (× `depth`)."""
    texture.allocated, texture.tex_id, texture.width, texture.height = True, tex_id, 4, 2
    texture.internal_format, texture.format, texture.data_type = internal_format, fmt, int(GL_FLOAT)
    if depth:
        texture.depth = depth
    return texture


class TextureBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gl = mock.MagicMock()
        for name in ('glClearTexImage', 'glGetTexImage', 'glBindTexture'):
            patcher = mock.patch.object(BASE, name, getattr(self.gl, name))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_each_kind_clears_its_own_texture_with_the_kept_color(self) -> None:
        for texture in (allocated(Texture(), 5, GL_RGBA16F, GL_RGBA), allocated(Texture3D(), 6, GL_RGBA16F, GL_RGBA, 3),
                        allocated(Texture2DArray(), 7, GL_RGBA16F, GL_RGBA, 3)):
            with self.subTest(type(texture).__name__):
                self.gl.glClearTexImage.reset_mock()
                texture.clear(0.5, 0.0, 0.0, 1.0)
                tex_id, level, fmt, data_type, color = self.gl.glClearTexImage.call_args.args
                self.assertEqual((tex_id, level, fmt, data_type), (texture.tex_id, 0, GL_RGBA, GL_FLOAT))
                self.assertEqual(list(color), [0.5, 0.0, 0.0, 1.0])

    def test_a_texture_not_allocated_is_not_cleared(self) -> None:
        Texture3D().clear(1.0)
        self.gl.glClearTexImage.assert_not_called()

    def test_each_kind_reads_on_its_target_in_its_shape(self) -> None:
        cases = ((allocated(Texture(), 5, GL_RGBA16F, GL_RGBA), GL_TEXTURE_2D, (2, 4, 4)),
                 (allocated(Texture3D(), 6, GL_R16F, GL_RED, 3), GL_TEXTURE_3D, (3, 2, 4)),
                 (allocated(Texture2DArray(), 7, GL_RGBA16F, GL_RGBA, 3), GL_TEXTURE_2D_ARRAY, (3, 2, 4, 4)))
        for texture, target, shape in cases:
            with self.subTest(type(texture).__name__):
                self.gl.glGetTexImage.reset_mock()
                pixels = texture.read_to_numpy()
                assert pixels is not None
                self.assertEqual(pixels.shape, shape)
                self.assertEqual(self.gl.glGetTexImage.call_args.args[0], target)

    def test_a_2d_read_is_flipped_to_the_images_rows(self) -> None:
        texture = allocated(Texture(), 5, GL_R16F, GL_RED)
        self.gl.glGetTexImage.side_effect = lambda target, level, fmt, data_type, pixels: pixels.__setitem__(0, 1.0)
        rows = texture.read_to_numpy()
        assert rows is not None
        self.assertEqual((rows[0, 0], rows[-1, 0]), (0.0, 1.0))

    def test_a_pair_clears_its_current_its_back_or_both(self) -> None:
        for pair in (SwapFbo(), SwapTexture3D()):
            with self.subTest(type(pair).__name__):
                first, second = pair._buffers
                allocated(first, 3, GL_RGBA16F, GL_RGBA, 2)
                allocated(second, 7, GL_RGBA16F, GL_RGBA, 2)
                for clear, cleared in ((pair.clear, [3]), (pair.clear_back, [7]), (pair.clear_all, [3, 7])):
                    self.gl.glClearTexImage.reset_mock()
                    clear(1.0)
                    self.assertEqual([call.args[0] for call in self.gl.glClearTexImage.call_args_list], cleared)


if __name__ == '__main__':
    unittest.main()
