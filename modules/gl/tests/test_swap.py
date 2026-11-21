"""A swap pair is its current buffer; `back` is the other one, and they trade places at a swap: the texture pairs
(SwapFbo, SwapTexture3D, SwapTexture2DArray) and a SwapFbo's targets (GL not called)."""

import unittest

from OpenGL.GL import GL_NEAREST

from modules.gl import Fbo, SwapFbo, SwapTexture2DArray, SwapTexture3D, Texture2DArray, Texture3D


class SwapTest(unittest.TestCase):
    PAIRS = ((SwapFbo, Fbo), (SwapTexture3D, Texture3D), (SwapTexture2DArray, Texture2DArray))

    def test_the_pair_itself_is_always_the_current_buffer(self) -> None:
        for pair_class, buffer_class in self.PAIRS:
            with self.subTest(pair_class.__name__):
                pair = pair_class()
                pair._buffers[0].tex_id, pair._buffers[1].tex_id = 3, 7
                self.assertIsInstance(pair, buffer_class)
                for expected in (7, 3, 7):
                    pair.swap()
                    self.assertEqual(pair.tex_id, expected)

    def test_a_volume_pair_has_its_current_buffers_depth(self) -> None:
        for pair_class in (SwapTexture3D, SwapTexture2DArray):
            with self.subTest(pair_class.__name__):
                pair = pair_class()
                pair._buffers[0].depth, pair._buffers[1].depth = 16, 32
                pair.swap()
                self.assertEqual(pair.depth, 32)

    def test_back_is_the_other_buffer(self) -> None:
        for pair_class, buffer_class in self.PAIRS:
            with self.subTest(pair_class.__name__):
                pair = pair_class()
                pair._buffers[0].tex_id, pair._buffers[1].tex_id = 3, 7
                self.assertIsInstance(pair.back, buffer_class)
                self.assertEqual(pair.back.tex_id, 7)
                pair.swap()
                self.assertEqual((pair.tex_id, pair.back.tex_id), (7, 3))

    def test_a_kept_back_buffer_is_the_current_one_after_a_swap(self) -> None:
        pair = SwapFbo()
        back = pair.back
        pair.swap()
        self.assertIs(back, pair.current)

    def test_the_framebuffer_is_the_current_buffers(self) -> None:
        pair = SwapFbo()
        pair._buffers[0].fbo_id, pair._buffers[1].fbo_id = 11, 12
        self.assertEqual(pair.fbo_id, 11)
        pair.swap()
        self.assertEqual(pair.fbo_id, 12)

    def test_a_pair_has_no_texture_property(self) -> None:
        for pair_class, _ in self.PAIRS:
            with self.subTest(pair_class.__name__):
                self.assertFalse(hasattr(pair_class(), 'texture'))

    def test_a_target_of_a_framebuffer_pair_follows_its_current_framebuffer(self) -> None:
        pair = SwapFbo(extra_targets=(GL_NEAREST,))
        self.assertIs(pair.current.targets[0], pair.current)
        target = pair.target(1)
        first = pair.current.targets[1]
        self.assertIs(target.current, first)
        self.assertIs(target.back, pair.back.targets[1])
        pair.swap()
        self.assertIs(target.current, pair.current.targets[1])
        self.assertIs(target.back, first)
        with self.assertRaises(RuntimeError):
            target.swap()                           # it swaps with its pair, never alone


if __name__ == '__main__':
    unittest.main()
