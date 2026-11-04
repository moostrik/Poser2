"""A SwapFbo's buffers as handed out: valid until the owner's next swap, then raising where they are used."""

import unittest

from modules.gl import SwapFbo, StaleTextureError


class SwapTextureTest(unittest.TestCase):
    def test_a_texture_taken_before_a_swap_raises_after_it(self) -> None:
        fbo = SwapFbo()
        texture = fbo.texture
        fbo.swap()
        with self.assertRaises(StaleTextureError):
            texture.tex_id
        with self.assertRaises(StaleTextureError):
            texture.bind()

    def test_a_texture_taken_after_the_swap_does_not(self) -> None:
        fbo = SwapFbo()
        fbo.swap()
        texture = fbo.texture
        self.assertEqual(texture.tex_id, fbo._fbos[fbo._swap_state].tex_id)

    def test_the_back_texture_expires_the_same_way(self) -> None:
        fbo = SwapFbo()
        back = fbo.back_texture
        self.assertEqual(back.tex_id, fbo._fbos[1 - fbo._swap_state].tex_id)
        fbo.swap()
        with self.assertRaises(StaleTextureError):
            back.tex_id

    def test_the_swap_fbo_itself_is_always_the_current_buffer(self) -> None:
        fbo = SwapFbo()
        for _ in range(3):
            fbo.swap()
            self.assertEqual(fbo.tex_id, fbo._fbos[fbo._swap_state].tex_id)

    def test_sizes_and_formats_read_without_a_check(self) -> None:
        fbo = SwapFbo()
        texture = fbo.texture
        fbo.swap()
        self.assertEqual((texture.width, texture.height, texture.allocated), (0, 0, False))


if __name__ == '__main__':
    unittest.main()
