"""A SwapFbo is its current buffer; `back` is the other one, and they trade places at a swap."""

import unittest

from modules.gl import SwapFbo


class SwapFboTest(unittest.TestCase):
    def test_the_swap_fbo_itself_is_always_the_current_buffer(self) -> None:
        fbo = SwapFbo()
        fbo._fbos[0].tex_id, fbo._fbos[1].tex_id = 3, 7
        for expected in (7, 3, 7):
            fbo.swap()
            self.assertEqual(fbo.tex_id, expected)

    def test_back_is_the_other_buffer(self) -> None:
        fbo = SwapFbo()
        fbo._fbos[0].tex_id, fbo._fbos[1].tex_id = 3, 7
        self.assertEqual(fbo.back.tex_id, 7)
        fbo.swap()
        self.assertEqual((fbo.tex_id, fbo.back.tex_id), (7, 3))

    def test_a_kept_back_buffer_is_the_current_one_after_a_swap(self) -> None:
        fbo = SwapFbo()
        back = fbo.back
        fbo.swap()
        self.assertIs(back, fbo._fbos[fbo._swap_state])

    def test_a_swap_fbo_has_no_texture_property(self) -> None:
        self.assertFalse(hasattr(SwapFbo(), 'texture'))


if __name__ == '__main__':
    unittest.main()
