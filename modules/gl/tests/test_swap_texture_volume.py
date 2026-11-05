"""A SwapTexture3D and a SwapTexture2DArray are their current buffer; `back` is the other one, and
they trade places at a swap."""

import unittest

from modules.gl import SwapTexture3D, SwapTexture2DArray, Texture3D, Texture2DArray


class SwapTextureVolumeTest(unittest.TestCase):
    OWNERS = ((SwapTexture3D, Texture3D), (SwapTexture2DArray, Texture2DArray))

    def test_the_owner_itself_is_always_the_current_buffer(self) -> None:
        for owner_class, buffer_class in self.OWNERS:
            with self.subTest(owner_class.__name__):
                owner = owner_class()
                owner._textures[0].tex_id, owner._textures[1].tex_id = 3, 7
                self.assertIsInstance(owner, buffer_class)
                for expected in (7, 3, 7):
                    owner.swap()
                    self.assertEqual(owner.tex_id, expected)

    def test_back_is_the_other_buffer(self) -> None:
        for owner_class, buffer_class in self.OWNERS:
            with self.subTest(owner_class.__name__):
                owner = owner_class()
                owner._textures[0].tex_id, owner._textures[1].tex_id = 3, 7
                self.assertIsInstance(owner.back, buffer_class)
                self.assertEqual(owner.back.tex_id, 7)
                owner.swap()
                self.assertEqual((owner.tex_id, owner.back.tex_id), (7, 3))

    def test_a_swap_texture_has_no_texture_property(self) -> None:
        for owner_class, _ in self.OWNERS:
            with self.subTest(owner_class.__name__):
                self.assertFalse(hasattr(owner_class(), 'texture'))


if __name__ == '__main__':
    unittest.main()
