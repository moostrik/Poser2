"""A SwapTexture3D's and a SwapTexture2DArray's buffers as handed out: valid until the owner's next
swap, then raising where they are used; the owner itself is always its current buffer."""

import unittest

from modules.gl import SwapTexture3D, SwapTexture2DArray, Texture3D, Texture2DArray, StaleTextureError


class SwapTextureVolumeTest(unittest.TestCase):
    OWNERS = ((SwapTexture3D, Texture3D), (SwapTexture2DArray, Texture2DArray))

    def test_a_texture_taken_before_a_swap_raises_after_it(self) -> None:
        for owner_class, _ in self.OWNERS:
            with self.subTest(owner_class.__name__):
                owner = owner_class()
                texture = owner.texture
                owner.swap()
                with self.assertRaises(StaleTextureError):
                    texture.tex_id
                with self.assertRaises(StaleTextureError):
                    texture.bind()

    def test_a_texture_taken_after_the_swap_does_not(self) -> None:
        for owner_class, _ in self.OWNERS:
            with self.subTest(owner_class.__name__):
                owner = owner_class()
                owner._textures[1].tex_id = 7
                owner.swap()
                self.assertEqual(owner.texture.tex_id, 7)

    def test_the_back_texture_expires_the_same_way(self) -> None:
        for owner_class, _ in self.OWNERS:
            with self.subTest(owner_class.__name__):
                owner = owner_class()
                owner._textures[1].tex_id = 7
                back = owner.back_texture
                self.assertEqual(back.tex_id, 7)
                owner.swap()
                with self.assertRaises(StaleTextureError):
                    back.tex_id

    def test_the_owner_itself_is_always_the_current_buffer(self) -> None:
        for owner_class, buffer_class in self.OWNERS:
            with self.subTest(owner_class.__name__):
                owner = owner_class()
                owner._textures[0].tex_id, owner._textures[1].tex_id = 3, 7
                self.assertIsInstance(owner, buffer_class)
                self.assertIsInstance(owner.texture, buffer_class)
                for expected in (7, 3, 7):
                    owner.swap()
                    self.assertEqual(owner.tex_id, expected)

    def test_sizes_and_formats_read_without_a_check(self) -> None:
        for owner_class, _ in self.OWNERS:
            with self.subTest(owner_class.__name__):
                owner = owner_class()
                texture = owner.texture
                owner.swap()
                self.assertEqual((texture.width, texture.height, texture.depth, texture.allocated), (0, 0, 0, False))


if __name__ == '__main__':
    unittest.main()
