"""The one format table gives what the three tables it replaced gave (Texture's, Image's and Tensor's), for each
format each of them handled; the rows they lacked are the table's own."""

import unittest

from OpenGL.GL import (GL_BGR, GL_BGRA, GL_FLOAT, GL_HALF_FLOAT, GL_NONE, GL_R16, GL_R16F, GL_R32F, GL_R32UI, GL_R8, GL_RED,
                       GL_RED_INTEGER, GL_RG, GL_RG16, GL_RG16F, GL_RG32F, GL_RG8, GL_RGB, GL_RGB10_A2, GL_RGB16F,
                       GL_RGB32F, GL_RGB8, GL_RGBA, GL_RGBA16F, GL_RGBA32F, GL_RGBA8, GL_UNSIGNED_BYTE, GL_UNSIGNED_INT,
                       GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_SHORT)

from modules.gl.texture.texture_format import bytes_per_channel, channel_count, data_type, pixel_format, tensor_type

# Texture's get_format and get_data_type, each format it handled
TEXTURE = {
    GL_RGB: (GL_RGB, GL_UNSIGNED_BYTE), GL_RGB8: (GL_RGB, GL_UNSIGNED_BYTE), GL_RGB16F: (GL_RGB, GL_FLOAT),
    GL_RGB32F: (GL_RGB, GL_FLOAT), GL_RGBA: (GL_RGBA, GL_UNSIGNED_BYTE), GL_RGBA8: (GL_RGBA, GL_UNSIGNED_BYTE),
    GL_RGB10_A2: (GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV), GL_RGBA16F: (GL_RGBA, GL_FLOAT),
    GL_RGBA32F: (GL_RGBA, GL_FLOAT), GL_R8: (GL_RED, GL_UNSIGNED_BYTE), GL_R16: (GL_RED, GL_UNSIGNED_SHORT),
    GL_R16F: (GL_RED, GL_FLOAT), GL_R32F: (GL_RED, GL_FLOAT), GL_R32UI: (GL_RED_INTEGER, GL_UNSIGNED_INT),
    GL_RG8: (GL_RG, GL_UNSIGNED_BYTE), GL_RG16: (GL_RG, GL_UNSIGNED_SHORT), GL_RG16F: (GL_RG, GL_FLOAT),
    GL_RG32F: (GL_RG, GL_FLOAT),
}
# Image's _get_format_bgr, RGB and BGR order
IMAGE = {
    GL_RGB: (GL_RGB, GL_BGR), GL_RGB8: (GL_RGB, GL_BGR), GL_RGB16F: (GL_RGB, GL_BGR), GL_RGB32F: (GL_RGB, GL_BGR),
    GL_RGBA: (GL_RGBA, GL_BGRA), GL_RGBA8: (GL_RGBA, GL_BGRA), GL_RGBA16F: (GL_RGBA, GL_BGRA),
    GL_RGBA32F: (GL_RGBA, GL_BGRA), GL_R8: (GL_RED, GL_RED), GL_R16F: (GL_RED, GL_RED), GL_R32F: (GL_RED, GL_RED),
    GL_RG8: (GL_RG, GL_RG), GL_RG16F: (GL_RG, GL_RG), GL_RG32F: (GL_RG, GL_RG),
}
# Tensor's channel count, bytes per channel and upload type
TENSOR = {
    GL_R8: (1, 1, GL_UNSIGNED_BYTE), GL_RG8: (2, 1, GL_UNSIGNED_BYTE), GL_RGB8: (3, 1, GL_UNSIGNED_BYTE),
    GL_RGBA8: (4, 1, GL_UNSIGNED_BYTE), GL_R16F: (1, 2, GL_HALF_FLOAT), GL_RG16F: (2, 2, GL_HALF_FLOAT),
    GL_RGB16F: (3, 2, GL_HALF_FLOAT), GL_RGBA16F: (4, 2, GL_HALF_FLOAT), GL_R32F: (1, 4, GL_FLOAT),
    GL_RG32F: (2, 4, GL_FLOAT), GL_RGB32F: (3, 4, GL_FLOAT), GL_RGBA32F: (4, 4, GL_FLOAT),
}


class TextureFormatTest(unittest.TestCase):
    def test_texture_gets_what_it_got(self) -> None:
        for internal, (pixel, kind) in TEXTURE.items():
            self.assertEqual((pixel_format(internal), data_type(internal)), (pixel, kind), internal)

    def test_image_gets_what_it_got_in_either_channel_order(self) -> None:
        for internal, (rgb, bgr) in IMAGE.items():
            self.assertEqual((pixel_format(internal), pixel_format(internal, bgr=True)), (rgb, bgr), internal)

    def test_tensor_gets_what_it_got(self) -> None:
        for internal, (channels, size, kind) in TENSOR.items():
            self.assertEqual((channel_count(internal), bytes_per_channel(internal), tensor_type(internal)),
                             (channels, size, kind), internal)

    def test_an_unknown_format_warns_and_falls_back(self) -> None:
        with self.assertLogs('modules.gl.texture.texture_format', 'WARNING'):
            self.assertEqual(pixel_format(12345), GL_NONE)


if __name__ == '__main__':
    unittest.main()
