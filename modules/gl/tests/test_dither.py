"""A Dither holds its own blue noise: made once at its allocate, freed once at its deallocate (GL and noise mocked)."""

import sys
import unittest
from unittest import mock

from modules.gl import Shader, shaders

DITHER = sys.modules['modules.gl.shaders.color.dither']


class DitherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.noise = mock.MagicMock(return_value='noise')
        for patcher in (mock.patch.object(Shader, '_compile', classmethod(lambda cls, sources: 1)),
                        mock.patch.object(Shader, '_active_uniforms', staticmethod(lambda program: {})),
                        mock.patch.object(sys.modules['modules.gl.shader'], 'glDeleteProgram'),
                        mock.patch.object(DITHER, 'blue_noise', self.noise),
                        mock.patch.object(DITHER, 'Image', side_effect=lambda **_: mock.MagicMock())):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(Shader._programs.clear)

    def test_its_noise_is_made_once_and_freed_once(self) -> None:
        dither = shaders.Dither()
        dither.allocate()
        dither.allocate()
        self.noise.assert_called_once_with(64)
        dither.deallocate()
        dither.deallocate()
        dither._blue_noise.deallocate.assert_called_once()

    def test_each_dither_holds_its_own(self) -> None:
        first, second = shaders.Dither(), shaders.Dither()
        self.assertIsNot(first._blue_noise, second._blue_noise)
        first.allocate()
        second.allocate()
        first.deallocate()
        second._blue_noise.deallocate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
