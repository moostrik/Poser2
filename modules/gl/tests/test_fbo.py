"""An Fbo frees what it makes: each framebuffer and renderbuffer it generates is deleted again (GL mocked)."""

import sys
import unittest
from unittest import mock

from OpenGL.GL import GL_RGBA8

from modules.gl import Fbo

FBO = sys.modules['modules.gl.texture.fbo']
TEXTURE = sys.modules['modules.gl.texture.texture']


class FboTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gl = mock.MagicMock()
        self.gl.glGenFramebuffers.side_effect = [11, 12, 13]
        self.gl.glGenRenderbuffers.side_effect = [21, 22, 23]
        for name in ('glGenFramebuffers', 'glDeleteFramebuffers', 'glGenRenderbuffers', 'glDeleteRenderbuffers',
                     'glBindFramebuffer', 'glBindRenderbuffer', 'glFramebufferTexture2D', 'glFramebufferRenderbuffer',
                     'glRenderbufferStorage', 'glClear', 'glClearColor'):
            patcher = mock.patch.object(FBO, name, getattr(self.gl, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        # the texture under the framebuffer: allocated without GL
        allocate = mock.patch.object(TEXTURE.Texture, 'allocate', lambda tex, w, h, f: setattr(tex, 'allocated', True))
        deallocate = mock.patch.object(TEXTURE.Texture, 'deallocate', lambda tex: setattr(tex, 'allocated', False))
        for patcher in (allocate, deallocate):
            patcher.start()
            self.addCleanup(patcher.stop)

    def deleted(self, name: str) -> list[int]:
        return [call.args[1][0] for call in getattr(self.gl, name).call_args_list]

    def test_each_framebuffer_made_is_deleted(self) -> None:
        fbo = Fbo()
        for _ in range(3):                                   # a resize reallocates
            fbo.allocate(64, 32, GL_RGBA8)
            fbo.deallocate()
        self.assertEqual(self.deleted('glDeleteFramebuffers'), [11, 12, 13])
        self.assertEqual(fbo.fbo_id, 0)

    def test_a_depth_buffer_is_deleted_with_it(self) -> None:
        fbo = Fbo(depth=True)
        fbo.allocate(64, 32, GL_RGBA8)
        fbo.deallocate()
        self.assertEqual(self.deleted('glDeleteFramebuffers'), [11])
        self.assertEqual(self.deleted('glDeleteRenderbuffers'), [21])

    def test_a_framebuffer_never_made_is_not_deleted(self) -> None:
        Fbo().deallocate()
        self.gl.glDeleteFramebuffers.assert_not_called()


if __name__ == '__main__':
    unittest.main()
