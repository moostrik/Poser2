"""BlendState keeps the blend state on the application side: a repeated call costs no GL call, a buffer's own mode is kept
until every buffer is set again, and push and pop restore a state with only the calls that differ (GL mocked)."""

import sys
import unittest
from unittest import mock

from modules.gl import BlendState

BLEND_STATE = sys.modules['modules.gl.state.blend_state']


class BlendStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gl = mock.MagicMock()
        for name in ('glEnable', 'glDisable', 'glBlendFunc', 'glBlendFunci'):
            patcher = mock.patch.object(BLEND_STATE, name, getattr(self.gl, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.stack = BLEND_STATE.BlendStack()
        previous = BLEND_STATE._stack
        BlendState.use(self.stack)
        self.addCleanup(BlendState.use, previous)

    def calls(self) -> list[str]:
        return [call[0] for call in self.gl.method_calls]

    def test_a_repeated_blend_mode_costs_no_gl_call(self) -> None:
        BlendState.set_blend_mode(BlendState.BlendMode.ADD)
        self.assertEqual(self.calls(), ['glEnable', 'glBlendFunc'])
        self.gl.reset_mock()
        BlendState.set_blend_mode(BlendState.BlendMode.ADD)
        self.assertEqual(self.calls(), [])
        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)
        self.assertEqual(self.calls(), ['glDisable', 'glBlendFunc'])

    def test_a_buffers_own_mode_holds_until_every_buffer_is_set(self) -> None:
        BlendState.set_blend_mode(BlendState.BlendMode.ALPHA)
        BlendState.set_buffer_blend_mode(1, BlendState.BlendMode.REVEALAGE)
        BlendState.set_buffer_blend_mode(1, BlendState.BlendMode.REVEALAGE)
        self.assertEqual(self.calls().count('glBlendFunci'), 1)
        self.gl.reset_mock()
        BlendState.set_blend_mode(BlendState.BlendMode.ALPHA)     # the same mode, but buffer 1 differs: set again
        self.assertEqual(self.calls(), ['glBlendFunc'])
        self.assertEqual(self.stack.current.buffer_modes, {})

    def test_push_and_pop_restore_with_only_the_calls_that_differ(self) -> None:
        BlendState.set_blend_mode(BlendState.BlendMode.ALPHA)
        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.ADD)
        self.gl.reset_mock()
        BlendState.pop()
        self.assertEqual(self.calls(), ['glBlendFunc'])  # blending stays on: no enable
        self.assertEqual(self.stack.current.blend_mode, BlendState.BlendMode.ALPHA)

    def test_a_buffers_equation_is_set_once_and_restored_by_pop(self) -> None:
        for name in ('glBlendEquationi',):
            patcher = mock.patch.object(BLEND_STATE, name, getattr(self.gl, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        BlendState.push()
        BlendState.set_buffer_blend_equation(0, BlendState.BlendEquation.MIN)
        BlendState.set_buffer_blend_equation(0, BlendState.BlendEquation.MIN)
        self.assertEqual(self.calls(), ['glBlendEquationi'])
        self.gl.reset_mock()
        BlendState.pop()
        self.gl.glBlendEquationi.assert_called_once_with(0, BlendState.BlendEquation.ADD.value)
        self.assertEqual(self.stack.current.buffer_equations, {})

    def test_a_pop_without_a_push_warns_and_keeps_the_state(self) -> None:
        BlendState.set_blend_mode(BlendState.BlendMode.ADD)
        with self.assertLogs(BLEND_STATE.logger, 'WARNING'):
            BlendState.pop()
        self.assertEqual(self.stack.current.blend_mode, BlendState.BlendMode.ADD)


if __name__ == '__main__':
    unittest.main()
