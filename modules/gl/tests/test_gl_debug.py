"""GL's debug messages are logged at the call that caused them; a notification is left out (GL mocked)."""

import unittest
from unittest import mock

from OpenGL.GL import GL_DEBUG_OUTPUT, GL_DEBUG_OUTPUT_SYNCHRONOUS, GL_DEBUG_SEVERITY_HIGH, GL_DEBUG_SEVERITY_MEDIUM, \
    GL_DEBUG_SEVERITY_NOTIFICATION

import modules.gl.gl_debug as GL_DEBUG


def message(text: str) -> tuple[bytes, int]:
    return text.encode(), len(text)


class GlDebugTest(unittest.TestCase):
    def test_an_error_is_logged_with_its_stack(self) -> None:
        buffer, length = message('GL_INVALID_OPERATION in glDrawArrays')
        with self.assertLogs(GL_DEBUG.logger, 'ERROR') as logs:
            GL_DEBUG.log_gl_message(0, 0, 1282, GL_DEBUG_SEVERITY_HIGH, length, buffer, None)
        self.assertIn('GL_INVALID_OPERATION in glDrawArrays', logs.output[0])
        self.assertIsNotNone(logs.records[0].stack_info)

    def test_a_lesser_message_warns_and_a_notification_is_left_out(self) -> None:
        buffer, length = message('performance: a buffer moved')
        with self.assertLogs(GL_DEBUG.logger, 'WARNING') as logs:
            GL_DEBUG.log_gl_message(0, 0, 7, GL_DEBUG_SEVERITY_MEDIUM, length, buffer, None)
            GL_DEBUG.log_gl_message(0, 0, 8, GL_DEBUG_SEVERITY_NOTIFICATION, length, buffer, None)
        self.assertEqual(len(logs.records), 1)

    def test_enabling_turns_on_synchronous_output_with_the_kept_callback(self) -> None:
        with mock.patch.object(GL_DEBUG, 'glEnable') as enable, \
                mock.patch.object(GL_DEBUG, 'glDebugMessageCallback') as register:
            GL_DEBUG.enable_debug_output()
        self.assertEqual([call.args[0] for call in enable.call_args_list], [GL_DEBUG_OUTPUT, GL_DEBUG_OUTPUT_SYNCHRONOUS])
        register.assert_called_once_with(GL_DEBUG._callback, None)


if __name__ == '__main__':
    unittest.main()
