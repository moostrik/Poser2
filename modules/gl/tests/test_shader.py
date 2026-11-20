"""Shaders share their class's program and reload it at one point (GL and the compile mocked)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

from modules.gl import ComputeShader, Shader

SHADER = sys.modules['modules.gl.shader']


class _Draw(Shader):
    """A shader of this test: its files need not exist, the compile is mocked."""


class _Compute(ComputeShader):
    """A compute shader of this test."""


class ShaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = iter(range(100, 200))
        self.compiles: list[type] = []
        self.failing: bool = False

        def compile(cls, stages):
            self.compiles.append(cls)
            return None if self.failing else next(self.ids)

        for patcher in (mock.patch.object(Shader, '_compile', classmethod(compile)),
                        mock.patch.object(SHADER, 'glDeleteProgram')):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.deleted = SHADER.glDeleteProgram
        self.addCleanup(Shader._programs.clear)

    def test_objects_of_one_class_share_its_program(self) -> None:
        first, second = _Draw(), _Draw()
        first.allocate()
        second.allocate()
        self.assertEqual(self.compiles, [_Draw])
        self.assertEqual(first.shader_program, second.shader_program)
        first.deallocate()
        self.deleted.assert_not_called()                   # the other still uses it
        second.deallocate()
        self.deleted.assert_called_once_with(100)
        self.assertIsNone(first.shader_program)

    def test_a_marked_shader_of_each_kind_is_reloaded_once_at_the_next_frame(self) -> None:
        draw, compute = _Draw(), _Compute()
        draw.allocate()
        compute.allocate()
        self.compiles.clear()
        for shader_class in (_Draw, _Compute):
            for _, path in shader_class._stages():
                Shader._on_file_changed(str(path))         # the watcher's thread
        self.assertEqual(self.compiles, [])                # nothing compiles off the frame
        Shader.reload_marked()
        Shader.reload_marked()
        self.assertEqual(sorted(c.__name__ for c in self.compiles), ['_Compute', '_Draw'])
        self.assertEqual((draw.shader_program, compute.shader_program), (102, 103))

    def test_a_failed_reload_keeps_the_program(self) -> None:
        draw = _Draw()
        draw.allocate()
        self.failing = True
        Shader._on_file_changed(str(_Draw._stages()[1][1]))
        with self.assertLogs(SHADER.logger, 'ERROR'):
            Shader.reload_marked()
        self.assertEqual(draw.shader_program, 100)

    def test_an_unrelated_file_marks_nothing(self) -> None:
        _Draw().allocate()
        self.compiles.clear()
        Shader._on_file_changed(str(Path(__file__).with_name('other.frag')))
        Shader.reload_marked()
        self.assertEqual(self.compiles, [])

    def test_a_compute_shader_has_one_stage_its_comp_file(self) -> None:
        stages = _Compute._stages()
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0][1].name, 'test_shader.comp')

    def test_without_its_own_vert_a_shader_takes_the_shared_one(self) -> None:
        self.assertEqual(_Draw._stages()[0][1], SHADER.SHARED_SHADERS / '_generic.vert')


if __name__ == '__main__':
    unittest.main()
