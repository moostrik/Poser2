"""Shaders share their class's program and reload it at one point (GL and the compile mocked)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules.gl import ComputeShader, Shader

SHADER = sys.modules['modules.gl.shader.shader']
ACTIVE_UNIFORMS = Shader._active_uniforms           # the real one; the tests' setUp mocks it
SOURCE: str = '#version 460\nvoid main() {}\n'


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

        self.active: dict[str, int] = {}
        for patcher in (mock.patch.object(Shader, '_compile', classmethod(compile)),
                        mock.patch.object(Shader, '_active_uniforms', staticmethod(lambda program: dict(self.active))),
                        mock.patch.object(SHADER, 'glDeleteProgram'),
                        mock.patch.object(SHADER, 'SETTLE', 0.0),           # F5's own tests set them
                        mock.patch.object(SHADER, 'READ_RETRIES', 0)):
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

    def test_an_edited_include_marks_every_shader_that_uses_it(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'common.glsl').write_text('float shared() { return 1.0; }\n')
            for name in ('a', 'b', 'c'):
                include = '#include "common.glsl"\n' if name != 'c' else ''
                (root / f'{name}.frag').write_text(f'#version 460\n{include}void main() {{}}\n')

            def shader_of(name: str) -> type:
                return type(f'_{name.upper()}', (Shader,), {'_stages': classmethod(
                    lambda cls, name=name: [(0, root / f'{name}.frag')])})

            shaders = [shader_of(name)() for name in ('a', 'b', 'c')]
            for shader in shaders:
                shader.allocate()
            self.compiles.clear()
            (root / 'common.glsl').write_text('float shared() { return 2.0; }\n')
            Shader._on_file_changed(str(root / 'common.glsl'))
            Shader.reload_marked()
        self.assertEqual(sorted(c.__name__ for c in self.compiles), ['_A', '_B'])

    def test_a_file_only_read_marks_nothing_and_a_save_marks_it(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.frag'
            path.write_text(SOURCE)
            self.shader_on(path)
            path.read_text()
            Shader._on_file_changed(str(path))              # Windows: the read's last access time, reported
            Shader.reload_marked()
            self.assertEqual(self.compiles, [])
            path.write_text(SOURCE + '// saved\n')
            Shader._on_file_changed(str(path))
            Shader.reload_marked()
        self.assertEqual(len(self.compiles), 1)

    def test_an_error_in_an_include_names_that_file_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'common.glsl').write_text('float a;\nfloat b = oops;\n')
            (root / 'main.frag').write_text('#version 460\n#include "common.glsl"\nvoid main() { bad; }\n')
            read: list[Path] = []
            source: str = Shader.read_shader_source(str(root / 'main.frag'), read)
        self.assertEqual([path.name for path in read], ['main.frag', 'common.glsl'])
        self.assertEqual(source.splitlines(), ['#version 460', '#line 1 1', 'float a;', 'float b = oops;', '#line 3 0',
                                               'void main() { bad; }'])
        log: str = ('1(2) : error C1008: undefined variable "oops"\n0(3) : error C1008: undefined variable "bad"\n'
                    'a line without a location')
        self.assertEqual(SHADER.name_error_lines(log, read).splitlines(),
                         ['common.glsl(2) : error C1008: undefined variable "oops"',
                          'main.frag(3) : error C1008: undefined variable "bad"', 'a line without a location'])

    def test_the_uniforms_are_read_after_linking_and_a_misspelled_one_warns_once(self) -> None:
        self.active = {'gain': 4}
        draw = _Draw()
        draw.allocate()
        with mock.patch.object(SHADER, 'glGetUniformLocation', return_value=-1) as lookup:
            self.assertEqual(draw.get_uniform_loc('gain'), 4)
            lookup.assert_not_called()                     # read once, after linking
            with self.assertLogs(SHADER.logger, 'WARNING') as logs:
                self.assertEqual(draw.get_uniform_loc('gian'), -1)
                self.assertEqual(draw.get_uniform_loc('gian'), -1)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(lookup.call_count, 1)

    def test_an_array_element_is_looked_up_once_and_does_not_warn(self) -> None:
        draw = _Draw()
        draw.allocate()
        with mock.patch.object(SHADER, 'glGetUniformLocation', return_value=9) as lookup,                 mock.patch.object(SHADER.logger, 'warning') as warning:
            self.assertEqual(draw.get_uniform_loc('colors[2]'), 9)
            self.assertEqual(draw.get_uniform_loc('colors[2]'), 9)
        self.assertEqual(lookup.call_count, 1)
        warning.assert_not_called()

    def test_active_uniforms_name_an_array_by_its_first_element_and_leave_out_block_members(self) -> None:
        names = {0: (b'gain', 1, 0), 1: (b'colors[0]', 4, 0), 2: (b'Block.member', 1, 0)}
        locations = {'gain': 3, 'colors[0]': 5, 'Block.member': -1}
        with mock.patch.object(SHADER, 'glGetProgramiv', return_value=3),                 mock.patch.object(SHADER, 'glGetActiveUniform', side_effect=lambda program, index: names[index]),                 mock.patch.object(SHADER, 'glGetUniformLocation', side_effect=lambda program, name: locations[name]):
            uniforms = ACTIVE_UNIFORMS(7)
        self.assertEqual(uniforms, {'gain': 3, 'colors[0]': 5, 'colors': 5})

    def shader_on(self, path: Path) -> Shader:
        shader_class = type('_File', (Shader,), {'_stages': classmethod(lambda cls: [(0, path)])})
        shader: Shader = shader_class()
        shader.allocate()
        self.compiles.clear()
        return shader

    def test_a_save_in_two_events_compiles_once_after_it_settles(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.frag'
            path.write_text(SOURCE)
            self.shader_on(path)
            path.write_text(SOURCE + '// saved\n')
            clock = [10.0]
            with mock.patch.object(SHADER, 'SETTLE', 0.1), mock.patch.object(SHADER, 'perf_counter', lambda: clock[0]):
                Shader._on_file_changed(str(path))
                clock[0] = 10.05
                Shader._on_file_changed(str(path))          # the save's second event
                Shader.reload_marked()
                self.assertEqual(self.compiles, [])         # not settled
                clock[0] = 10.2
                Shader.reload_marked()
                Shader.reload_marked()
        self.assertEqual(len(self.compiles), 1)

    def test_a_half_written_file_is_tried_again_not_reported(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.frag'
            path.write_text(SOURCE)
            self.shader_on(path)
            with mock.patch.object(SHADER, 'READ_RETRIES', 30), mock.patch.object(SHADER.logger, 'error') as error:
                path.write_text('')                         # the editor has truncated it, not yet written
                Shader._on_file_changed(str(path))
                Shader.reload_marked()
                self.assertEqual(self.compiles, [])
                path.write_text(SOURCE)
                Shader.reload_marked()
            error.assert_not_called()
        self.assertEqual(len(self.compiles), 1)

    def test_two_files_saved_together_both_reload(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / 'a.frag', Path(folder) / 'b.frag'
            for path in (first, second):
                path.write_text(SOURCE)
            a, b = self.shader_on(first), self.shader_on(second)
            for path in (first, second):
                path.write_text(SOURCE + '// saved\n')
            Shader._on_file_changed(str(first))
            Shader._on_file_changed(str(second))            # within the same moment: nothing dropped
            Shader.reload_marked()
        self.assertEqual(sorted(c.__name__ for c in self.compiles), ['_File', '_File'])

    def test_a_compute_shader_has_one_stage_its_comp_file(self) -> None:
        stages = _Compute._stages()
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0][1].name, 'test_shader.comp')

    def test_without_its_own_vert_a_shader_takes_the_shared_one(self) -> None:
        self.assertEqual(_Draw._stages()[0][1], SHADER.SHARED_SHADERS / '_generic.vert')


if __name__ == '__main__':
    unittest.main()
