"""The settings store: each root group saved and loaded on its own, the startup files, names and bad files."""

import json
import tempfile
import unittest
from pathlib import Path

from modules.settings import BaseSettings, Field, Group, SettingsStore


class Camera(BaseSettings):
    exposure: Field[int] = Field(1000)
    resolution: Field[int] = Field(1080, access=Field.INIT)
    fps: Field[float] = Field(30.0, access=Field.INIT)


class Setup(BaseSettings):
    num_cams: Field[int] = Field(1)
    input_fps: Field[float] = Field(30.0, access=Field.INIT)
    camera: Group[Camera] = Group(Camera, share=[input_fps.as_('fps')])


class Motion(BaseSettings):
    gain: Field[float] = Field(1.0)
    fps: Field[float] = Field(30.0, access=Field.INIT)


class Flow(BaseSettings):
    input_fps: Field[float] = Field(30.0, access=Field.INIT)
    motion: Group[Motion] = Group(Motion, share=[input_fps.as_('fps')])


class Root(BaseSettings):
    input_fps: Field[float] = Field(30.0, access=Field.INIT)
    setup: Group[Setup] = Group(Setup, share=[input_fps])
    flow: Group[Flow] = Group(Flow, share=[input_fps])


class StoreTest(unittest.TestCase):

    def setUp(self) -> None:
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = SettingsStore(self.folder)

    def read(self, group: str, name: str) -> dict:
        return json.loads(self.store.path(group, name).read_text(encoding='utf-8'))


class TestFiles(StoreTest):

    def test_the_setup_file_holds_the_root_fields_beside_setup(self) -> None:
        root = Root(input_fps=60.0)
        root.setup.camera.exposure = 3000
        self.store.save(root, 'setup', 'studio')
        data = self.read('setup', 'studio')
        self.assertEqual(list(data), ['input_fps', 'setup'])
        self.assertEqual(data['input_fps'], 60.0)
        self.assertEqual(data['setup']['camera']['exposure'], 3000)
        self.assertNotIn('input_fps', data['setup'])            # shared down: its parent's

    def test_another_root_groups_file_holds_that_group(self) -> None:
        root = Root()
        root.flow.motion.gain = 2.0
        self.store.save(root, 'flow', 'calm')
        self.assertEqual(self.read('flow', 'calm'), {'motion': {'gain': 2.0}})

    def test_each_root_group_round_trips(self) -> None:
        root = Root(input_fps=60.0)
        root.setup.camera.exposure = 3000
        root.flow.motion.gain = 2.0
        self.store.save(root, 'setup', 'studio')
        self.store.save(root, 'flow', 'calm')
        other = Root()
        self.assertTrue(self.store.load(other, 'setup', 'studio'))
        self.assertTrue(self.store.load(other, 'flow', 'calm'))
        self.assertEqual(other.to_dict(), root.to_dict())
        self.assertEqual(other.flow.motion.fps, 60.0)            # the root's field shared into the flow

    def test_loading_one_root_group_leaves_the_others_as_they_were(self) -> None:
        root = Root(input_fps=60.0)
        root.flow.motion.gain = 2.0
        self.store.save(root, 'flow', 'calm')
        other = Root(input_fps=50.0)
        other.setup.camera.exposure = 4000
        other.setup.num_cams = 2
        before = other.to_dict()
        self.store.load(other, 'flow', 'calm')
        after = other.to_dict()
        self.assertEqual(after['flow'], {'motion': {'gain': 2.0}})
        self.assertEqual({k: v for k, v in after.items() if k != 'flow'}, {k: v for k, v in before.items() if k != 'flow'})

    def test_loading_fires_callbacks(self) -> None:
        root = Root()
        root.setup.camera.exposure = 5000
        self.store.save(root, 'setup', 'studio')
        root.setup.camera.exposure = 1000
        seen: list[int] = []
        root.setup.camera.bind(Camera.exposure, seen.append)
        self.store.load(root, 'setup', 'studio')
        self.assertEqual(seen, [5000])

    def test_a_save_replaces_the_file_and_leaves_no_temporary_one(self) -> None:
        root = Root()
        root.setup.camera.exposure = 4242
        self.store.save(root, 'setup', 'studio')
        root.setup.camera.exposure = 9999
        self.store.save(root, 'setup', 'studio')
        self.assertEqual(self.read('setup', 'studio')['setup']['camera']['exposure'], 9999)
        self.assertEqual([p.name for p in (self.folder / 'setup').iterdir()], ['studio.json'])

    def test_a_missing_or_unreadable_file_loads_nothing(self) -> None:
        root = Root()
        with self.assertLogs('modules.settings.store', 'WARNING'):
            self.assertFalse(self.store.load(root, 'flow', 'missing'))
        path = self.store.path('flow', 'bad')
        path.parent.mkdir(parents=True)
        path.write_text('{corrupt json!!', encoding='utf-8')
        with self.assertLogs('modules.settings.store', 'WARNING'):
            self.assertFalse(self.store.load(root, 'flow', 'bad'))
        self.store.path('flow', 'list').write_text('[1, 2]', encoding='utf-8')
        self.store.path('setup', 'list').parent.mkdir(parents=True)
        self.store.path('setup', 'list').write_text('[1, 2]', encoding='utf-8')
        for group in ('flow', 'setup'):
            with self.subTest(group=group), self.assertLogs('modules.settings.store', 'WARNING'):
                self.assertFalse(self.store.load(root, group, 'list'))
        self.assertEqual(root.to_dict(), Root().to_dict())

    def test_scan_lists_a_root_groups_names_sorted(self) -> None:
        root = Root()
        for name in ('beta', 'alpha'):
            self.store.save(root, 'flow', name)
        self.store.save(root, 'setup', 'studio')
        (self.folder / 'flow' / '.hidden.json').write_text('{}', encoding='utf-8')
        self.assertEqual(self.store.scan('flow'), ['alpha', 'beta'])
        self.assertEqual(self.store.scan('setup'), ['studio'])
        self.assertEqual(self.store.scan('other'), [])

    def test_delete(self) -> None:
        self.store.save(Root(), 'flow', 'calm')
        self.store.delete('flow', 'calm')
        self.assertEqual(self.store.scan('flow'), [])


class TestNames(StoreTest):

    def test_a_safe_name_passes(self) -> None:
        for name in ('studio', 'very nice', 'take-1', 'Smile'):
            self.store.validate_name(name)

    def test_an_unsafe_name_raises(self) -> None:
        for name in ('', 'foo/bar', 'foo\\bar', '.hidden', '../etc/passwd'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.validate_name(name)

    def test_a_save_under_an_unsafe_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save(Root(), 'flow', '../out')


class TestStartup(StoreTest):

    def test_each_root_group_starts_on_the_default_name_until_one_is_set(self) -> None:
        self.assertEqual(self.store.startup('setup'), 'studio')
        self.store.set_startup('flow', 'calm')
        self.assertEqual(self.store.startup('flow'), 'calm')
        self.assertEqual(self.store.startup('setup'), 'studio')
        self.assertEqual(SettingsStore(self.folder, 'Smile').startup('setup'), 'Smile')

    def test_set_startup_rejects_an_unsafe_name(self) -> None:
        with self.assertRaises(ValueError):
            self.store.set_startup('flow', '../etc/passwd')

    def test_load_startup_loads_the_setup_then_the_others(self) -> None:
        root = Root(input_fps=60.0)
        root.setup.camera.exposure = 3000
        root.flow.motion.gain = 2.0
        self.store.save(root, 'setup', 'studio')
        self.store.save(root, 'flow', 'calm')
        self.store.set_startup('flow', 'calm')
        other = Root()
        self.store.load_startup(other)
        self.assertEqual(other.to_dict(), root.to_dict())

    def test_without_the_setup_file_the_start_goes_on_from_the_defaults(self) -> None:
        root = Root()
        with self.assertLogs('modules.settings.store', 'WARNING') as logs:
            self.store.load_startup(root)
        self.assertTrue(any("setup starts on its defaults" in line for line in logs.output))
        self.assertEqual(root.to_dict(), Root().to_dict())
        self.store.save(root, 'setup', self.store.startup('setup'))       # the panel's Save writes the file
        self.assertTrue(self.store.path('setup', 'studio').exists())

    def test_without_another_root_groups_file_its_defaults_stay(self) -> None:
        root = Root()
        root.setup.camera.exposure = 3000
        self.store.save(root, 'setup', 'studio')
        other = Root()
        with self.assertLogs('modules.settings.store', 'WARNING'):
            self.store.load_startup(other)
        self.assertEqual(other.setup.camera.exposure, 3000)
        self.assertEqual(other.flow.motion.gain, 1.0)


if __name__ == '__main__':
    unittest.main()
