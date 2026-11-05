"""The two flow systems' settings: each app preset loads into its system without an unknown key, and
every field of the flow's settings classes is named somewhere in the flow's code."""

import json
import re
import unittest
from pathlib import Path

from modules.settings import BaseSettings, Group
from modules.render.layers import Motion2DLayerSettings, Bridge2DLayerSettings, Fluid2DLayerSettings, \
                                  Motion3DLayerSettings, Bridge3DLayerSettings, Fluid3DLayerSettings

ROOT: Path = Path(__file__).resolve().parents[3]
PRESETS_2D: list[Path] = [ROOT / 'apps' / 'hd_trio' / 'data' / 'settings' / name for name in ('studio.json', 'umu.json')]
PRESETS_3D: list[Path] = [ROOT / 'apps' / 'deep_flow' / 'data' / 'settings' / 'settings.json']
SOURCES: list[Path] = [ROOT / 'modules' / 'flow', ROOT / 'modules' / 'render' / 'layers' / 'flow']
STAGES: tuple[str, ...] = ('motion', 'bridge', 'fluid')
SYSTEMS: dict[str, dict[str, type[BaseSettings]]] = {
    '2D': {'motion': Motion2DLayerSettings, 'bridge': Bridge2DLayerSettings, 'fluid': Fluid2DLayerSettings},
    '3D': {'motion': Motion3DLayerSettings, 'bridge': Bridge3DLayerSettings, 'fluid': Fluid3DLayerSettings},
}


def settings_classes(cls: type[BaseSettings], seen: set[type[BaseSettings]] | None = None) -> set[type[BaseSettings]]:
    """`cls` and the settings classes of its groups, recursively."""
    seen = seen if seen is not None else set()
    if cls in seen:
        return seen
    seen.add(cls)
    for attribute in vars(cls).values():
        if isinstance(attribute, Group):
            settings_classes(attribute.settings_type, seen)
    return seen


def flow_source() -> str:
    return '\n'.join(path.read_text(encoding='utf-8') for root in SOURCES for path in sorted(root.rglob('*.py')))


class PresetTest(unittest.TestCase):
    """A preset's stage groups load into the system's settings classes with nothing left over."""

    def _load(self, presets: list[Path], system: dict[str, type[BaseSettings]]) -> None:
        for preset in presets:
            render: dict = json.loads(preset.read_text(encoding='utf-8'))['render']
            for stage in STAGES:
                with self.subTest(preset=preset.name, stage=stage):
                    settings: BaseSettings = system[stage]()
                    with self.assertNoLogs('modules.settings.base_settings', level='WARNING'):
                        settings.update_from_dict(render[stage])

    def test_hd_trio_presets_load_into_the_2d_system(self) -> None:
        self._load(PRESETS_2D, SYSTEMS['2D'])

    def test_deep_flow_preset_loads_into_the_3d_system(self) -> None:
        self._load(PRESETS_3D, SYSTEMS['3D'])


class DeadFieldTest(unittest.TestCase):
    """A field no code names, beyond its own declaration, is dead."""

    def test_every_field_is_named_in_the_flow_code(self) -> None:
        source: str = flow_source()
        classes: set[type[BaseSettings]] = set()
        for system in SYSTEMS.values():
            for cls in system.values():
                settings_classes(cls, classes)
        for cls in sorted(classes, key=lambda c: c.__name__):
            for name in cls().fields:
                with self.subTest(settings=cls.__name__, field=name):
                    uses: int = len(re.findall(rf'\b{re.escape(name)}\b', source))
                    self.assertGreaterEqual(uses, 2, f'{cls.__name__}.{name} is declared and named nowhere else')


if __name__ == '__main__':
    unittest.main()
