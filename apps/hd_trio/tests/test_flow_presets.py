"""hd_trio's presets load into its copy of the 2D flow's settings without an unknown key."""

import json
import unittest
from pathlib import Path

from modules.settings import BaseSettings
from apps.hd_trio.settings import Motion2DLayerSettings, Bridge2DLayerSettings, Fluid2DLayerSettings

ROOT: Path = Path(__file__).resolve().parents[3]
PRESETS: list[Path] = [ROOT / 'apps' / 'hd_trio' / 'data' / 'settings' / name for name in ('studio.json', 'umu.json')]
STAGES: dict[str, type[BaseSettings]] = {'motion': Motion2DLayerSettings, 'bridge': Bridge2DLayerSettings,
                                         'fluid': Fluid2DLayerSettings}


class FlowPresetTest(unittest.TestCase):
    """A preset's stage groups load into hd_trio's stage settings with nothing left over."""

    def test_presets_load_into_the_2d_flow(self) -> None:
        for preset in PRESETS:
            render: dict = json.loads(preset.read_text(encoding='utf-8'))['setup']['render']
            for stage, cls in STAGES.items():
                with self.subTest(preset=preset.name, stage=stage):
                    settings: BaseSettings = cls()
                    with self.assertNoLogs('modules.settings.base_settings', level='WARNING'):
                        settings.update_from_dict(render[stage])


if __name__ == '__main__':
    unittest.main()
