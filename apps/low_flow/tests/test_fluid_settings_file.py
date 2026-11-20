"""Low Flow's studio settings file holds the 2D fluid's settings: every key of `Fluid2DSettings` at the settings file's fluid, and
nothing the settings no longer have; the values are the app's own look."""

import json
import unittest
from pathlib import Path

from modules.flow import Fluid2DSettings

ROOT: Path = Path(__file__).resolve().parents[3]
LOW_FLOW: Path = ROOT / 'apps' / 'low_flow' / 'data' / 'settings' / 'flow' / 'studio.json'


def leaves(node: dict, at: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    for key, value in node.items():
        if isinstance(value, dict):
            found |= leaves(value, at + (key,))
        else:
            found.add(at + (key,))
    return found


class TestFluidSettingsFile(unittest.TestCase):

    def setUp(self) -> None:
        self.settings_file: dict = json.loads(LOW_FLOW.read_text(encoding='utf-8'))['fluid']

    def test_the_settings_file_holds_every_key_of_the_settings(self) -> None:
        missing = leaves(Fluid2DSettings().to_dict()) - leaves(self.settings_file)
        self.assertEqual(missing, set(), 'keys of Fluid2DSettings missing from the settings file')

    def test_the_settings_file_loads_into_the_settings(self) -> None:
        settings = Fluid2DSettings()
        settings.update_from_dict(self.settings_file)
        self.assertEqual(leaves(settings.to_dict()), leaves(Fluid2DSettings().to_dict()))


if __name__ == '__main__':
    unittest.main()
