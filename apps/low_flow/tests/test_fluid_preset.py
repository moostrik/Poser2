"""Low Flow's fluid takes deep_flow's values: each setting the 2D fluid shares with the 3D holds, in the
studio preset, deep_flow's value at the same path, the sizes apart."""

import json
import unittest
from pathlib import Path

from modules.flow import Fluid2DSettings

ROOT: Path = Path(__file__).resolve().parents[3]
LOW_FLOW: Path = ROOT / 'apps' / 'low_flow' / 'data' / 'settings' / 'studio.json'
DEEP: Path = ROOT / 'apps' / 'deep_flow' / 'data' / 'settings' / 'settings.json'

# A textbook default, not deep_flow's: a plane's sizes, and gravity down the screen (deep_flow's points along z)
TEXTBOOK: set[tuple[str, ...]] = {('width',), ('height',), ('gravity', 'y')}


def leaves(node: dict, at: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    found: dict[tuple[str, ...], object] = {}
    for key, value in node.items():
        if isinstance(value, dict):
            found.update(leaves(value, at + (key,)))
        else:
            found[at + (key,)] = value
    return found


def at(node: dict, path: tuple[str, ...]) -> object:
    for key in path:
        node = node[key]
    return node


class TestFluidPreset(unittest.TestCase):

    def test_deep_flow_values(self) -> None:
        low_flow = json.loads(LOW_FLOW.read_text(encoding='utf-8'))['render']['fluid']
        deep = json.loads(DEEP.read_text(encoding='utf-8'))['render']['fluid']
        for path in leaves(Fluid2DSettings().to_dict()):
            if path in TEXTBOOK:
                continue
            with self.subTest(path='.'.join(path)):
                self.assertEqual(at(low_flow, path), at(deep, path))


if __name__ == '__main__':
    unittest.main()
