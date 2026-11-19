"""The 2D fluid's settings are the 3D's for what a plane has: every 2D setting sits at the 3D's path, under its name,
in its unit, with its default, the sizes apart."""

import re
import unittest

from modules.flow import Fluid2DSettings, Fluid3DSettings
from modules.settings import BaseSettings, Field

# The 2D's own defaults: a plane's sizes are not the box's
OWN_DEFAULTS: set[tuple[str, ...]] = {('width',), ('height',)}

# A description's units, each as the text writes it
UNIT = re.compile(r"\((?:ms|s|fps|1/s)\)|widths/s²|widths/s|widths² per second|widths|[Ss]econds|width/s")


def fields(settings: BaseSettings, at: tuple[str, ...] = ()) -> dict[tuple[str, ...], Field]:
    """Every field of `settings` by its path."""
    found: dict[tuple[str, ...], Field] = {at + (name,): field for name, field in settings.fields.items()}
    for name, child in settings.children.items():
        found.update(fields(child, at + (name,)))
    return found


def units(field: Field) -> list[str]:
    return sorted(set(UNIT.findall(field.description or '')))


class TestFluid2DSettings(unittest.TestCase):

    def setUp(self) -> None:
        self.plane = fields(Fluid2DSettings())
        self.box = fields(Fluid3DSettings())

    def test_every_setting_at_the_3d_path(self) -> None:
        missing = [path for path in self.plane if path not in self.box]
        self.assertEqual(missing, [], "2D settings with no 3D path")

    def test_the_3d_unit(self) -> None:
        for path, field in self.plane.items():
            with self.subTest(path='.'.join(path)):
                self.assertEqual(units(field), units(self.box[path]))

    def test_the_3d_default(self) -> None:
        for path, field in self.plane.items():
            if path in OWN_DEFAULTS:
                continue
            with self.subTest(path='.'.join(path)):
                self.assertEqual(field.default, self.box[path].default)


if __name__ == '__main__':
    unittest.main()
