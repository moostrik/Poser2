"""The panel's tabs: the children of each root group; a root group placed after a card has no tabs, its children
follow that card."""

import unittest

from modules.settings import BaseSettings, Field, Group, mirror

from ..page import TabEntry, placements, tab_entries


class Camera(BaseSettings):
    exposure: Field[float] = Field(1.0)


class Pose(BaseSettings):
    frequency: Field[float] = Field(30.0)


class Hidden(BaseSettings):
    port: Field[int] = Field(666, visible=False)


class View(BaseSettings):
    width: Field[int] = Field(448)


class Setup(BaseSettings):
    camera: Group[Camera] = Group(Camera)
    pose: Group[Pose] = Group(Pose)
    panel: Group[Hidden] = Group(Hidden)


class Flow(BaseSettings):
    view: Group[View] = Group(View)


class Rooted(BaseSettings):
    input_fps: Field[float] = Field(60.0)
    setup: Group[Setup] = Group(Setup)
    flow: Group[Flow] = Group(Flow)


class Placed(BaseSettings):
    setup: Group[Setup] = Group(Setup)
    flow: Group[Flow] = Group(Flow, after="setup.camera")


class TestTabs(unittest.TestCase):

    def test_a_tab_per_child_of_a_root_group(self):
        root = Rooted()
        self.assertEqual(tab_entries(root), [
            TabEntry("setup.camera", "setup", root.setup.camera),
            TabEntry("setup.pose", "setup", root.setup.pose),
            TabEntry("flow.view", "flow", root.flow.view),
        ])

    def test_a_placed_root_group_has_no_tabs_and_follows_its_card(self):
        root = Placed()
        self.assertEqual(tab_entries(root), [
            TabEntry("setup.camera", "setup", root.setup.camera),
            TabEntry("setup.pose", "setup", root.setup.pose),
        ])
        self.assertEqual(placements(root), {"setup.camera": [(root.flow, "view")]})

    def test_the_mirror_keeps_the_placement(self):
        copy = mirror.build_mirror(mirror.describe(Placed()))
        self.assertEqual([entry.path for entry in tab_entries(copy)], ["setup.camera", "setup.pose"])
        self.assertEqual({card: [name for _, name in cards] for card, cards in placements(copy).items()},
                         {"setup.camera": ["view"]})


if __name__ == "__main__":
    unittest.main()
