"""The panel's tabs: the children of each root group, or one tab for a root group declared `tab`."""

import unittest

from modules.settings import BaseSettings, Field, Group, mirror

from ..page import TabEntry, tab_entries


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


class OneTab(BaseSettings):
    setup: Group[Setup] = Group(Setup)
    flow: Group[Flow] = Group(Flow, tab=True)


class TestTabs(unittest.TestCase):

    def test_a_tab_per_child_of_a_root_group(self):
        root = Rooted()
        self.assertEqual(tab_entries(root), [
            TabEntry("setup.camera", "setup", root.setup.camera),
            TabEntry("setup.pose", "setup", root.setup.pose),
            TabEntry("flow.view", "flow", root.flow.view),
        ])

    def test_a_root_group_declared_tab_is_one_tab(self):
        root = OneTab()
        self.assertEqual(tab_entries(root), [
            TabEntry("setup.camera", "setup", root.setup.camera),
            TabEntry("setup.pose", "setup", root.setup.pose),
            TabEntry("flow", "", root.flow),
        ])

    def test_the_mirror_keeps_a_one_tab_root_group(self):
        copy = mirror.build_mirror(mirror.describe(OneTab()))
        self.assertEqual([(entry.path, entry.group) for entry in tab_entries(copy)],
                         [("setup.camera", "setup"), ("setup.pose", "setup"), ("flow", "")])


if __name__ == "__main__":
    unittest.main()
