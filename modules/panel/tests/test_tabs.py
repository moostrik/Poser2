"""The panel's tabs: the children of each root group, or of a root without root groups."""

import unittest

from modules.settings import BaseSettings, Field, Group

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


class Flat(BaseSettings):
    camera: Group[Camera] = Group(Camera)
    panel: Group[Hidden] = Group(Hidden)
    pose: Group[Pose] = Group(Pose)


class TestTabs(unittest.TestCase):

    def test_a_tab_per_child_of_a_root_group(self):
        root = Rooted()
        self.assertEqual(tab_entries(root), [
            TabEntry("setup.camera", "setup", root.setup.camera),
            TabEntry("setup.pose", "setup", root.setup.pose),
            TabEntry("flow.view", "flow", root.flow.view),
        ])

    def test_a_root_without_setup_has_a_tab_per_child(self):
        root = Flat()
        self.assertEqual(tab_entries(root), [
            TabEntry("camera", "", root.camera),
            TabEntry("pose", "", root.pose),
        ])


if __name__ == "__main__":
    unittest.main()
