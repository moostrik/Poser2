"""The screen recording compositor's snap: the first source's shape, or the window's when the window is recorded."""

import unittest

from modules.render import LayerBase, ScreenRecordingCompositor, ScreenRecordingInput, ScreenRecordingSource
from modules.video import ScreenRecorderSettings, ScreenSize


class NoLayer(LayerBase):
    def deallocate(self) -> None:
        pass

    def update(self) -> None:
        pass


class SnapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ScreenRecorderSettings()
        self.compositor = ScreenRecordingCompositor(self.settings, 'unused')
        sources = [ScreenRecordingSource(NoLayer(), 1440, 2560), ScreenRecordingSource(NoLayer(), 640, 480)]
        self.compositor.set_input(ScreenRecordingInput(sources, 1918, 1080, 1.0))

    def tearDown(self) -> None:
        self.compositor.deallocate()

    def test_snap_takes_the_first_source(self) -> None:
        ScreenRecorderSettings.snap.fire(self.settings)
        self.assertEqual((self.settings.width, self.settings.height), (1440, 2560))
        self.assertIs(self.settings.size, ScreenSize.QHD)

    def test_snap_takes_the_window_when_it_is_recorded(self) -> None:
        self.settings.main_window = True
        ScreenRecorderSettings.snap.fire(self.settings)
        self.assertEqual((self.settings.width, self.settings.height), (1918, 1080))

    def test_snap_without_sources_keeps_the_size(self) -> None:
        self.compositor.set_input(ScreenRecordingInput([], 1918, 1080, 1.0))
        ScreenRecorderSettings.snap.fire(self.settings)
        self.assertEqual((self.settings.width, self.settings.height), (2160, 3840))


if __name__ == '__main__':
    unittest.main()
