"""The screen recording compositor's snap: the first source's shape, or the window's when the window is recorded."""

import unittest

from modules.render import LayerBase, ScreenRecordingCompositor, ScreenRecordingInput, ScreenRecordingSource
from modules.video import ScreenRecorderFrameSettings, ScreenRecorderSettings, ScreenSize


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
        ScreenRecorderFrameSettings.snap.fire(self.settings.frame)
        self.assertEqual((self.settings.frame.width, self.settings.frame.height), (1440, 2560))
        self.assertIs(self.settings.frame.size, ScreenSize.QHD)

    def test_snap_takes_the_window_when_it_is_recorded(self) -> None:
        self.settings.frame.main_window = True
        ScreenRecorderFrameSettings.snap.fire(self.settings.frame)
        self.assertEqual((self.settings.frame.width, self.settings.frame.height), (1918, 1080))

    def test_snap_without_sources_keeps_the_size(self) -> None:
        self.compositor.set_input(ScreenRecordingInput([], 1918, 1080, 1.0))
        ScreenRecorderFrameSettings.snap.fire(self.settings.frame)
        self.assertEqual((self.settings.frame.width, self.settings.frame.height), (2160, 3840))


if __name__ == '__main__':
    unittest.main()
