"""The laughter test: one laugh round the screens, each screen lit with its number while it sounds, to check
data/audio/screen_angles.txt on site (the panel's Audio tab)."""

import logging
import os
from typing import Callable

from modules import sphere

from apps.smile.audio.settings import AudioSettings

logger = logging.getLogger(__name__)

SCREENS: int = sphere.SCREENS   # the sphere: two walls of CELLS
CELLS: int = sphere.CELLS       # cells per wall
SAMPLE_FOLDER: str = 'apps/smile/data/audio/laughter'
SAMPLE_TYPES: tuple[str, ...] = ('.wav', '.aif', '.aiff', '.mp3', '.flac', '.m4a', '.ogg')

Send = Callable[[int, int, float, int], None]   # wall, cell, seconds, the number to show: the light on the wall
Bang = Callable[[int, int], None]               # screen index, sample number: the test bang to Max


def list_samples(folder: str) -> list[str]:
    """The sound files of the folder in the order the patch's `polybuffer~ readfolder` numbers them, so a
    file's position + 1 is its number in the patch. That order is the folder's listing as Windows gives it
    (NTFS: alphabetical, case ignored), which `readfolder` does not sort further; a laugh that is not the
    chosen file means it differs. A missing folder is an empty list."""
    try:
        names: list[str] = [n for n in os.listdir(folder) if n.lower().endswith(SAMPLE_TYPES) and not n.startswith('.')]
    except OSError:
        return []
    return sorted(names, key=str.upper)


class ScreenTest:
    """While `laughter_test` is on, every `laughter_test_step` seconds the next of the 48 screens is lit, in
    the sphere's order (1 to 5 down the first rib, 6 to 8 the next ...), on the cell the wiring gives it
    (`sphere.cell_of_screen`): `send(wall, cell, seconds, number)` lights it on the wall with the number
    (the clip player's `light`), and `bang(index, sample)` sends Max `/test_bang` with the cell's index (the
    patch plays the sample from where the table puts that cell, past its gates; meanwhile `/laughter_test`
    closes the gate of the clips' laughs, so this is the only one). The sample is the file
    `laughter_test_sample`, offered from the laughter folder (`laughter_samples`), sent as its number in the
    patch. `laughter_test_screen` shows the screen. Driven from Main.sequence_callback, the app clock, and
    never raises there: a light or bang that fails is one warning per switch-on."""

    def __init__(self, settings: AudioSettings, send: Send | None = None, bang: Bang | None = None,
                 folder: str = SAMPLE_FOLDER) -> None:
        self.settings: AudioSettings = settings
        self._send: Send = send if send is not None else (lambda wall, cell, seconds, number: None)
        self._bang: Bang = bang if bang is not None else (lambda index, sample: None)
        self._index: int = -1           # the screen lit last, -1 while off
        self._next_time: float = 0.0    # when the next screen is due
        self._warned: bool = False      # a failed send was logged since the switch-on
        # the switch is saved with the presets: one saved on must not start the installation testing
        settings.laughter_test = False
        settings.laughter_test_screen = 0
        # the samples to choose from; a choice that is no file any more falls back to the first
        samples: list[str] = list_samples(folder)
        settings.laughter_samples = samples if samples else ['']
        if settings.laughter_test_sample not in samples:
            settings.laughter_test_sample = samples[0] if samples else ''
        if not samples:
            logger.warning('No laughter samples in %s', folder)

    def update(self, now: float) -> None:
        settings: AudioSettings = self.settings
        if not settings.laughter_test:
            if self._index != -1:
                self._index = -1
                self._next_time = 0.0
                self._warned = False
                settings.laughter_test_screen = 0
            return
        if now < self._next_time:
            return
        # the screens in their own order, 1 to 48 down each rib, on whatever cell the wiring puts them
        self._index = (self._index + 1) % SCREENS
        screen: int = self._index + 1
        wall, cell = sphere.cell_of_screen(screen)
        step: float = settings.laughter_test_step
        try:
            self._send(wall, cell, step, screen)
            self._bang(wall * CELLS + cell, self.sample_number())     # the patch's index is the cell's
        except Exception:
            if not self._warned:
                self._warned = True
                logger.exception('Laughter test: screen %d failed (wall %d, cell %d)', screen, wall + 1, cell)
        settings.laughter_test_screen = screen
        self._next_time = now + step

    def sample_number(self) -> int:
        """The chosen file's number in the patch (its position in the folder + 1); 1 when it is not listed."""
        samples: list[str] = list(self.settings.laughter_samples)
        name: str = self.settings.laughter_test_sample
        return samples.index(name) + 1 if name in samples else 1
