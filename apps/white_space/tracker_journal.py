"""Diagnosis journal for the panoramic tracker.

A subscribe-only tap on the tracker's two output channels and the cameras' raw tracklets,
written as JSONL for offline analysis — which world ids are born, linked, split and retired
around a doubling incident, and at what measured margins. It changes nothing in the pipeline:
while the `enabled` switch is off the taps return on a single flag check, and while capturing
the hot path is one bounded-queue put, with serialization and file I/O on the capture's own
writer thread. When the queue is full the entry is dropped and counted, never waited on.

The switch toggles live: each enable captures into its own folder under ``data/diagnosis/`` —
a ``meta`` line (settings snapshot, legend, footage folder), then ``in`` (raw device tracklets
per camera), ``obs`` (every observation and rejected detection, once per tracker tick), ``emit``
(the primaries, keyed by player slot), ``mark`` (the operator flagging an incident) and
``track`` (the tracker settings, re-written whenever they change mid-capture, so tuning stays
interpretable) lines, closed by an ``end`` line with the drop count.
"""

import json
import logging
import math
import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Callable

from modules.oak import DepthTracklet, make_folder_name
from modules.settings import BaseSettings, Field, Widget
from modules.tracker import PanoramicAnnotation, Tracklet, TrackletDict

logger = logging.getLogger(__name__)

JOURNAL_FILE = 'tracker_journal.jsonl'
# How often a capture compares the tracker settings against what it last wrote.
TRACK_POLL_SECONDS: float = 1.0

# What the positions in each line kind's `items` arrays mean; written into the meta line so a
# journal file explains itself.
LEGEND: dict[str, list[str]] = {
    'in': ['ext', 'status', 'age', 'x', 'y', 'w', 'h'],
    'obs': ['obs', 'world', 'cam', 'ext', 'status', 'local', 'azimuth', 'height', 'rejected', 'player'],
    'emit': ['player', 'obs', 'cam', 'azimuth'],
}


def _round(value: float, digits: int) -> float | None:
    """JSON-safe: a non-finite reading becomes null rather than poisoning the line."""
    return round(value, digits) if math.isfinite(value) else None


def in_item(tracklet: Tracklet) -> list:
    """One raw device tracklet, as `LEGEND['in']` reads it."""
    roi = tracklet.roi
    return [tracklet.external_id, tracklet.status.value, tracklet.external_age_in_frames,
            _round(roi.x, 4), _round(roi.y, 4), _round(roi.width, 4), _round(roi.height, 4)]


def obs_item(tracklet: Tracklet) -> list:
    """One observation or rejected detection, as `LEGEND['obs']` reads it."""
    a = tracklet.annotation
    assert isinstance(a, PanoramicAnnotation)
    rejected: int | None = None if a.rejected is None else int(a.rejected)
    return [tracklet.obs_id, tracklet.id, tracklet.cam_id, tracklet.external_id,
            tracklet.status.value, _round(a.local_angle, 2), _round(a.world_angle, 2),
            _round(a.height, 3), rejected, a.player]


def emit_item(player: int, tracklet: Tracklet) -> list:
    """One emitted primary, as `LEGEND['emit']` reads it."""
    a = tracklet.annotation
    assert isinstance(a, PanoramicAnnotation)
    return [player, tracklet.obs_id, tracklet.cam_id, _round(a.world_angle, 2)]


class TrackerJournalSettings(BaseSettings):
    enabled:     Field[bool] = Field(False, widget=Widget.switch,
                                     description="Capture a tracker diagnosis journal (toggles live; one folder per capture)")
    output_path: Field[str]  = Field("diagnosis", access=Field.INIT,
                                     description="Journal output directory, under the app's data path")
    mark:        Field[bool] = Field(False, widget=Widget.button,
                                     description="Mark an incident now (or press M in the render window)")
    marks:       Field[int]  = Field(0, access=Field.READ,
                                     description="Incidents marked this capture")


class TrackerJournal:
    """Writes the journal. Always wired — subscribe `submit_cam_tracklets`, `submit_observations`
    and `submit_tracklets` beside the board's taps; they return on one flag check while the
    `enabled` switch is off. Each enable captures into its own folder on its own writer thread.
    `mark()` flags an incident from any thread; an idle press is logged, never counted."""

    def __init__(self, settings: TrackerJournalSettings, data_path: str = "",
                 meta_provider: Callable[[], dict] | None = None,
                 track_provider: Callable[[], dict] | None = None) -> None:
        self.settings: TrackerJournalSettings = settings
        self._data_path: str = data_path
        self._meta_provider: Callable[[], dict] | None = meta_provider
        self._track_provider: Callable[[], dict] | None = track_provider

        self._queue: Queue[tuple] = Queue(maxsize=1024)
        self._capturing: bool = False
        self._started: bool = False
        self._lifecycle_lock = Lock()       # one begin/end transition at a time
        self._writer: Thread | None = None
        self._poll_seconds: float = TRACK_POLL_SECONDS
        self._dropped: int = 0
        self._mark_count: int = 0
        self.folder_path: Path | None = None

        self.settings.bind(TrackerJournalSettings.enabled, self._on_enabled)
        self.settings.bind(TrackerJournalSettings.mark, self._on_mark)

    # LIFECYCLE
    def start(self) -> None:
        """App start: capture begins now if the switch (preset, panel or `-journal`) is on."""
        self._started = True
        if self.settings.enabled:
            self._begin()

    def stop(self) -> None:
        self._started = False
        self.settings.unbind(TrackerJournalSettings.enabled, self._on_enabled)
        self.settings.unbind(TrackerJournalSettings.mark, self._on_mark)
        self._end()

    def _begin(self) -> None:
        with self._lifecycle_lock:
            if self._capturing:
                return
            base: Path = Path(self._data_path) / self.settings.output_path
            folder: Path = base / make_folder_name()
            suffix: int = 1
            while (folder / JOURNAL_FILE).exists():     # a re-enable within the same second
                folder = base / f"{make_folder_name()}-{suffix}"
                suffix += 1
            folder.mkdir(parents=True, exist_ok=True)
            self.folder_path = folder
            while not self._queue.empty():              # entries that raced a previous end
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            self._dropped = 0
            self._mark_count = 0
            self.settings.marks = 0
            self._writer = Thread(target=self._write, name='tracker_journal', daemon=True)
            self._capturing = True
            self._writer.start()
            logger.info("TrackerJournal capturing → %s", folder / JOURNAL_FILE)

    def _end(self) -> None:
        with self._lifecycle_lock:
            if not self._capturing:
                return
            self._capturing = False                     # the writer drains the queue, then closes
            assert self._writer is not None
            self._writer.join(timeout=5.0)
            self._writer = None
            logger.info("TrackerJournal capture closed (%d marks)", self._mark_count)

    # WRITER THREAD
    def _write(self) -> None:
        assert self.folder_path is not None
        with open(self.folder_path / JOURNAL_FILE, 'w', encoding='utf-8') as file:
            meta: dict = {'k': 'meta', 't': time.time(), 'legend': LEGEND}
            if self._meta_provider is not None:
                try:
                    meta.update(self._meta_provider())
                except Exception:
                    logger.exception("TrackerJournal meta provider failed; capture continues without it")
            file.write(json.dumps(meta) + '\n')

            track_provider: Callable[[], dict] | None = self._track_provider
            last_track: dict | None = None
            next_poll: float = 0.0                      # first poll initializes the baseline
            while self._capturing or not self._queue.empty():
                if track_provider is not None and time.monotonic() >= next_poll:
                    next_poll = time.monotonic() + self._poll_seconds
                    try:
                        track: dict = track_provider()
                        if last_track is not None and track != last_track:
                            file.write(json.dumps({'k': 'track', 't': time.time(), 'track': track}) + '\n')
                        last_track = track
                    except Exception:
                        logger.exception("TrackerJournal track poll failed; polling stops")
                        track_provider = None
                try:
                    entry: tuple = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    file.write(self._to_line(entry) + '\n')
                except Exception:
                    logger.exception("TrackerJournal failed to write an entry")

            file.write(json.dumps({'k': 'end', 't': time.time(),
                                   'dropped': self._dropped, 'marks': self._mark_count}) + '\n')
        if self._dropped:
            logger.warning("TrackerJournal dropped %d entries (queue full)", self._dropped)

    @staticmethod
    def _to_line(entry: tuple) -> str:
        kind: str = entry[0]
        if kind == 'in':
            _, t, cam_id, tracklets = entry
            return json.dumps({'k': 'in', 't': t, 'cam': cam_id,
                               'items': [in_item(x) for x in tracklets]})
        if kind == 'obs':
            _, t, observations = entry
            return json.dumps({'k': 'obs', 't': t, 'items': [obs_item(x) for x in observations]})
        if kind == 'emit':
            _, t, emitted = entry
            return json.dumps({'k': 'emit', 't': t,
                               'items': [emit_item(p, x) for p, x in emitted.items()]})
        _, t, n = entry
        return json.dumps({'k': 'mark', 't': t, 'n': n})

    def _put(self, entry: tuple) -> None:
        try:
            self._queue.put_nowait(entry)
        except Full:
            self._dropped += 1

    # INTAKE — wired beside the board's taps; one flag check when idle, one queue put capturing
    def submit_cam_tracklets(self, cam_id: int, cam_tracklets: list[DepthTracklet]) -> None:
        """The raw device tracklets, converted exactly as the tracker converts them."""
        if not self._capturing:
            return
        tracklets: list[Tracklet] = []
        for t in cam_tracklets:
            tracklet: Tracklet | None = Tracklet.from_depthcam(cam_id, t)
            if tracklet is not None:
                tracklets.append(tracklet)
        self._put(('in', time.time(), cam_id, tracklets))

    def submit_observations(self, observations: list[Tracklet]) -> None:
        """Every observation plus every rejected detection — once per tracker tick."""
        if not self._capturing:
            return
        self._put(('obs', time.time(), observations))

    def submit_tracklets(self, tracklets: TrackletDict) -> None:
        """The emitted primaries, keyed by player slot."""
        if not self._capturing:
            return
        self._put(('emit', time.time(), tracklets))

    def mark(self) -> None:
        """Flag an incident at this moment; callable from any thread."""
        if not self._capturing:
            logger.info("Mark pressed, but the journal is not capturing (record → journal → enabled)")
            return
        self._mark_count += 1
        self._put(('mark', time.time(), self._mark_count))
        self.settings.marks = self._mark_count

    def _on_enabled(self, value: object) -> None:
        if not self._started:
            return                          # before app start the switch only arms `start()`
        if value:
            self._begin()
        else:
            self._end()

    def _on_mark(self, value: object = None) -> None:
        if value:
            self.mark()
