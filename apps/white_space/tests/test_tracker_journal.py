"""Tests for the tracker diagnosis journal: the line codec and the writer thread."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from modules.camera import CameraTracklet, TrackingStatus
from modules.tracker import PanoramicAnnotation, Tracklet
from modules.utils import Rect

from apps.white_space.tracker_journal import (
    JOURNAL_FILE, LEGEND, TrackerJournal, TrackerJournalSettings, emit_item, in_item, obs_item,
)


def make_observation(obs_id: int = 7, world_id: int = 3, player: int | None = 1) -> Tracklet:
    annotation = PanoramicAnnotation(50.0, 95.0, False, 2.5, 1.8, player=player)
    return Tracklet(cam_id=0, id=world_id, obs_id=obs_id, status=TrackingStatus.TRACKED,
                    roi=Rect(0.4, 0.1, 0.1, 0.5), annotation=annotation,
                    external_id=2, external_age_in_frames=10, time_stamp=1.0, created_at=1.0, last_active=1.0)


class TestLineCodec(unittest.TestCase):

    def test_items_match_their_legend(self) -> None:
        raw = Tracklet(cam_id=0, status=TrackingStatus.TRACKED, roi=Rect(0.4, 0.1, 0.1, 0.5),
                       external_id=2, external_age_in_frames=10, time_stamp=1.0, created_at=1.0, last_active=1.0)
        observation = make_observation()
        self.assertEqual(len(in_item(raw)), len(LEGEND['in']))
        self.assertEqual(len(obs_item(observation)), len(LEGEND['obs']))
        self.assertEqual(len(emit_item(1, observation)), len(LEGEND['emit']))

    def test_items_round_trip_through_json(self) -> None:
        observation = make_observation()
        item = json.loads(json.dumps(obs_item(observation)))
        legend = {name: value for name, value in zip(LEGEND['obs'], item)}
        self.assertEqual(legend['obs'], 7)
        self.assertEqual(legend['world'], 3)
        self.assertEqual(legend['cam'], 0)
        self.assertEqual(legend['ext'], 2)
        self.assertEqual(legend['status'], TrackingStatus.TRACKED.value)
        self.assertAlmostEqual(legend['local'], 50.0)
        self.assertAlmostEqual(legend['azimuth'], 95.0)
        self.assertAlmostEqual(legend['height'], 1.8)
        self.assertIsNone(legend['rejected'])
        self.assertEqual(legend['player'], 1)

    def test_a_waiting_observation_has_a_null_player(self) -> None:
        item = obs_item(make_observation(player=None))
        self.assertIsNone(item[LEGEND['obs'].index('player')])


class TestJournalWriter(unittest.TestCase):

    def capture_folders(self, data_path: str, settings: TrackerJournalSettings) -> list[Path]:
        base = Path(data_path) / settings.output_path
        return sorted(base.iterdir()) if base.exists() else []

    def read_lines(self, folder: Path) -> list[dict]:
        text = (folder / JOURNAL_FILE).read_text(encoding='utf-8')
        return [json.loads(line) for line in text.splitlines()]

    def test_a_capture_writes_meta_entries_marks_and_end(self) -> None:
        settings = TrackerJournalSettings()
        settings.enabled = True
        with tempfile.TemporaryDirectory() as tmp:
            journal = TrackerJournal(settings, data_path=tmp, meta_provider=lambda: {'app': 'test'})
            journal.start()
            journal.submit_observations([make_observation()])
            journal.submit_tracklets({1: make_observation()})
            journal.mark()
            journal.stop()

            folders = self.capture_folders(tmp, settings)
            self.assertEqual(len(folders), 1)
            lines = self.read_lines(folders[0])
            kinds = [line['k'] for line in lines]
            self.assertEqual(kinds, ['meta', 'obs', 'emit', 'mark', 'end'])
            self.assertEqual(lines[0]['app'], 'test')
            self.assertEqual(lines[0]['legend'], LEGEND)
            self.assertEqual(len(lines[1]['items']), 1)
            self.assertEqual(lines[2]['items'][0][0], 1)          # emit keyed by player slot
            self.assertEqual(lines[3]['n'], 1)
            self.assertEqual(lines[4]['dropped'], 0)
            self.assertEqual(lines[4]['marks'], 1)
            self.assertEqual(settings.marks, 1)

    def test_camera_tracklets_write_an_in_line(self) -> None:
        settings = TrackerJournalSettings()
        settings.enabled = True
        raw = CameraTracklet(cam_id=2, id=7, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=12, time=1.5)
        with tempfile.TemporaryDirectory() as tmp:
            journal = TrackerJournal(settings, data_path=tmp)
            journal.start()
            journal.submit_cam_tracklets(2, [raw], 1.5)
            journal.stop()

            lines = self.read_lines(self.capture_folders(tmp, settings)[0])
            self.assertEqual([line['k'] for line in lines], ['meta', 'in', 'end'])
            self.assertEqual(lines[1]['cam'], 2)
            self.assertEqual(lines[1]['items'][0][0], 7)          # the device id

    def test_idle_journal_accepts_nothing_and_writes_nothing(self) -> None:
        settings = TrackerJournalSettings()                       # enabled stays False
        with tempfile.TemporaryDirectory() as tmp:
            journal = TrackerJournal(settings, data_path=tmp)
            journal.submit_observations([make_observation()])     # before start: ignored
            journal.start()
            journal.submit_observations([make_observation()])     # idle: ignored
            journal.mark()                                        # idle: logged, not counted
            journal.stop()

            self.assertEqual(self.capture_folders(tmp, settings), [])
            self.assertEqual(settings.marks, 0)

    def test_the_switch_toggles_captures_each_into_its_own_folder(self) -> None:
        settings = TrackerJournalSettings()
        with tempfile.TemporaryDirectory() as tmp:
            journal = TrackerJournal(settings, data_path=tmp)
            journal.start()
            settings.enabled = True                               # capture 1
            journal.submit_observations([make_observation()])
            settings.enabled = False                              # closed (the bind joins the writer)
            settings.enabled = True                               # capture 2
            journal.mark()
            journal.stop()

            folders = self.capture_folders(tmp, settings)
            self.assertEqual(len(folders), 2)
            first = [line['k'] for line in self.read_lines(folders[0])]
            second = [line['k'] for line in self.read_lines(folders[1])]
            self.assertEqual(first, ['meta', 'obs', 'end'])
            self.assertEqual(second, ['meta', 'mark', 'end'])
            self.assertEqual(self.read_lines(folders[1])[-1]['marks'], 1)   # counted per capture

    def test_a_settings_change_mid_capture_writes_a_track_line(self) -> None:
        settings = TrackerJournalSettings()
        settings.enabled = True
        gates = {'link_angle': 15.0}
        with tempfile.TemporaryDirectory() as tmp:
            journal = TrackerJournal(settings, data_path=tmp, track_provider=lambda: dict(gates))
            journal._poll_seconds = 0.02
            journal.start()
            time.sleep(0.3)                                       # baseline polled, unchanged
            gates['link_angle'] = 18.0
            time.sleep(0.3)
            journal.stop()

            lines = self.read_lines(self.capture_folders(tmp, settings)[0])
            track_lines = [line for line in lines if line['k'] == 'track']
            self.assertEqual(len(track_lines), 1)                 # only the change, not every poll
            self.assertEqual(track_lines[0]['track'], {'link_angle': 18.0})


if __name__ == '__main__':
    unittest.main()
