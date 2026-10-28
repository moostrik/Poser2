import json
import logging
import os
import random

from apps.smile.audio.settings import AudioSettings

logger = logging.getLogger(__name__)

MUSIC_FOLDER: str = 'apps/smile/data/audio/SMILE-BS-Music'    # the tracks, named "<audio id> <title>.mp3", as the Max patch reads them
LAUGHTER_FOLDER: str = 'apps/smile/data/audio/laughter'        # the laughter samples (mp3 and wav), as the Max patch reads them


def track_path(audio_id: str, folder: str = MUSIC_FOLDER) -> str | None:
    """The track's file: the one in the folder whose name starts with the id; None for `none` or no file."""
    if not audio_id or audio_id == 'none' or not os.path.isdir(folder):
        return None
    for name in sorted(os.listdir(folder)):
        if name.startswith(audio_id) and name.lower().endswith('.mp3'):
            return os.path.join(folder, name)
    return None


def laugh_files(folder: str = LAUGHTER_FOLDER) -> list[tuple[str, str]]:
    """The laughter samples as (file name, path), sorted; for the phones (the relay keeps a copy)."""
    if not os.path.isdir(folder):
        return []
    return [(name, os.path.join(folder, name)) for name in sorted(os.listdir(folder)) if name.lower().endswith(('.mp3', '.wav'))]


def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)
        return data_dict


class DB():
    def __init__(self, settings: AudioSettings, file_path: str) -> None:
        self.settings: AudioSettings = settings
        self._file_path: str = file_path

        self._data: dict = load_json(file_path)

        self._movement_id: str = 'none'
        self._audio_id: str = 'none'

    def update(self, movement_id: str) -> None:
        self._movement_id: str = movement_id
        self._audio_id = self._pick(movement_id)
        self.settings.music_id = self._audio_id

    def _pick(self, movement_id: str) -> str:
        if not movement_id in self._data:
            logger.warning('Movement %s is not in %s: no music', movement_id, self._file_path)
            return 'none'

        audio_entries: dict = self._data[movement_id]['audio_ids']
        filtered_entries: dict = {}
        for e in audio_entries:
            if not audio_entries[e]['rejected']:
                filtered_entries[e] = audio_entries[e]

        if filtered_entries == {}:
            logger.warning('Movement %s has no usable entries in %s: no music', movement_id, self._file_path)
            return 'none'

        keys, scores = zip(*[(key, entry["score"]) for key, entry in filtered_entries.items()])
        random_index: int = random.choices(range(len(keys)), weights=scores)[0]
        return keys[random_index]

    def get_audio_id(self) -> str:
        return self._audio_id

    def track_path(self) -> str | None:
        """The file of the track picked last (for the phone and the saved clip; Max finds it itself)."""
        return track_path(self._audio_id)
