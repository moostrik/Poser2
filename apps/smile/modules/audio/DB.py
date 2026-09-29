import json
import random

from modules.audio.settings import AudioSettings

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
            print('warning', movement_id, self._file_path)
            return 'none'

        audio_entries: dict = self._data[movement_id]['audio_ids']
        filtered_entries: dict = {}
        for e in audio_entries:
            if not audio_entries[e]['rejected']:
                filtered_entries[e] = audio_entries[e]

        if filtered_entries == {}:
            print('warning', movement_id, 'no viable entries found')
            return 'none'

        keys, scores = zip(*[(key, entry["score"]) for key, entry in filtered_entries.items()])
        random_index: int = random.choices(range(len(keys)), weights=scores)[0]
        return keys[random_index]

    def get_audio_id(self) -> str:
        return self._audio_id
