import json
import random

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)
        return data_dict

def save_json(file_path: str, data_dict: dict) -> None:
    json_str: str = json.dumps(data_dict, separators=(',', ':')).replace('},', '},\n\t\t').replace(',"', ', "').replace(':', ': ').replace('"audio_ids": {', '"audio_ids": {\n\t\t').replace('}}},\n\t\t', '}}},\n\t')
    formatted_json: str = '{\n\t' + json_str[1:-1] + '\n}'
    with open(file_path, 'w', encoding='utf-8') as jsonfile:
        jsonfile.write(formatted_json)

def get_key_by_entry(dictionary, entry_key, entry_value) -> str | None:
    for key, entry in dictionary.items():
        if entry[entry_key] == entry_value:
            return key
    return None


class DB():
    def __init__(self, file_path: str) -> None:
        self._file_path: str = file_path

        self._data: dict = load_json(file_path)

        self._movement_id: str = 'none'
        self._audio_id: str = 'none'
        self._can_vote = False
        self._can_reject = False

    def update(self, movement_id: str) -> None:
        self._movement_id: str = movement_id

        if not self._movement_id in self._data:
            print('warning', movement_id, self._file_path)
            self._audio_id: str = 'none'
            return

        audio_entries: dict = self._data[self._movement_id]['audio_ids']
        filtered_entries: dict = {}
        for e in audio_entries:
            if not audio_entries[e]['rejected']:
                filtered_entries[e] = audio_entries[e]

        if filtered_entries == {}:
            print('warning', self._movement_id, 'no viable entries found')
            self._audio_id: str = 'none'
            return

        keys, scores = zip(*[(key, entry["score"]) for key, entry in filtered_entries.items()])
        random_index: int = random.choices(range(len(keys)), weights=scores)[0]
        self._audio_id: str = keys[random_index]
        self._can_vote = True
        self._can_reject = True

    def get_audio_id(self) -> str:
        return self._audio_id

    def get_score(self) -> float:
        if not self._movement_id in self._data:
            print('movement id not found', self._movement_id)
            return 0.0
        if not self._audio_id in self._data[self._movement_id]['audio_ids']:
            print('audio id not found', self._audio_id)
            return 0.0
        return self._data[self._movement_id]['audio_ids'][self._audio_id]['score']

    def reject(self) -> None:
        if not self._can_reject:
            return
        if not self._movement_id in self._data:
            return
        if not self._audio_id in self._data[self._movement_id]['audio_ids']:
            return

        self._can_vote = False
        self._data[self._movement_id]['audio_ids'][self._audio_id]['rejected'] = True
        save_json(self._file_path, self._data)

        self.update(self._movement_id)

    def vote(self) -> None:
        if not self._can_vote:
            return
        if not self._movement_id in self._data:
            return
        if not self._audio_id in self._data[self._movement_id]['audio_ids']:
            return

        self._can_vote = False
        self._data[self._movement_id]['audio_ids'][self._audio_id]['score'] += 1
        save_json(self._file_path, self._data)

    def check_audio(self) -> None:
        pass

