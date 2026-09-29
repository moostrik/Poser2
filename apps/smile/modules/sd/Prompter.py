import json
from threading import Lock
from modules.sd.ApiConnect import interrogate
import random
import re
import os
from copy import deepcopy

prompt_folder = 'prompts/'

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)
        return data_dict

def save_json(file_path: str, data_dict: dict) -> None:
    json_str = json.dumps(data_dict, separators=(',', ':')).replace('},', '},\n\t').replace(',"', ', "').replace(':', ': ')
    formatted_json = '{\n\t' + json_str[1:-1] + '\n}'
    with open(file_path, 'w', encoding='utf-8') as jsonfile:
        jsonfile.write(formatted_json)

def get_key_by_entry(dictionary, entry_key, entry_value) -> str | None:
    for key, entry in dictionary.items():
        if entry[entry_key] == entry_value:
            return key
    return None

class KeyWord():
    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path

        self._data: dict = load_json(file_path)
        self._names: list = [entry['name'] for entry in self._data.values()]
        self._keys: list = list(self._data.keys())

        self._id: str
        self.set_id(next(iter(self._data)))
        self._next_id: str = self._id
        self._active: bool = True
        self._vote: bool = False
        self.update()

    def update(self, weighted: bool = True ) -> None:
        if self._active:
            self.set_id(self._next_id)
            if weighted:
                keys, scores = zip(*[(key, entry["score"]) for key, entry in self._data.items() if key != self._next_id])
                random_index = random.choices(range(len(keys)), weights=scores)[0]
                self._next_id = keys[random_index]
            else:
                random_item = random.choice(self._keys)
                self._next_id = random_item

    def update_next(self) -> None:
        if self._active:
            self.set_id(self._next_id)
            # use next key in list
            self._next_id = self._keys[(self._keys.index(self._id) + 1) % len(self._keys)]


    def set_id(self, value) -> bool:
        if not value in self._data:
            return False
        else:
            self._id = value
            self._next_id = self._keys[(self._keys.index(self._id) + 1) % len(self._keys)]
            return True

    def save(self) -> None:
        save_json(self.file_path, self._data)

    def get_id(self) -> str:
        return self._id

    def get_next_id(self) -> str:
        return self._next_id

    def get_name(self) -> str:
        return self._data[self._id]['name']

    def get_next_name(self) -> str:
        return self._data[self._next_id]['name']

    def get_score(self) -> float:
        return self._data[self._id]['score']

    def set_score(self, value: float) -> None:
        self._data[self._id]['score'] = value

    def get_vote(self) -> bool:
        return self._vote

    def set_vote(self, value: bool) -> None:
        self._vote = value

    def get_active(self) -> bool:
        return self._active

    def set_active(self, value: bool) -> None:
        self._active = value
        self.update()

    def get_id_from_name(self, value:str) -> str | None:
        return get_key_by_entry(self._data, 'name', value)

    def get_names(self) -> list :
        return [entry['name'] for entry in self._data.values()]


class Prompter():
    def __init__(self) -> None:

        self.keyword_names: list = ['smile']
        self.smile_repeat: int = 24
        self.smile_repeat_count: int = 0
        self.ask_names: list = ['face', 'clip', 'booru']
        self.wildcard_names = [os.path.splitext(f)[0] for f in os.listdir(prompt_folder) if f.endswith('.json')]
        self.keyword_symbol = '*'
        self.ask_symbol = '+'
        self.wildcard_symbol = '#'

        self.keywords: dict[str, KeyWord] = {}
        for w in self.keyword_names:
            self.keywords[w] = KeyWord(prompt_folder + w + '.json')
        next_keyword_names = deepcopy(self.keyword_names)
        for n in self.keyword_names:
            next_keyword_names.append('next_' + n)
        self.keyword_names = next_keyword_names

        self.wildcards: dict[str, KeyWord] = {}
        for w in self.wildcard_names:
            self.wildcards[w] = KeyWord(prompt_folder + w + '.json')

        self._input_mutex: Lock = Lock()

        self.interrogate = False
        self.interrogateBooru = False
        self.interrogateFace = False
        self.interrogateFirst = False

        self._pos_input: str = ''
        self._pos_prompt: str = ''
        self._neg_input: str = ''
        self._neg_prompt: str = ''
        self._face_prompt: str = ''
        self._rep_input: str = ''

        self._received_vote = False


    def update(self, image: str) -> None:

        pos_prompt = self.get_pos_input()
        neg_prompt = self.get_neg_input()
        self._received_vote = False

        for key in self.wildcards:
            keyword: KeyWord = self.wildcards[key]

            pattern = re.escape(f'{self.wildcard_symbol}{key}{self.wildcard_symbol}')
            while re.findall(pattern, pos_prompt):
                pos_prompt = re.sub(pattern, keyword.get_name(), pos_prompt, count=1)
                keyword.update(False)

        for key in self.keywords:
            keyword: KeyWord = self.keywords[key]
            if self.smile_repeat_count >= self.smile_repeat:
                self.smile_repeat_count = 0
                keyword.update_next()
            self.smile_repeat_count += 1

            pattern = re.escape(f'{self.keyword_symbol}{key}{self.keyword_symbol}')
            pos_prompt = re.sub(pattern, keyword.get_name(), pos_prompt)
            neg_prompt = re.sub(pattern, keyword.get_name(), neg_prompt)

            pattern = re.escape(f'{self.keyword_symbol}next_{key}{self.keyword_symbol}')
            pos_prompt = re.sub(pattern, keyword.get_next_name(), pos_prompt)
            neg_prompt = re.sub(pattern, keyword.get_next_name(), neg_prompt)

        pattern = re.escape(f'{self.ask_symbol}face{self.ask_symbol}')
        if re.search(pattern, pos_prompt):
            pos_prompt = re.sub(pattern, self._face_prompt, pos_prompt)

        pattern = re.escape(f'{self.ask_symbol}clip{self.ask_symbol}')
        if re.search(pattern, pos_prompt):
            p = interrogate(image, False)
            pos_prompt = re.sub(pattern, p, pos_prompt)

        pattern = re.escape(f'{self.ask_symbol}booru{self.ask_symbol}')
        if re.search(pattern, pos_prompt):
            p = interrogate(image, True)
            pos_prompt = re.sub(pattern, p, pos_prompt)

        with self._input_mutex:
            self._pos_prompt = pos_prompt
            self._neg_prompt = neg_prompt

    def vote(self) -> None:
        if self._received_vote:
            return
        self._received_vote = True
        for key in self.keywords:
                if  self.keywords[key].get_vote():
                    score: float = self.keywords[key].get_score()
                    score = score + 1.0
                    self.keywords[key].set_score(score)
                    self.keywords[key].save()


    def un_vote(self) -> None:
        if not self._received_vote:
            return
        self._received_vote = False
        for key in self.keywords:
                if  self.keywords[key].get_vote():
                    score: float = self.keywords[key].get_score()
                    score = max(score - 1.0, 1.0)
                    self.keywords[key].set_score(score)
                    self.keywords[key].save()


    def get_pos_prompt(self) -> str:
        with self._input_mutex:
            return self._pos_prompt
    def get_neg_prompt(self) -> str:
        with self._input_mutex:
            return self._neg_prompt

    def set_pos_input(self, prompt: str) -> None:
        with self._input_mutex:
            self._pos_input = prompt
    def get_pos_input(self) -> str:
        with self._input_mutex:
            return self._pos_input

    def set_rep_input(self, prompt: str) -> None:
        with self._input_mutex:
            self._rep_input = prompt
    def get_rep_input(self) -> str:
        with self._input_mutex:
            return self._rep_input

    def set_neg_input(self, prompt: str) -> None:
        with self._input_mutex:
            self._neg_input = prompt
    def get_neg_input(self) -> str:
        with self._input_mutex:
            return self._neg_input

    def set_face_prompt(self, prompt: str) -> None:
        with self._input_mutex:
            self._face_prompt = prompt
    def get_face_prompt(self) -> str:
        with self._input_mutex:
            return self._face_prompt


    def set_keyword_id(self, key: str, value: str) -> bool:
        return self.keywords[key].set_id(value)

    def get_keyword_id(self, key: str) -> str:
        return self.keywords[key].get_id()

    def get_keyword_next_id(self, key: str) -> str:
        # hack
        if self.smile_repeat_count >= self.smile_repeat:
            return self.keywords[key].get_next_id()
        else:
            return self.keywords[key].get_id()

    def get_keyword_name(self, key: str) -> str:
        return self.keywords[key].get_name()

    def set_keyword_vote(self, key: str, value: bool) -> None:
        self.keywords[key].set_vote(value)

    def set_keyword_random(self, key: str, value: bool) -> None:
        self.keywords[key].set_active(value)

    def set_keyword_score(self, key: str, value: float) -> None:
        self.keywords[key].set_score(value)

    def get_keyword_score(self, key: str) -> float | None:
        return self.keywords[key].get_score()

    def get_id_from_name(self, key: str, value:str) -> str | None:
        return self.keywords[key].get_id_from_name(value)

    def get_keyword_names(self, key) -> list :
        return self.keywords[key].get_names()

