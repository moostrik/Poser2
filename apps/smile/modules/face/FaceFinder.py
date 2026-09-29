from cv2.face import Facemark
import numpy as np
import time
from deepface import DeepFace
# from deepface.extendedmodels import Age, Emotion, Race, Gender
from threading import Thread, Lock
from typing import Callable
from random import choice
import cv2
import copy
from modules.sd.WebUi import WebUi

# from keras.preprocessing import image
import keras.utils as image

class FaceMessage():
    def __init__(self, img: np.ndarray, size:float, score: float, age: int, age_str: str, emotion: str, gender: str, race: str) -> None:
        self.image: np.ndarray = img
        self.size: float = round(size, 3)
        self.score: float = round(score, 3)
        self.age: int = age
        self.age_str: str = age_str
        self.emotion: str = emotion
        self.gender: str = gender
        self.race: str = race
        self.prompt: str = 'TEST'
        self.holland: list[str] = ['TEST']

MessageCallback = Callable[[FaceMessage], None]

def sd_find_face(web_ui: WebUi, img: np.ndarray, min_size: float, portrait_zoom: float, portrait_offset: float, portrait_width: int, portrait_height: int) -> FaceMessage:
    face_info: dict | None = web_ui.get_face_info(img)
    if face_info is None:
        raise ValueError('no face found')

    score: float = float(face_info['score'])
    image_height, image_width = img.shape[:2]
    i_x0: float = float(face_info['x0'])
    i_x1: float = float(face_info['x1'])
    i_y0: float = float(face_info['y0'])
    i_y1: float = float(face_info['y1'])
    real_size: float = (i_y1 - i_y0)
    normalized_size: float = real_size / float(image_height)

    if normalized_size < min_size:
        raise ValueError('face to small')

    x_Centre: float = (i_x0 + i_x1) * 0.5
    y_Centre: float = (i_y0 + i_y1) * 0.5

    p_AR: float = portrait_width / portrait_height
    p_x0: int = int(x_Centre - real_size * portrait_zoom * 0.5)
    p_x1: int = int(x_Centre + real_size * portrait_zoom * 0.5)
    p_y0: int = int(y_Centre - real_size * portrait_zoom * 0.5 / p_AR - (portrait_offset * real_size))
    p_y1: int = int(y_Centre + real_size * portrait_zoom * 0.5 / p_AR - (portrait_offset * real_size))
    if p_x0 < 0 or p_x1 > image_width or p_y0 < 0 or p_y1 > image_height:
        raise ValueError('portrait out of bounds')

    portrait_img: np.ndarray = cv2.resize(img[int(p_y0):int(p_y1),int(p_x0):int(p_x1)], (portrait_width, portrait_height))


    a_x0: int = int(x_Centre - real_size * 0.5)
    a_x1: int = int(x_Centre + real_size * 0.5)
    a_y0: int = int(i_y0)
    a_y1: int = int(i_y1)

    if a_x0 < 0 or a_x1 > image_width:
        raise ValueError('face out of bounds')
    attribute_img: np.ndarray = cv2.resize(img[a_y0:a_y1,a_x0:a_x1], (224, 224))

    attribute_pixels: np.ndarray = image.img_to_array(attribute_img)  # what this line doing? must?
    attribute_pixels = np.expand_dims(attribute_pixels, axis=0)
    attribute_pixels /= 255  # normalize input in [0, 1]

    age: int = find_age(attribute_pixels)
    age_str: str = find_age_string(age)
    emotion: str = find_emotion(attribute_pixels)
    gender: str = find_gender(attribute_pixels)
    race: str = find_race(attribute_pixels)

    return FaceMessage(portrait_img, normalized_size, score, age, age_str, emotion, gender, race)

def find_age(img: np.ndarray) -> int:
    model = DeepFace.build_model("Age", "facial_attribute")
    age = model.predict(img)
    return int(age)

def find_emotion(img: np.ndarray) -> str:
    model = DeepFace.build_model("Emotion", "facial_attribute")
    emotion_predictions = model.predict(img)
    labels: list[str] = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    emotion: str = labels[np.argmax(emotion_predictions)]
    return emotion

def find_gender(img: np.ndarray) -> str:
    model = DeepFace.build_model("Gender", "facial_attribute")
    gender_predictions = model.predict(img)
    if gender_predictions[0] > 0.33 and gender_predictions[1] > 0.33:
        return "androgyne"
    else:
        labels: list[str] = ["Woman", "Man"]
        return labels[np.argmax(gender_predictions)]

def find_race(img: np.ndarray) -> str:
    model = DeepFace.build_model("Race", "facial_attribute")
    race_predictions = model.predict(img)

    labels: list[str] = ["asian", "indian", "black", "white", "middle eastern", "latino"]
    race: str = labels[np.argmax(race_predictions)]
    return race

def find_politics() -> str:
    politics: list[str] = ['far-left', 'left', 'liberal', 'centre', 'conservative', 'right', 'far-right']
    return choice(politics)

def find_issue() -> str:
    issues: list[str] = ['climate', 'immigration', 'economy', 'human-rights', 'ukraine', 'terrorism', 'religion']
    return choice(issues)

def find_age_string(age: int) -> str:
    age_string: str = ''
    if age < 13:
        age_string = 'kid'
    elif age < 20:
        age_string = 'teen'
    elif age < 30:
        age_string = '20yo'
    elif age < 45:
        age_string = 'adult'
    elif age < 65:
        age_string = 'mature'
    else:
        age_string = 'old'
    return age_string

def make_prompt(face:FaceMessage, emotion: bool = True, age: bool = True, race: bool = True, gender: bool = True) -> None:
    prompt: str = ''
    if emotion:
        e: str = face.emotion
        if e == 'fear':
            e = 'frightened'
        if e == 'surprise':
            e = 'surprised'
        if e == 'disgust':
            e = 'disgusted'
        if e == 'neutral':
            e = ''
        prompt += e + ' '
    if age:
        prompt += face.age_str + ' '
    if race:
        prompt += face.race + ' '
    if gender:
        prompt += face.gender + ' '
    face.prompt = prompt

def make_holland(face:FaceMessage) -> None:
    d_p: list[str] = ['WHAT AI ADVERTISERS THINK ABOUT YOU', '', '']
    d_p[1] += 'emotion: '    + face.emotion.upper()
    d_p[1] += '  age: '      + str(face.age)
    d_p[1] += '  race: '     + face.race.upper()
    d_p[1] += '  gender: '   + face.gender.upper()

    d_p[2] += 'politics: '   + find_politics().upper()
    d_p[2] += '  most important issue: ' + find_issue().upper()
    face.holland = d_p

class FaceFinder(Thread):
    def __init__(self, composition_width: int, composition_height: int, web_ui: WebUi) -> None:
        super().__init__()
        self._input_mutex: Lock = Lock()
        self._thread_running: bool = False
        self._image: np.ndarray | None = None
        self._message_callbacks: list[MessageCallback] = []

        self._composition_width: int = composition_width
        self._composition_height: int = composition_height

        self.size_filter: float = 0.1
        self.zoom: float = 1.0
        self.offset: float = 0.0

        self.prompt_emotion: bool = True
        self.prompt_age: bool = True
        self.prompt_gender: bool = True
        self.prompt_race: bool = True

        self.web_ui: WebUi = web_ui

    def stop(self) -> None:
        if (self._thread_running):
            self._thread_running = False
            self.join()

    def run(self) -> None:
        self._thread_running = True
        while self._thread_running:
            image: np.ndarray | None = self.get_image()
            self.set_image(None)
            if image is not None:
                try:
                    face_info: FaceMessage | None = sd_find_face(self.web_ui, image, self.size_filter, self.zoom, self.offset, self._composition_width, self._composition_height)
                    make_prompt(face_info, self.prompt_emotion, self.prompt_age, self.prompt_race, self.prompt_gender)
                    make_holland(face_info)
                    self.message_callback(face_info)
                except ValueError as e:
                    self.error_callback(e)

            else:
                time.sleep(0.01)

    def set_image(self, image: np.ndarray | None) -> None:
        with self._input_mutex:
            self._image = image
    def get_image(self) -> np.ndarray | None:
        with self._input_mutex:
            return self._image

    def message_callback(self, message: FaceMessage) -> None:
        with self._input_mutex:
            for c in self._message_callbacks: c(message)

    def error_callback(self, error: ValueError) -> None:
        print(error)

    def add_message_callback(self, callback: MessageCallback) -> None:
        with self._input_mutex:
            self._message_callbacks.append(callback)

    def remove_message_callback(self, callback: MessageCallback) -> None:
        with self._input_mutex:
            if callback in self._message_callbacks:
                self._message_callbacks.remove(callback)

    def clear_message_callbacks(self) -> None:
        with self._input_mutex:
            self._message_callbacks = []