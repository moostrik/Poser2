from enum import IntEnum
import time
from threading import Lock
from typing import Callable
import pytweening as easing

class SequenceType(IntEnum):
    NONE            = 0         # Nothing happens, last diffusion shown
    BLOOM           = 1         # Destructing the last diffusion, bloom
    FADE            = 2         # Destructing the last diffusion, fade to camera
    CAMERA_SHOW     = 3         # Show camera (not face)
    CAMERA_DETECT   = 4         # Record faces
    CAMERA_REC      = 5         # Record faces
    FLASH           = 6         # Capture, Flash in
    CAPTURE         = 7         # MOMENT of capture
    ASK             = 8         # Capture, Flash out
    CONTROL_0       = 9         # Preparing first Control Net
    CONTROL_1       = 10        # Preparing second Control Net
    CONTROL_2       = 11        # Preparing third Control Net
    DIFFUSION       = 12        # Diffusion, creating image
    DIFF_LOOP       = 13        # Diffusion, creating images in loop
    TIMEOUT         = 14        # Diffusion, showing image
    RESET           = 15        # MOMENT of reset

SequenceTypeNames: list[str] = [member.name for member in SequenceType]

class AnimationType(IntEnum):
    NONE            = 0         #
    DESTRUCT_IN     = 1         # Destructing the last diffusion, bloom
    DESTRUCT_OUT    = 2         # Destructing the last diffusion, fade to camera
    CAMERA_IN       = 3         # Show camera, Idle
    CAMERA_OUT      = 4         # Showing camera, Person found
    FLASH_IN        = 5         # Flash in, the capture routine
    FLASH_OUT       = 6         # Flash out, prepearing control nets
    DIFFUSE_IN      = 7         # Diffusion, creating image
    DIFFUSE_OUT     = 8         # Diffusion, looping, showing image

AnimationTypeNames: list[str] = [member.name for member in AnimationType]

def ani_type_from_seq_type(seq: SequenceType) -> AnimationType:
    if seq == SequenceType.BLOOM:
        return AnimationType.DESTRUCT_IN
    if seq == SequenceType.FADE:
        return AnimationType.DESTRUCT_OUT
    if seq == SequenceType.CAMERA_SHOW:
        return AnimationType.CAMERA_IN
    if seq >= SequenceType.CAMERA_DETECT and seq <= SequenceType.CAMERA_REC:
        return AnimationType.CAMERA_OUT
    if seq >= SequenceType.FLASH and seq <= SequenceType.ASK:
        return AnimationType.FLASH_IN
    if seq >= SequenceType.CONTROL_0 and seq <= SequenceType.CONTROL_2:
        return AnimationType.FLASH_OUT
    if seq == SequenceType.DIFFUSION:
        return AnimationType.DIFFUSE_IN
    if seq >= SequenceType.DIFF_LOOP and seq <= SequenceType.RESET:
        return AnimationType.DIFFUSE_OUT
    return AnimationType.NONE

class SeqenceState():
    def __init__(self) -> None:
        self.type: SequenceType = SequenceType.NONE
        self.previous_type: SequenceType = SequenceType.NONE
        self.start_time: float = 0.0
        self.start_time2: float = 0.0

class SeqenceMessage():
    def __init__(self, type: SequenceType, progress: float, initial: bool = False, value_1: float = 0.0, value_2: float = 0.0) -> None:
        self.sequence_type: SequenceType = type
        self.animation_type: AnimationType = ani_type_from_seq_type(type)
        self.progress: float = progress
        self.initial: bool = initial
        self.value_1: float = value_1
        self.value_2: float = value_2

Easing = Callable[[float], float]

def bloom_message(state: SeqenceState, bloom_duration: float, bloom_easing: Easing, fade_duration: float, fade_easing: Easing) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.time()

    duration: float = bloom_duration + fade_duration
    value: float = min((time.time() - state.start_time) / duration, 1.0)
    value = bloom_easing(value)

    progress: float = min((time.time() - state.start_time) / bloom_duration, 1.0)
    if progress < 1.0:
        return SeqenceMessage(state.type, progress, initial, value)
    else:
        state.type = SequenceType.FADE
        return SeqenceMessage(state.type, 0, initial, value)

def fade_message(state: SeqenceState, bloom_duration: float, bloom_easing: Easing, fade_duration: float, fade_easing: Easing) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time2 = time.time()

    duration: float = bloom_duration + fade_duration
    bloom_value: float = min((time.time() - state.start_time) / duration, 1.0)
    bloom_value = bloom_easing(bloom_value)

    progress: float = min((time.time() - state.start_time2) / fade_duration, 1.0)
    omission_value: float = fade_easing(progress)

    if progress < 1.0:
        return SeqenceMessage(state.type, progress, initial, omission_value, bloom_value)
    else:
        state.type = SequenceType.CAMERA_SHOW
        return SeqenceMessage(state.type, 0, initial, 0)

def cam_show_message(state: SeqenceState, duration: float, easing: Easing = easing.linear) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.time()

    progress: float = min((time.time() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    state.type = SequenceType.CAMERA_SHOW
    return SeqenceMessage(state.type, progress, initial, value)

def cam_detect_message(state: SeqenceState, duration: float, easing: Easing = easing.linear) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.time()

    progress: float = min((time.time() - state.start_time) / duration, 1.0)
    value: float = easing(progress)

    if progress < 1.0:
        state.type = SequenceType.CAMERA_DETECT
        return SeqenceMessage(state.type, progress, initial, value)
    else:
        state.type = SequenceType.CAMERA_REC
        return SeqenceMessage(state.type, 1.0)

def cam_record_message(state: SeqenceState, duration: float, easing: Easing = easing.linear) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.time()

    progress: float = min((time.time() - state.start_time) / duration, 1.0)
    value: float = easing(progress)

    if progress < 1.0:
        state.type = SequenceType.CAMERA_REC
        return SeqenceMessage(state.type, progress, initial, value)
    else:
        state.type = SequenceType.FLASH
        return SeqenceMessage(state.type, 1.0, initial)

def pre_capture_message(state: SeqenceState, duration: float, easing: Easing, delay: float) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.time()

    progress: float = min((time.time() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    if progress < delay:
        state.type = SequenceType.FLASH
        return SeqenceMessage(state.type, progress, initial, value)
    else:
        state.type = SequenceType.CAPTURE
        return SeqenceMessage(state.type, progress, initial, value)

def capture_message(state: SeqenceState, duration: float, easing: Easing = easing.linear) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type

    progress: float = min((time.time() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    state.type = SequenceType.ASK
    return SeqenceMessage(state.type, progress, initial, value)

def post_capture_message(state: SeqenceState, duration: float, easing: Easing = easing.linear) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type

    progress: float = min((time.time() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    return SeqenceMessage(state.type, progress, initial, value)

def control_message(state: SeqenceState, duration: float, easing: Easing = easing.linear, first: bool = False) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        if first:
            state.start_time = time.time()

    progress: float = min((time.time() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    return SeqenceMessage(state.type, progress, initial, value)

def diffusion_message(state: SeqenceState, progress: float) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.time()

    return SeqenceMessage(state.type, progress, initial)

def timeout_message(state: SeqenceState, duration: float, easing: Easing = easing.linear) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.time()

    progress: float = min((time.time() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    if progress < 1.0:
        return SeqenceMessage(state.type, progress, initial)
    else:
        state.type = SequenceType.RESET
        return SeqenceMessage(state.type, 0, initial, 0)

def reset_message(state: SeqenceState) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type

    state.type = SequenceType.BLOOM

    return SeqenceMessage(state.previous_type, 1.0, initial)

MessageCallback = Callable[[SeqenceMessage], None]

class Sequencer():
    def __init__(self) -> None:
        self._input_mutex: Lock = Lock()
        self._message_callbacks: list[MessageCallback] = []
        self.message: SeqenceMessage

        self._state: SeqenceState = SeqenceState()
        self._state.type = SequenceType.CAMERA_SHOW
        self._state.previous_type = SequenceType.NONE

        self.destroy_in_duration: float = 1.0
        self.destroy_in_ease: Callable[[float], float] = easing.linear
        self.destroy_out_duration: float = 1.0
        self.destroy_out_ease: Callable[[float], float] = easing.linear

        self.cam_det_duration: float = 1.0
        self.cam_det_ease: Callable[[float], float] = easing.linear
        self.cam_rec_duration: float = 1.0
        self.cam_rec_ease: Callable[[float], float] = easing.linear
        self.flash_in_duration: float = 1.0
        self.flash_in_ease: Callable[[float], float] = easing.linear
        self.flash_out_duration: float = 1.0
        self.flash_out_ease: Callable[[float], float] = easing.linear
        self.flash_out_auto: bool = False
        self.timout_duration: float = 1.0

        self.first_sequence: bool = True
        self.diffusion_progress: float = 0.0

        self.flash_out_start: float = 0.0
        self.flash_out_calculation: bool = False
        self.flash_out_history: list[float] = []
        self.flash_out_init: bool = True

    def update(self) -> SeqenceMessage:
        self.message = SeqenceMessage(SequenceType.NONE, 0.0)

        if self._state.type == SequenceType.NONE:
            pass
        elif self._state.type == SequenceType.BLOOM:
            self.message = bloom_message(self._state, self.destroy_in_duration, self.destroy_in_ease, self.destroy_out_duration, self.destroy_out_ease)
        elif self._state.type == SequenceType.FADE:
            self.message = fade_message(self._state, self.destroy_in_duration, self.destroy_in_ease, self.destroy_out_duration, self.destroy_out_ease)
        elif self._state.type == SequenceType.CAMERA_SHOW:
            self.message = cam_show_message(self._state, 1.0)
        elif self._state.type == SequenceType.CAMERA_DETECT:
            self.message = cam_detect_message(self._state, self.cam_det_duration, self.cam_det_ease)
        elif self._state.type == SequenceType.CAMERA_REC:
            self.message = cam_record_message(self._state, self.cam_rec_duration, self.cam_rec_ease)
        elif self._state.type == SequenceType.FLASH:
            self.message = pre_capture_message(self._state, self.flash_in_duration, self.flash_in_ease, 0.5)
        elif self._state.type == SequenceType.CAPTURE:
            self.message = capture_message(self._state, self.flash_in_duration, self.flash_in_ease)
        elif self._state.type == SequenceType.ASK:
            self.message = post_capture_message(self._state, self.flash_in_duration, self.flash_in_ease)
        elif self._state.type == SequenceType.CONTROL_0:
            self.message = control_message(self._state, self.flash_out_duration, self.flash_out_ease, True)
        elif self._state.type == SequenceType.CONTROL_1:
            self.message = control_message(self._state, self.flash_out_duration, self.flash_out_ease)
        elif self._state.type == SequenceType.CONTROL_2:
            self.message = control_message(self._state, self.flash_out_duration, self.flash_out_ease)
        elif self._state.type == SequenceType.DIFFUSION:
            self.message = diffusion_message(self._state, self.diffusion_progress)
        elif self._state.type == SequenceType.DIFF_LOOP:
            self.message = diffusion_message(self._state, self.diffusion_progress)
        elif self._state.type == SequenceType.TIMEOUT:
            self.message = timeout_message(self._state, self.timout_duration)
        elif self._state.type == SequenceType.RESET:
            self.message = reset_message(self._state)

        self.flash_out_duration = self.calculate_flash_out_duration(self.message, self.flash_out_duration)

        self.message_callback(self.message)
        return self.message

    def reset(self) -> None:
        self.set_sequence(SequenceType.RESET)

    def set_face_found(self) -> None:
        self.start_sequence()

    def start_sequence(self) -> None:
        if self.get_sequence() != SequenceType.CAMERA_SHOW:
            return
        self.set_sequence(SequenceType.CAMERA_DETECT)


    def set_sequence(self, type: SequenceType) -> None:
        self._state.type = type

    def get_sequence(self) -> SequenceType:
        return self._state.type

    def set_diffusion_progress(self, value) -> None:
        self.diffusion_progress = value

    def calculate_flash_out_duration(self, message: SeqenceMessage, current_duration: float) -> float:
        duration: float = current_duration

        if message.sequence_type == SequenceType.CONTROL_0 and message.initial:
            if self.flash_out_init:
                self.flash_out_init = False
                self.flash_out_calculation = False
                return duration
            self.flash_out_start = time.time()
            self.flash_out_calculation = True

        if not self.flash_out_auto:
            self.flash_out_calculation = False
            return duration

        if message.sequence_type == SequenceType.DIFFUSION:
            if message.initial:
                if self.flash_out_calculation:
                    self.flash_out_calculation = False
                    D: float = time.time() - self.flash_out_start
                    self.flash_out_history.append(D)
                    if len(self.flash_out_history) > 10:
                        self.flash_out_history.pop(0)
                    duration = sum(self.flash_out_history) / len(self.flash_out_history) - 0.2

        if message.sequence_type < SequenceType.CONTROL_0 or message.sequence_type > SequenceType.CONTROL_2:
            if self.flash_out_calculation:
                self.flash_out_calculation = False

        return max(min(duration, 6.0), 1.0)

    def set_flash_out_auto(self, value: bool):
        self.flash_out_auto = value
        self.flash_out_history.clear()

    def message_callback(self, message: SeqenceMessage) -> None:
        with self._input_mutex:
            for c in self._message_callbacks: c(message)

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