from enum import IntEnum
import time
from threading import Lock, RLock
from typing import Callable
import pytweening as easing

from apps.smile.render.settings import SequencerSettings, easing_function

class SequenceType(IntEnum):
    NONE            = 0         # Nothing happens, last diffusion shown
    BLOOM           = 1         # Destructing the last diffusion, bloom
    FADE            = 2         # Destructing the last diffusion, fade to camera
    CAMERA_SHOW     = 3         # Show camera, nobody qualifies: the charge is empty
    CAMERA_DETECT   = 4         # A face is charging (or its charge is draining) toward the flash
    FLASH           = 5         # Capture, Flash in
    CAPTURE         = 6         # MOMENT of capture
    ASK             = 7         # Capture, Flash out
    CONTROL_0       = 8         # Preparing first Control Net
    CONTROL_1       = 9         # Preparing second Control Net
    CONTROL_2       = 10        # Preparing third Control Net
    DIFFUSION       = 11        # Diffusion, creating image
    DIFF_LOOP       = 12        # Diffusion, creating images in loop
    TIMEOUT         = 13        # Diffusion, showing image
    RESET           = 14        # MOMENT of reset

SequenceTypeNames: list[str] = [member.name for member in SequenceType]

class AnimationType(IntEnum):
    NONE            = 0         #
    DESTRUCT_IN     = 1         # Destructing the last diffusion, bloom
    DESTRUCT_OUT    = 2         # Destructing the last diffusion, fade to camera
    CAMERA_IN       = 3         # Show camera, Idle
    CAMERA_OUT      = 4         # Showing camera, a face is charging: progress is the charge
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
    if seq == SequenceType.CAMERA_DETECT:
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
        self.charge: float = 0.0            # 0..1: how long a face has held its qualification; 1 starts the flash
        self.charge_time: float = 0.0       # when the charge was last moved

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
        state.start_time = time.perf_counter()

    duration: float = bloom_duration + fade_duration
    value: float = min((time.perf_counter() - state.start_time) / duration, 1.0)
    value = bloom_easing(value)

    progress: float = min((time.perf_counter() - state.start_time) / bloom_duration, 1.0)
    if progress < 1.0:
        return SeqenceMessage(state.type, progress, initial, value)
    else:
        # the first FADE message: its values where a fade has them (value_1 the fade, value_2 the bloom)
        state.type = SequenceType.FADE
        return SeqenceMessage(state.type, 0, initial, 0.0, value)

def fade_message(state: SeqenceState, bloom_duration: float, bloom_easing: Easing, fade_duration: float, fade_easing: Easing) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time2 = time.perf_counter()

    duration: float = bloom_duration + fade_duration
    bloom_value: float = min((time.perf_counter() - state.start_time) / duration, 1.0)
    bloom_value = bloom_easing(bloom_value)

    progress: float = min((time.perf_counter() - state.start_time2) / fade_duration, 1.0)
    omission_value: float = fade_easing(progress)

    if progress < 1.0:
        return SeqenceMessage(state.type, progress, initial, omission_value, bloom_value)
    else:
        state.type = SequenceType.CAMERA_SHOW
        return SeqenceMessage(state.type, 0, initial, 0)

CAMERA_STATES: tuple[SequenceType, ...] = (SequenceType.CAMERA_SHOW, SequenceType.CAMERA_DETECT)

# a detector frame older than this says nothing about who is there now
FACE_STALE: float = 0.5

# the part of the flash in at which the capture is taken (the FLASH state ends there)
CAPTURE_AT: float = 0.5

def camera_message(state: SeqenceState, rising: bool, charge_duration: float, drain_duration: float, easing: Easing = easing.linear) -> SeqenceMessage:
    """The camera phase: the charge rises while a face qualifies and drains while none does.
    CAMERA_SHOW with an empty charge, CAMERA_DETECT while there is some (progress is the charge),
    and the flash starts when it is full."""
    now: float = time.perf_counter()
    if state.previous_type not in CAMERA_STATES:
        # entering the camera phase: an empty charge, and the idle progress starts
        state.charge = 0.0
        state.charge_time = now
        state.start_time = now

    elapsed: float = max(0.0, now - state.charge_time)
    state.charge_time = now
    if rising:
        state.charge += elapsed / max(charge_duration, 1e-6)
    else:
        state.charge -= elapsed / max(drain_duration, 1e-6)
    state.charge = min(max(state.charge, 0.0), 1.0)

    current: SequenceType = SequenceType.CAMERA_DETECT if state.charge > 0.0 else SequenceType.CAMERA_SHOW
    initial: bool = state.previous_type != current
    state.previous_type = current
    state.type = current

    if current == SequenceType.CAMERA_SHOW:
        progress: float = min((now - state.start_time) / 1.0, 1.0)
        return SeqenceMessage(current, progress, initial, 0.0)

    message = SeqenceMessage(current, state.charge, initial, easing(state.charge))
    if state.charge >= 1.0:
        state.charge = 0.0
        state.type = SequenceType.FLASH
    return message

def pre_capture_message(state: SeqenceState, duration: float, easing: Easing, delay: float) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.perf_counter()

    progress: float = min((time.perf_counter() - state.start_time) / duration, 1.0)
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

    progress: float = min((time.perf_counter() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    state.type = SequenceType.ASK
    return SeqenceMessage(state.type, progress, initial, value)

def post_capture_message(state: SeqenceState, duration: float, easing: Easing = easing.linear, hold: float = 0.0, reveal: bool = False) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type

    elapsed: float = time.perf_counter() - state.start_time
    progress: float = min(elapsed / duration, 1.0)
    value: float = easing(progress)
    message = SeqenceMessage(state.type, progress, initial, value)
    # the flash out starts when the capture's depth is in and the full flash has stood for the hold
    if reveal and elapsed >= duration + hold:
        state.type = SequenceType.CONTROL_0
    return message

def control_message(state: SeqenceState, duration: float, easing: Easing = easing.linear, first: bool = False) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        if first:
            state.start_time = time.perf_counter()

    progress: float = min((time.perf_counter() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    return SeqenceMessage(state.type, progress, initial, value)

def diffusion_message(state: SeqenceState, progress: float) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.perf_counter()

    return SeqenceMessage(state.type, progress, initial)

def timeout_message(state: SeqenceState, duration: float, easing: Easing = easing.linear) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type
        state.start_time = time.perf_counter()

    progress: float = min((time.perf_counter() - state.start_time) / duration, 1.0)
    value: float = easing(progress)
    if progress < 1.0:
        return SeqenceMessage(state.type, progress, initial)
    else:
        # progress stays full, as in reset_message: the lights and the music follow it
        state.type = SequenceType.RESET
        return SeqenceMessage(state.type, 1.0, initial, 0)

def reset_message(state: SeqenceState) -> SeqenceMessage:
    initial = False
    if state.previous_type != state.type:
        initial = True
        state.previous_type = state.type

    state.type = SequenceType.BLOOM

    return SeqenceMessage(state.previous_type, 1.0, initial)

MessageCallback = Callable[[SeqenceMessage], None]

class Sequencer():
    """Time-based state machine of a visitor cycle. update() runs once per render frame, on the render
    thread; set_sequence() and the other setters are called from the generator, face and key threads.
    The flash starts when a face has held its qualification long enough (the charge, see
    camera_message). Two steps wait for the generator: the flash holds (ASK) until start_reveal(), and the grin
    (DIFF_LOOP) starts when the render's player starts playing. finish() ends the visitor (TIMEOUT)."""

    def __init__(self, settings: SequencerSettings) -> None:
        self.settings: SequencerSettings = settings
        self._input_mutex: Lock = Lock()
        self._state_mutex: RLock = RLock()
        self._message_callbacks: list[MessageCallback] = []
        self.message: SeqenceMessage = SeqenceMessage(SequenceType.NONE, 0.0)

        self._state: SeqenceState = SeqenceState()
        self._state.type = SequenceType.CAMERA_SHOW
        self._state.previous_type = SequenceType.NONE

        self.first_sequence: bool = True
        self.diffusion_progress: float = 0.0
        self._reveal: bool = False      # the capture's depth is in: the flash out may start
        self._finish: bool = False      # the generator is done: TIMEOUT follows the flash out
        self.blocked: bool = False      # no new sequence: the generator is not ready for a capture, or the raw camera is shown
        self._face_qualified: bool = False  # the newest detector frame has a qualifying face
        self._face_time: float = 0.0        # when that frame was made
        self._forced: bool = False      # a manual start: the charge rises without a face

        self.flash_out_start: float = 0.0
        self.flash_out_calculation: bool = False
        self.flash_out_history: list[float] = []
        self.flash_out_init: bool = True

        self.settings.bind(SequencerSettings.start, self._on_start)
        self.settings.bind(SequencerSettings.reset, self._on_reset)
        self.settings.bind(SequencerSettings.flash_out_auto, self._on_flash_out_auto)

    def close(self) -> None:
        self.settings.unbind(SequencerSettings.start, self._on_start)
        self.settings.unbind(SequencerSettings.reset, self._on_reset)
        self.settings.unbind(SequencerSettings.flash_out_auto, self._on_flash_out_auto)

    def update(self) -> SeqenceMessage:
        # the callbacks run outside the state lock: they may set the sequence themselves
        with self._state_mutex:
            self.message = self._next_message()
        self.message_callback(self.message)
        return self.message

    def _next_message(self) -> SeqenceMessage:
        s: SequencerSettings = self.settings
        message = SeqenceMessage(SequenceType.NONE, 0.0)

        if self._state.type == SequenceType.NONE:
            pass
        elif self._state.type == SequenceType.BLOOM:
            message = bloom_message(self._state, s.destroy_in, easing_function(s.destroy_in_ease), s.destroy_out, easing_function(s.destroy_out_ease))
        elif self._state.type == SequenceType.FADE:
            message = fade_message(self._state, s.destroy_in, easing_function(s.destroy_in_ease), s.destroy_out, easing_function(s.destroy_out_ease))
        elif self._state.type in CAMERA_STATES:
            message = camera_message(self._state, self._rising(), s.charge, s.drain, easing_function(s.charge_ease))
            if self._state.type == SequenceType.FLASH:
                self._forced = False
        elif self._state.type == SequenceType.FLASH:
            message = pre_capture_message(self._state, s.flash_in, easing_function(s.flash_in_ease), CAPTURE_AT)
            if message.initial:
                self._reveal = False
                self._finish = False
        elif self._state.type == SequenceType.CAPTURE:
            message = capture_message(self._state, s.flash_in, easing_function(s.flash_in_ease))
        elif self._state.type == SequenceType.ASK:
            message = post_capture_message(self._state, s.flash_in, easing_function(s.flash_in_ease), s.flash_hold, self._reveal)
        elif self._state.type == SequenceType.CONTROL_0:
            message = control_message(self._state, s.flash_out, easing_function(s.flash_out_ease), True)
            # the generator finished without a portrait to end the flash out: the capture stands as the result
            if self._finish and message.progress >= 1.0:
                self._state.type = SequenceType.TIMEOUT
        elif self._state.type == SequenceType.CONTROL_1:
            message = control_message(self._state, s.flash_out, easing_function(s.flash_out_ease))
        elif self._state.type == SequenceType.CONTROL_2:
            message = control_message(self._state, s.flash_out, easing_function(s.flash_out_ease))
        elif self._state.type == SequenceType.DIFFUSION:
            message = diffusion_message(self._state, self.diffusion_progress)
        elif self._state.type == SequenceType.DIFF_LOOP:
            message = diffusion_message(self._state, self.diffusion_progress)
        elif self._state.type == SequenceType.TIMEOUT:
            message = timeout_message(self._state, s.timeout)
        elif self._state.type == SequenceType.RESET:
            message = reset_message(self._state)

        flash_out: float = self.calculate_flash_out_duration(message, s.flash_out)
        if flash_out != s.flash_out:
            s.flash_out = flash_out

        s.state = message.sequence_type.name
        s.state_progress = message.value_1
        s.animation = message.animation_type.name
        s.animation_progress = message.progress
        return message

    def _on_start(self, _=None) -> None:
        self.start_sequence()

    def _on_reset(self, _=None) -> None:
        self.reset()

    def _on_flash_out_auto(self, _=None) -> None:
        self.flash_out_history.clear()

    def reset(self) -> None:
        self._forced = False
        self.set_sequence(SequenceType.RESET)

    def set_face(self, qualified: bool, face_time: float) -> None:
        """The newest detector frame: whether its subject qualifies, and when it was made."""
        with self._state_mutex:
            self._face_qualified = qualified
            self._face_time = face_time

    def _rising(self) -> bool:
        """Whether the charge rises this frame: a manual start, or a fresh qualifying face while
        new visitors are welcome (not paused, the generator ready)."""
        if self.blocked:
            self._forced = False    # a manual start does not wait for the block to end
            return False
        if self._forced:
            return True
        if self.settings.pause:
            return False    # no new visitors; the start button and the Enter key still work
        return self._face_qualified and time.perf_counter() - self._face_time <= FACE_STALE

    def start_sequence(self) -> None:
        """A manual start: the charge rises to the flash without a face."""
        with self._state_mutex:
            if self.blocked or self._state.type not in CAMERA_STATES:
                return
            self._forced = True


    def start_reveal(self) -> None:
        """The capture's depth is in: start the flash out, once the full flash has stood for flash_hold."""
        with self._state_mutex:
            self._reveal = True

    def finish(self) -> None:
        """The generator is done: the result holds (TIMEOUT). With the capture's depth in and the flash
        still running (nothing was generated: skip_generation, or a failure right after the depth) the
        flash out plays first."""
        with self._state_mutex:
            if self._reveal and SequenceType.FLASH <= self._state.type <= SequenceType.CONTROL_2:
                self._finish = True
            else:
                self._state.type = SequenceType.TIMEOUT

    def set_sequence(self, type: SequenceType) -> None:
        with self._state_mutex:
            self._state.type = type

    def get_sequence(self) -> SequenceType:
        with self._state_mutex:
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
            self.flash_out_start = time.perf_counter()
            self.flash_out_calculation = True

        if not self.settings.flash_out_auto:
            self.flash_out_calculation = False
            return duration

        if message.sequence_type == SequenceType.DIFFUSION:
            if message.initial:
                if self.flash_out_calculation:
                    self.flash_out_calculation = False
                    D: float = time.perf_counter() - self.flash_out_start
                    self.flash_out_history.append(D)
                    if len(self.flash_out_history) > 10:
                        self.flash_out_history.pop(0)
                    duration = sum(self.flash_out_history) / len(self.flash_out_history) - 0.2

        if message.sequence_type < SequenceType.CONTROL_0 or message.sequence_type > SequenceType.CONTROL_2:
            if self.flash_out_calculation:
                self.flash_out_calculation = False

        return max(min(duration, 6.0), 1.0)

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