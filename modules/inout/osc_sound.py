from threading import Thread, Event, Lock
from collections import deque
from typing import Protocol
import time

import numpy as np

from modules.pose import Frame, FrameDict, Angles, AngleVelocity, AngleSymmetry, Similarity, MotionGate, LeaderScore, MotionTime, Age, AngleLandmark, BBox, LegDeviation, TorsoTilt

from modules.session import SequencerState
from modules.settings import BaseSettings, Field, Widget
from .osc_client import OscClient
from .osc_encoder import OscMessage, encode_message

import logging
logger = logging.getLogger(__name__)

# Sent as /global/state when there is no show running (shutdown blackout). Shutdown is
# not a show stage, and receivers key scenes off this integer — an out-of-band value
# keeps them from firing a cue for whichever stage happens to be numbered 0.
_NO_STATE = -1

_NAN_BBOX: np.ndarray = np.full(4, np.nan, dtype=np.float32)


class OscSoundSettings(BaseSettings):
    enabled: Field[bool]     = Field(True,                      widget=Widget.switch,       description="Send OSC to the sound; off sends one blackout")
    max_players: Field[int]  = Field(4,    min=1,    max=16,    access=Field.INIT,         description="Max number of player poses to send (IDs 0 to N-1)")
    ip: Field[str]           = Field("127.0.0.1",               widget=Widget.ip_field,     description="Target OSC IP address")
    port: Field[int]         = Field(9000, min=1024, max=65535, widget=Widget.number_field, description="Target OSC port")
    stage: Field[int]        = Field(0,                                                     description="Pipeline stage to read poses from")
    active_players: Field[int] = Field(0, access=Field.READ,                                description="Current number of active players being sent")
    interval_mean_ms: Field[float] = Field(0.0, access=Field.READ,                          description="Mean interval between sent OSC bundles (ms)")
    interval_min_ms: Field[float]  = Field(0.0, access=Field.READ,                          description="Shortest interval between sent OSC bundles (ms)")
    interval_max_ms: Field[float]  = Field(0.0, access=Field.READ,                          description="Longest interval between sent OSC bundles (ms)")


class OscSoundExtras(Protocol):
    """An app's own messages in the sound's bundle."""

    def global_messages(self, frames: FrameDict, seq_state: SequencerState | None) -> list[OscMessage]:
        """After the tick-wide messages; `seq_state` None is the blackout."""
        ...

    def player_messages(self, frame: Frame) -> list[OscMessage]:
        """After a present player's messages."""
        ...

    def inactive_messages(self, id: int) -> list[OscMessage]:
        """After an empty slot's reset; the same every time, since the reset is encoded once."""
        ...

    def angle_values(self, frame: Frame) -> np.ndarray:
        """The nine values of ``/pose/{id}/angle/rad``, in ``AngleLandmark`` order."""
        ...


class OscSound:
    """
    Sends smooth pose data over OSC at a configurable frame rate in its own thread.

    It sends `slot_count` slots (default `max_players`); the per-player rows (similarity, gate, leader) stay
    `max_players` wide. An app adds its own messages through `extras`.
    """
    def __init__(self, settings: OscSoundSettings, slot_count: int | None = None, extras: OscSoundExtras | None = None) -> None:

        self.settings: OscSoundSettings = settings
        self._slot_count: int = slot_count if slot_count is not None else settings.max_players
        self._extras: OscSoundExtras | None = extras
        self._frames: FrameDict = {}
        self._seq_state: SequencerState | None = None
        self._input_lock: Lock = Lock()
        self._client: OscClient = OscClient(settings, OscSoundSettings.ip, OscSoundSettings.port, 'OscSound')
        self._blacked_out: bool = False     # switched off, the blackout went out: nothing more until switched on

        # Send-interval measurement (rolling window of inter-send gaps).
        self._last_send_time: float | None = None
        self._interval_samples: deque[float] = deque(maxlen=120)

        self._running = False
        self._update_event: Event = Event()
        self._thread: Thread | None = None
        # Throttle: only send a slot's inactive reset for the first 2 ticks after it empties.
        self._inactive_counts: dict[int, int] = {id: 0 for id in range(self._slot_count)}
        self._inactive_encoded: dict[int, list[bytes]] = {}     # each slot's reset, encoded at its first use

    @property
    def running(self) -> bool:
        return self._running

    def submit_frames(self, stage: int, frames: FrameDict) -> None:
        """Push new pose data and trigger a send if stage matches."""
        if stage != self.settings.stage:
            return
        with self._input_lock:
            self._frames = dict(frames)
        self._update_event.set()

    def set_sequencer_state(self, state: SequencerState) -> None:
        """Push new sequencer state."""
        with self._input_lock:
            self._seq_state = state

    def start(self) -> None:
        """Start the OSC sender thread"""
        if self._thread is not None and self._thread.is_alive():
            return

        self._client.start()
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name='OscSound')
        self._thread.start()

    def stop(self) -> None:
        """Stop the OSC sender thread; a blackout goes out unless one went out when it was switched off"""
        if self._thread is None:
            return
        self._running = False
        self._update_event.set()
        self._thread.join()
        self._thread = None
        if not self._blacked_out:
            try:
                self._client.send(self._blackout_messages())
            except Exception:
                logger.exception('OscSound: the blackout at stop was not sent')
        self._client.stop()

    def _run(self) -> None:
        """Sends a bundle for each submit; switched off, one blackout and then nothing"""
        while self._running:
            self._update_event.wait()
            self._update_event.clear()
            if not self._running:
                break
            try:
                if self.settings.enabled:
                    self._blacked_out = False
                    self._record_interval()
                    with self._input_lock:
                        frames = self._frames
                        seq_state = self._seq_state
                    self._client.send(self._tick_messages(frames, seq_state))
                elif not self._blacked_out:
                    self._blacked_out = True
                    self._client.send(self._blackout_messages())
            except Exception:
                logger.exception('OscSound: a bundle was not sent')

    def _record_interval(self) -> None:
        """Track the gap between consecutive sends and publish mean/deviation (ms)."""
        now = time.perf_counter()
        if self._last_send_time is not None:
            self._interval_samples.append(now - self._last_send_time)
        self._last_send_time = now
        if self._interval_samples:
            samples = np.fromiter(self._interval_samples, dtype=float)
            self.settings.interval_mean_ms = float(samples.mean() * 1000.0)
            self.settings.interval_min_ms = float(samples.min() * 1000.0)
            self.settings.interval_max_ms = float(samples.max() * 1000.0)

    def _blackout_messages(self) -> list[bytes]:
        """A final all-zero tick so the receiver ends in a clean state: zeroed globals and every slot's reset.
        Composes the same leaf builders as the live path, so `_tick_messages` stays blackout-agnostic."""
        encoded: list[bytes] = _encode(self._global_messages({}, None))    # no frames, None → zeroed globals
        for id in range(self._slot_count):
            encoded += self._inactive_reset(id)
        self.settings.active_players = 0
        return encoded

    def _tick_messages(self, frames: FrameDict, seq_state: SequencerState | None) -> list[bytes]:
        encoded: list[bytes] = _encode(self._global_messages(frames, seq_state))
        self.settings.active_players = sum(1 for id in range(self._slot_count) if id in frames)

        for id in range(self._slot_count):
            if id in frames:
                self._inactive_counts[id] = 0
                encoded += _encode(self._active_messages(frames[id]))
            elif self._inactive_counts[id] < 2:
                # Only send a slot's inactive reset for the first 2 ticks after it empties.
                self._inactive_counts[id] += 1
                encoded += self._inactive_reset(id)
        return encoded

    def _inactive_reset(self, id: int) -> list[bytes]:
        """A slot's reset, encoded once."""
        reset: list[bytes] | None = self._inactive_encoded.get(id)
        if reset is None:
            reset = _encode(self._inactive_messages(id))
            self._inactive_encoded[id] = reset
        return reset

    def _global_messages(self, frames: FrameDict, seq_state: SequencerState | None) -> list[OscMessage]:
        """Tick-wide messages, built from the tick's frames and sequencer state. A None
        seq_state yields the all-zero (idle/blackout) globals."""
        messages: list[OscMessage] = [
            ("/global/state", (int(seq_state.stage) if seq_state is not None else _NO_STATE,)),
            ("/global/state/progress", (float(seq_state.stage_progress) if seq_state is not None else 0.0,)),
            ("/global/progress", (float(seq_state.progress) if seq_state is not None else 0.0,)),
        ]
        if self._extras is not None:
            messages += self._extras.global_messages(frames, seq_state)
        return messages

    def _inactive_messages(self, id: int) -> list[OscMessage]:
        zeros: np.ndarray = np.zeros(self.settings.max_players, dtype=np.float32)
        angles: np.ndarray = np.zeros(len(AngleLandmark), dtype=np.float32)
        messages: list[OscMessage] = [
            (f"/pose/{id}/active", (0,)),
            (f"/pose/{id}/bbox", (np.zeros(4, dtype=np.float32),)),     # center_x, center_y, width, height
            (f"/pose/{id}/time/motion", (0.0,)),
            (f"/pose/{id}/time/age", (0.0,)),
            (f"/pose/{id}/angle/rad", (angles,)),
            (f"/pose/{id}/angle/vel", (angles,)),
            (f"/pose/{id}/angle/sym", (np.zeros(AngleSymmetry.length(), dtype=np.float32),)),
            (f"/pose/{id}/angle/legs", (0.0,)),
            (f"/pose/{id}/angle/tilt", (0.0,)),
            (f"/pose/{id}/similarity", (zeros,)),
            (f"/pose/{id}/similarity/pose", (zeros,)),
            (f"/pose/{id}/similarity/gate", (zeros,)),
            (f"/pose/{id}/similarity/motion", (zeros,)),
            (f"/pose/{id}/similarity/leader", (zeros,)),
        ]
        if self._extras is not None:
            messages += self._extras.inactive_messages(id)
        return messages

    def _active_messages(self, frame: Frame) -> list[OscMessage]:
        id: int = frame.track_id
        zeros: np.ndarray = np.zeros(self.settings.max_players, dtype=np.float32)
        # range [0, 1] - pose similarity per player; NaN (no pair, or no such player) is sent as 0
        similarity: np.ndarray = np.nan_to_num(frame[Similarity].values, nan=0.0) if Similarity in frame else zeros
        angles: np.ndarray = self._extras.angle_values(frame) if self._extras is not None else frame[Angles].values
        messages: list[OscMessage] = [
            (f"/pose/{id}/active", (1,)),
            # center_x, center_y, width, height
            (f"/pose/{id}/bbox", (frame[BBox].values if BBox in frame else _NAN_BBOX,)),
            # when there is normal motion, about 1 per second
            (f"/pose/{id}/time/motion", (float(frame[MotionTime].value) if MotionTime in frame else 0.0,)),
            # in seconds
            (f"/pose/{id}/time/age", (float(frame[Age].value) if Age in frame else 0.0,)),
            # range [-pi, pi]
            (f"/pose/{id}/angle/rad", (angles,)),
            # range [-finite, finite] -> [-2pi, 2pi]
            (f"/pose/{id}/angle/vel", (frame[AngleVelocity].values,)),
            # range [-1, 1] - signed left-minus-right per pair: shoulder, elbow, hip, knee, arms, legs
            (f"/pose/{id}/angle/sym", (frame[AngleSymmetry].values if AngleSymmetry in frame else np.zeros(AngleSymmetry.length(), dtype=np.float32),)),
            # range [0, 1] - joint-weighted leg deviation from standing straight
            (f"/pose/{id}/angle/legs", (float(frame[LegDeviation].value) if LegDeviation in frame else 0.0,)),
            # range [-1, 1] - signed sideways torso lean against the image vertical
            (f"/pose/{id}/angle/tilt", (float(frame[TorsoTilt].value) if TorsoTilt in frame else 0.0,)),
            (f"/pose/{id}/similarity/pose", (similarity,)),
            # range [0, 1] - motion gate (both poses moving)
            (f"/pose/{id}/similarity/gate", (frame[MotionGate].values if MotionGate in frame else zeros,)),
            # range [0, 1] - the same row as similarity/pose
            (f"/pose/{id}/similarity/motion", (similarity,)),
            # range [-1, 1] - leader scores (negative=this leads, positive=other leads)
            (f"/pose/{id}/similarity/leader", (frame[LeaderScore].values if LeaderScore in frame else zeros,)),
            # range [0, 1] - DEPRECATED: the same row as similarity/pose, on the old address
            (f"/pose/{id}/similarity", (similarity,)),
        ]
        if self._extras is not None:
            messages += self._extras.player_messages(frame)
        return messages


def _encode(messages: list[OscMessage]) -> list[bytes]:
    return [encode_message(address, args) for address, args in messages]
