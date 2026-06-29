from threading import Thread, Event, Lock
from collections import deque
import socket
import time

import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from modules.pose.frame import Frame, FrameDict
from modules.pose.features import Angles, AngleVelocity, AngleSymmetry, Similarity, MotionGate, LeaderScore, MotionTime, Age, AngleLandmark, BBox

from modules.session import SequencerState
from modules.settings import BaseSettings, Field, Widget
from .net_probe import validate_connection

import logging
logger = logging.getLogger(__name__)

class OscSoundSettings(BaseSettings):
    max_players: Field[int]  = Field(4,    min=1,    max=16,    access=Field.INIT,         description="Max number of player poses to send (IDs 0 to N-1)")
    ip_addresses: Field[str] = Field("127.0.0.1",               widget=Widget.ip_field,     description="Target OSC IP address")
    port: Field[int]         = Field(9000, min=1024, max=65535, widget=Widget.number_field, description="Target OSC port")
    stage: Field[int]        = Field(0,                                                     description="Pipeline stage to read poses from")
    active_players: Field[int] = Field(0, access=Field.READ,                                description="Current number of active players being sent")
    interval_mean_ms: Field[float] = Field(0.0, access=Field.READ,                          description="Mean interval between sent OSC bundles (ms)")
    interval_min_ms: Field[float]  = Field(0.0, access=Field.READ,                          description="Shortest interval between sent OSC bundles (ms)")
    interval_max_ms: Field[float]  = Field(0.0, access=Field.READ,                          description="Longest interval between sent OSC bundles (ms)")


class OscSound:
    """
    Sends smooth pose data over OSC at a configurable frame rate in its own thread.
    """
    def __init__(self, settings: OscSoundSettings) -> None:

        self._config: OscSoundSettings = settings
        self._poses: FrameDict = {}
        self._seq_state: SequencerState | None = None
        self._input_lock: Lock = Lock()
        self._client_lock: Lock = Lock()
        self._client: SimpleUDPClient = SimpleUDPClient(self._config.ip_addresses, self._config.port)

        logger.info(f"SoundOSC client initialized to {self._config.ip_addresses}:{self._config.port}")

        self._config.bind(OscSoundSettings.ip_addresses, self._on_connection_change)  # type: ignore[arg-type]
        self._config.bind(OscSoundSettings.port, self._on_connection_change)          # type: ignore[arg-type]

        # Send-interval measurement (rolling window of inter-send gaps).
        self._last_send_time: float | None = None
        self._interval_samples: deque[float] = deque(maxlen=120)

        self._running = False
        self._update_event: Event = Event()
        self._thread: Thread | None = None
        # Throttle: only send a player's inactive reset for the first 2 ticks after it disappears.
        self._inactive_counts: dict[int, int] = {id: 0 for id in range(self._config.max_players)}

    @property
    def running(self) -> bool:
        return self._running

    def set_frames(self, stage: int, poses: FrameDict) -> None:
        """Push new pose data and trigger a send if stage matches."""
        if stage != self._config.stage:
            return
        with self._input_lock:
            self._poses = dict(poses)
        self._update_event.set()

    def set_sequencer_state(self, state: SequencerState) -> None:
        """Push new sequencer state."""
        with self._input_lock:
            self._seq_state = state

    def start(self) -> None:
        """Start the OSC sender thread"""
        if self._thread is not None and self._thread.is_alive():
            return

        self._running = True
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the OSC sender thread"""
        self._running = False
        self._update_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)  # Wait up to 1 second for thread to exit
            self._thread = None
        try:
            self._send_blackout()
        except Exception as e:
            logger.error(f"Error sending SoundOSC blackout: {e}")

    def _run(self) -> None:
        """Main loop for sending OSC data at regular intervals"""
        # Validate network and IP before starting
        if not validate_connection(self._config.ip_addresses, self._config.port, "OscSound"):
            self._running = False
            return

        while self._running:
            self._update_event.wait()
            self._update_event.clear()
            if self._running:

                self._record_interval()
                try:
                    self._send_data()
                except socket.error as e:
                    logger.error(f"SoundOSC Socket error with exception: {e}")
                    self._running = False
                    break
                except Exception as e:
                    logger.error(f"Error sending SoundOSC data: {e}")

    def _record_interval(self) -> None:
        """Track the gap between consecutive sends and publish mean/deviation (ms)."""
        now = time.perf_counter()
        if self._last_send_time is not None:
            self._interval_samples.append(now - self._last_send_time)
        self._last_send_time = now
        if self._interval_samples:
            samples = np.fromiter(self._interval_samples, dtype=float)
            self._config.interval_mean_ms = float(samples.mean() * 1000.0)
            self._config.interval_min_ms = float(samples.min() * 1000.0)
            self._config.interval_max_ms = float(samples.max() * 1000.0)

    def _send_data(self) -> None:
        with self._input_lock:
            frames = self._poses
            seq_state = self._seq_state
        self._send_bundle(self._build_bundle(frames, seq_state))

    def _send_blackout(self) -> None:
        """Send a final all-zero bundle so the receiver ends in a clean state.

        Zeroed globals + every frame's inactive reset, in one bundle. Composes the
        same leaf builders as the live path, so _build_bundle stays blackout-agnostic.
        """
        bundle = OscBundleBuilder(IMMEDIATELY)  # type: ignore
        self._add_global_messages(bundle, None)  # None → zeroed globals
        num_players = self._config.max_players
        for id in range(num_players):
            self._add_inactive_frame_messages(bundle, id, num_players)
        self._config.active_players = 0
        self._send_bundle(bundle)

    def _build_bundle(self, frames: FrameDict, seq_state: SequencerState | None) -> OscBundleBuilder:
        bundle = OscBundleBuilder(IMMEDIATELY)  # type: ignore
        self._add_global_messages(bundle, seq_state)

        num_players = self._config.max_players
        self._config.active_players = sum(1 for id in range(num_players) if id in frames)

        for id in range(num_players):
            if id in frames:
                self._inactive_counts[id] = 0
                self._add_active_frame_messages(bundle, frames[id], frames, num_players)
            elif self._inactive_counts[id] < 2:
                # Only send a player's inactive reset for the first 2 ticks after it disappears.
                self._inactive_counts[id] += 1
                self._add_inactive_frame_messages(bundle, id, num_players)

        return bundle

    def _send_bundle(self, bundle: OscBundleBuilder) -> None:
        if bundle._contents:
            with self._client_lock:
                self._client.send(bundle.build())

    def _on_connection_change(self, _=None) -> None:
        with self._client_lock:
            self._client = SimpleUDPClient(self._config.ip_addresses, self._config.port)
        logger.info(f"Reconnected to {self._config.ip_addresses}:{self._config.port}")

    def _add_global_messages(self, bundle_builder: OscBundleBuilder, seq_state: SequencerState | None) -> None:
        """Tick-wide messages. A None seq_state yields the all-zero (idle/blackout) globals."""
        stage_msg = OscMessageBuilder(address="/global/state")
        stage_msg.add_arg(seq_state.stage if seq_state is not None else 0, arg_type=OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(stage_msg.build()) # type: ignore

        stage_progress_msg = OscMessageBuilder(address="/global/state/progress")
        stage_progress_msg.add_arg(seq_state.stage_progress if seq_state is not None else 0.0, arg_type=OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(stage_progress_msg.build()) # type: ignore

        progress_msg = OscMessageBuilder(address="/global/progress")
        progress_msg.add_arg(seq_state.progress if seq_state is not None else 0.0, arg_type=OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(progress_msg.build()) # type: ignore

    def _add_inactive_frame_messages(self, bundle_builder: OscBundleBuilder, id: int, num_players: int) -> None:
        # Send active=0 message
        msg_builder = OscMessageBuilder(address=f"/pose/{id}/active")
        msg_builder.add_arg(0, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(msg_builder.build()) # type: ignore

        # Reset bbox to 0 (centre_x, centre_y, width, height)
        bbox_msg = OscMessageBuilder(address=f"/pose/{id}/bbox")
        for _ in range(4):
            bbox_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(bbox_msg.build()) # type: ignore

        # Reset time/motion to 0
        motion_msg = OscMessageBuilder(address=f"/pose/{id}/time/motion")
        motion_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(motion_msg.build()) # type: ignore

        # Reset time/age to 0
        age_msg = OscMessageBuilder(address=f"/pose/{id}/time/age")
        age_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(age_msg.build()) # type: ignore

        # Reset angle/rad values to 0
        angle_reset_msg = OscMessageBuilder(address=f"/pose/{id}/angle/rad")
        for _ in range(17):  # 17 angle values based on AngleLandmark
            angle_reset_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(angle_reset_msg.build()) # type: ignore

        # Reset angle/vel values to 0
        vel_reset_msg = OscMessageBuilder(address=f"/pose/{id}/angle/vel")
        for _ in range(17):
            vel_reset_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(vel_reset_msg.build()) # type: ignore

        # Reset angle/sym to 0
        sym_msg = OscMessageBuilder(address=f"/pose/{id}/angle/sym")
        sym_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(sym_msg.build()) # type: ignore

        # Reset similarity values to 0
        similarity_reset_msg = OscMessageBuilder(address=f"/pose/{id}/similarity")
        for _ in range(num_players):
            similarity_reset_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(similarity_reset_msg.build()) # type: ignore

        # Reset similarity/pose to 0
        pose_sim_reset_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/pose")
        for _ in range(num_players):
            pose_sim_reset_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(pose_sim_reset_msg.build()) # type: ignore

        # Reset similarity/gate to 0
        gate_reset_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/gate")
        for _ in range(num_players):
            gate_reset_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(gate_reset_msg.build()) # type: ignore

        # Reset similarity/motion to 0
        motion_sim_reset_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/motion")
        for _ in range(num_players):
            motion_sim_reset_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(motion_sim_reset_msg.build()) # type: ignore

        # Reset similarity/leader to 0
        leader_reset_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/leader")
        for _ in range(num_players):
            leader_reset_msg.add_arg(0.0, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(leader_reset_msg.build()) # type: ignore

    def _add_active_frame_messages(self, bundle_builder: OscBundleBuilder, frame: Frame, frames: FrameDict, num_players: int) -> None:
        id: int = frame.track_id
        active_msg = OscMessageBuilder(address=f"/pose/{id}/active")
        active_msg.add_arg(1, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(active_msg.build()) # type: ignore

        # bbox: centre_x, centre_y, width, height
        bbox_values: list[float] = frame[BBox].values.tolist() if BBox in frame else [np.nan] * 4
        bbox_msg = OscMessageBuilder(address=f"/pose/{id}/bbox")
        for val in bbox_values:
            bbox_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(bbox_msg.build()) # type: ignore

        # when there is normal motion, about 1 per second
        motion_time: float = frame[MotionTime].value if MotionTime in frame else 0.0
        change_msg = OscMessageBuilder(address=f"/pose/{id}/time/motion")
        change_msg.add_arg(float(motion_time), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(change_msg.build()) # type: ignore

        # in seconds
        age: float = frame[Age].value if Age in frame else 0.0
        change_msg = OscMessageBuilder(address=f"/pose/{id}/time/age")
        change_msg.add_arg(float(age), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(change_msg.build()) # type: ignore

        # range [-pi, pi]
        angle_rad_values: list[float] = frame[Angles].values.tolist()
        angle_rad_msg = OscMessageBuilder(address=f"/pose/{id}/angle/rad")
        for angle in angle_rad_values:
            angle_rad_msg.add_arg(angle, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(angle_rad_msg.build()) # type: ignore

        # range [-finite, finite] -> [-2pi, 2pi]
        angle_vel_values: list[float] = frame[AngleVelocity].values.tolist()
        angle_vel_msg = OscMessageBuilder(address=f"/pose/{id}/angle/vel")
        for angle_vel in angle_vel_values:
            angle_vel_msg.add_arg(angle_vel, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(angle_vel_msg.build()) # type: ignore

        # range [0, 1]
        mean_sym: float = frame[AngleSymmetry].overall_symmetry() if AngleSymmetry in frame else 0.0
        mean_sym_msg = OscMessageBuilder(address=f"/pose/{id}/angle/sym")
        mean_sym_msg.add_arg(float(mean_sym), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(mean_sym_msg.build()) # type: ignore

        # range [0, 1] - raw angle similarity
        pose_sim_values: list[float] = frame[Similarity].values.tolist() if Similarity in frame else [0.0] * num_players
        pose_sim_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/pose")
        for val in pose_sim_values:
            pose_sim_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(pose_sim_msg.build()) # type: ignore

        # range [0, 1] - motion gate (both poses moving)
        gate_values: list[float] = frame[MotionGate].values.tolist() if MotionGate in frame else [0.0] * num_players
        gate_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/gate")
        for val in gate_values:
            gate_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(gate_msg.build()) # type: ignore

        # range [0, 1] - motion-gated similarity (similarity * motion_gate)
        motion_sim_values: list[float] = frame[Similarity].values.tolist() if Similarity in frame else [0.0] * num_players
        motion_sim_values = [0.0 if np.isnan(v) else v for v in motion_sim_values]
        motion_sim_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/motion")
        for val in motion_sim_values:
            motion_sim_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(motion_sim_msg.build()) # type: ignore

        # range [-1, 1] - leader scores (negative=this leads, positive=other leads)
        leader_values: list[float] = frame[LeaderScore].values.tolist() if LeaderScore in frame else [0.0] * num_players
        leader_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/leader")
        for val in leader_values:
            leader_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(leader_msg.build()) # type: ignore

        # DEPRECATED: Compute motion-gated similarity using old method (backward compatibility)

        similarity_values: list[float] = np.nan_to_num(frame[Similarity].values, nan=0.0).tolist() if Similarity in frame else [0.0] * num_players

        # range [0, 1] - DEPRECATED backward-compatible motion-gated similarity
        similarity_msg = OscMessageBuilder(address=f"/pose/{id}/similarity")
        for similarity in similarity_values:
            similarity_msg.add_arg(similarity, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(similarity_msg.build()) # type: ignore

