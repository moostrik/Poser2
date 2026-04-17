from threading import Thread, Event, Lock
from time import perf_counter
import socket

import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_bundle import OscBundle
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from modules.pose.frame import Frame, FrameDict
from modules.pose.features import Angles, AngleVelocity, AngleSymmetry, Similarity, MotionGate, LeaderScore, MotionTime, Age
from modules.pose.features.Angles import AngleLandmark

from modules.session.sequencer import SequencerState
from modules.settings import BaseSettings, Field, Widget
from modules.utils.HotReloadMethods import HotReloadMethods
from modules.inout.network_validation import validate_connection

import logging
logger = logging.getLogger(__name__)

class OscSoundSettings(BaseSettings):
    max_players: Field[int]  = Field(4,    min=1,    max=16,    access=Field.INIT,         description="Max number of player poses to send (IDs 0 to N-1)")
    ip_addresses: Field[str] = Field("127.0.0.1",               widget=Widget.ip_field,     description="Target OSC IP address")
    port: Field[int]         = Field(9000, min=1024, max=65535, widget=Widget.number_field, description="Target OSC port")


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

        self._running = False
        self._update_event: Event = Event()
        self._thread: Thread | None = None
        # Initialize inactive message counts for all players
        self._inactive_message_counts: dict[int, int] = {id: 0 for id in range(self._config.max_players)}

        # Pre-build inactive messages for all players
        self._inactive_messages: dict[int, list[OscBundle]] = {}
        for id in range(self._config.max_players):
            bundle_builder = OscBundleBuilder(IMMEDIATELY)  # type: ignore
            OscSound._build_inactive_message(id, bundle_builder, self._config.max_players)
            self._inactive_messages[id] = bundle_builder._contents

        # hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def running(self) -> bool:
        return self._running

    def set_poses(self, poses: FrameDict) -> None:
        """Push new pose data and trigger a send."""
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

                try:
                    self._send_data()
                except socket.error as e:
                    logger.error(f"SoundOSC Socket error with exception: {e}")
                    self._running = False
                    break
                except Exception as e:
                    logger.error(f"Error sending SoundOSC data: {e}")

    def _send_data(self) -> None:
        t = perf_counter()

        bundle_builder: OscBundleBuilder = OscBundleBuilder(IMMEDIATELY)  # type: ignore

        # Snapshot current data
        with self._input_lock:
            poses = self._poses
            seq_state = self._seq_state

        stage_msg = OscMessageBuilder(address="/global/state")
        stage_msg.add_arg(seq_state.stage if seq_state is not None else 0, arg_type=OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(stage_msg.build()) # type: ignore

        progress_msg = OscMessageBuilder(address="/global/time")
        progress_msg.add_arg(seq_state.progress if seq_state is not None else 0.0, arg_type=OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(progress_msg.build()) # type: ignore

        num_players = self._config.max_players

        for id in range(num_players):
            if id not in poses:
                # Only send inactive messages twice
                if self._inactive_message_counts[id] < 2:
                    self._inactive_message_counts[id] += 1
                    # Add pre-built inactive messages to bundle
                    for msg in self._inactive_messages[id]:
                        bundle_builder.add_content(msg)
            else:
                # Reset inactive count when pose becomes active
                self._inactive_message_counts[id] = 0
                OscSound._build_active_message(poses[id], bundle_builder, poses, num_players)

        bundle: OscBundle = bundle_builder.build()

        # Send the bundle if it contains any messages
        if bundle_builder._contents:
            with self._client_lock:
                self._client.send(bundle)

    def _on_connection_change(self, _=None) -> None:
        with self._client_lock:
            self._client = SimpleUDPClient(self._config.ip_addresses, self._config.port)
        logger.info(f"SoundOSC: Reconnected to {self._config.ip_addresses}:{self._config.port}")

    @staticmethod
    def _build_inactive_message(id: int, bundle_builder: OscBundleBuilder, num_players: int) -> None:
        # Send active=0 message
        msg_builder = OscMessageBuilder(address=f"/pose/{id}/active")
        msg_builder.add_arg(0, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(msg_builder.build()) # type: ignore

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

    @staticmethod
    def _build_active_message(pose: Frame, bundle_builder: OscBundleBuilder, poses: dict[int, Frame], num_players: int) -> None:
        id: int = pose.track_id
        active_msg = OscMessageBuilder(address=f"/pose/{id}/active")
        active_msg.add_arg(1, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(active_msg.build()) # type: ignore

        # when there is normal motion, about 1 per second
        motion_time: float = pose[MotionTime].value if MotionTime in pose else 0.0
        change_msg = OscMessageBuilder(address=f"/pose/{id}/time/motion")
        change_msg.add_arg(float(motion_time), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(change_msg.build()) # type: ignore

        # in seconds
        age: float = pose[Age].value if Age in pose else 0.0
        change_msg = OscMessageBuilder(address=f"/pose/{id}/time/age")
        change_msg.add_arg(float(age), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(change_msg.build()) # type: ignore

        # range [-pi, pi]
        angle_rad_values: list[float] = pose[Angles].values.tolist()
        angle_rad_msg = OscMessageBuilder(address=f"/pose/{id}/angle/rad")
        for angle in angle_rad_values:
            angle_rad_msg.add_arg(angle, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(angle_rad_msg.build()) # type: ignore

        # range [-finite, finite] -> [-2pi, 2pi]
        angle_vel_values: list[float] = pose[AngleVelocity].values.tolist()
        angle_vel_msg = OscMessageBuilder(address=f"/pose/{id}/angle/vel")
        for angle_vel in angle_vel_values:
            angle_vel_msg.add_arg(angle_vel, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(angle_vel_msg.build()) # type: ignore

        # range [0, 1]
        mean_sym: float = pose[AngleSymmetry].overall_symmetry() if AngleSymmetry in pose else 0.0
        mean_sym_msg = OscMessageBuilder(address=f"/pose/{id}/angle/sym")
        mean_sym_msg.add_arg(float(mean_sym), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(mean_sym_msg.build()) # type: ignore

        # range [0, 1] - raw angle similarity
        pose_sim_values: list[float] = pose[Similarity].values.tolist() if Similarity in pose else [0.0] * num_players
        pose_sim_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/pose")
        for val in pose_sim_values:
            pose_sim_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(pose_sim_msg.build()) # type: ignore

        # range [0, 1] - motion gate (both poses moving)
        gate_values: list[float] = pose[MotionGate].values.tolist() if MotionGate in pose else [0.0] * num_players
        gate_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/gate")
        for val in gate_values:
            gate_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(gate_msg.build()) # type: ignore

        # range [0, 1] - motion-gated similarity (similarity * motion_gate)
        motion_sim_values: list[float] = pose[Similarity].values.tolist() if Similarity in pose else [0.0] * num_players
        motion_sim_values = [0.0 if np.isnan(v) else v for v in motion_sim_values]
        motion_sim_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/motion")
        for val in motion_sim_values:
            motion_sim_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(motion_sim_msg.build()) # type: ignore

        # range [-1, 1] - leader scores (negative=this leads, positive=other leads)
        leader_values: list[float] = pose[LeaderScore].values.tolist() if LeaderScore in pose else [0.0] * num_players
        leader_msg = OscMessageBuilder(address=f"/pose/{id}/similarity/leader")
        for val in leader_values:
            leader_msg.add_arg(val, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(leader_msg.build()) # type: ignore

        # DEPRECATED: Compute motion-gated similarity using old method (backward compatibility)

        similarity_values: list[float] = np.nan_to_num(pose[Similarity].values, nan=0.0).tolist() if Similarity in pose else [0.0] * num_players

        # range [0, 1] - DEPRECATED backward-compatible motion-gated similarity
        similarity_msg = OscMessageBuilder(address=f"/pose/{id}/similarity")
        for similarity in similarity_values:
            similarity_msg.add_arg(similarity, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(similarity_msg.build()) # type: ignore
