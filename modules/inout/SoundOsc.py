from dataclasses import dataclass, field
from threading import Thread, Event
from time import perf_counter

import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_bundle import OscBundle
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from modules.pose.Frame import Frame
from modules.pose.features.Angles import AngleLandmark
from modules.pose.features.AngleSymmetry import SymmetryElement, AggregationMethod
from modules.pose.similarity import SimilarityBatch, SimilarityFeature, AggregationMethod

from modules.DataHub import DataHub, DataHubType

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class SoundOscConfig:
    ip_addresses: str = field(default_factory=lambda: "127.0.0.1")
    port: int = field(default=9000)
    num_players: int = field(default=8)
    data_type: DataHubType = field(default=DataHubType.pose_I)


class SoundOsc:
    """
    Sends smooth pose data over OSC at a configurable frame rate in its own thread.
    """
    def __init__(self, data_hub: DataHub, settings: SoundOscConfig) -> None:

        self._config: SoundOscConfig = settings
        self._data_hub: DataHub = data_hub
        self._client: SimpleUDPClient = SimpleUDPClient(self._config.ip_addresses, self._config.port)

        print(f"SoundOSC: Initialized OSC client to {self._config.ip_addresses}:{self._config.port}")

        self._running = False
        self._update_event: Event = Event()
        self._thread: Thread | None = None

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def running(self) -> bool:
        return self._running

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

    def notify_update(self) -> None:
        """Notify the sender thread that new data is available"""
        self._update_event.set()

    def _run(self) -> None:
        """Main loop for sending OSC data at regular intervals"""
        while self._running:
            self._update_event.wait()
            self._update_event.clear()
            if self._running:

                try:
                    self._send_data()
                except Exception as e:
                    print(f"Error sending OSC data: {e}")

    def _send_data(self) -> None:
        t = perf_counter()

        bundle_builder: OscBundleBuilder = OscBundleBuilder(IMMEDIATELY)  # type: ignore

        poses: dict[int, Frame] = self._data_hub.get_dict(self._config.data_type)
        num_players = self._config.num_players

        motions: dict[int, float] = {}
        for id in range(num_players):
            if id not in poses:
                motions[id] = 0
            else:
                motion = poses[id].angle_motion.aggregate(AggregationMethod.MAX)
                motion = min(1.0, motion * 1.5)
                motions[id] = motion

        motion_similarities: dict[int, list[float]] = {}

        for id in range(num_players):
            if id in poses:
                similarity_values: list[float] = poses[id].similarity.values.tolist()
                other_ids = [i for i in range(3) if i != id]

                for o_id in other_ids:
                    similarity_values[o_id] *= min(motions[id], motions[o_id])

                motion_similarities[id] = similarity_values

        for id in range(num_players):
            if id not in poses:
                SoundOsc._build_inactive_message(id, bundle_builder)
            else:
                SoundOsc._build_active_message(poses[id], bundle_builder, motion_similarities[id])

        # similarity: SimilarityBatch | None = self._data_hub.get_item(DataHubType.sim_P)
        # if similarity is not None:
        #     SoundOsc._build_similarity_message(similarity, bundle_builder, num_players)

        bundle: OscBundle = bundle_builder.build()

        # Send the bundle if it contains any messages
        if bundle_builder._contents:
            self._client.send(bundle)

        # print(f"SoundOSC: Sent OSC with {len(bundle_builder._contents)} messages in {perf_counter() - t:.4f} seconds")

    @ staticmethod
    def _build_inactive_message(id: int, bundle_builder: OscBundleBuilder) -> None:
        msg_builder = OscMessageBuilder(address=f"/pose/{id}/active")
        msg_builder.add_arg(0, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(msg_builder.build()) # type: ignore


    @ staticmethod
    def _build_active_message(pose: Frame, bundle_builder: OscBundleBuilder, m_s: list[float] ) -> None:
        id: int = pose.track_id
        active_msg = OscMessageBuilder(address=f"/pose/{id}/active")
        active_msg.add_arg(1, OscMessageBuilder.ARG_TYPE_INT)
        bundle_builder.add_content(active_msg.build()) # type: ignore

        # when there is normal motion, about 1 per second
        motion_time: float = pose.motion_time
        change_msg = OscMessageBuilder(address=f"/pose/{id}/time/motion")
        change_msg.add_arg(float(motion_time), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(change_msg.build()) # type: ignore

        # in seconds
        age: float = pose.age
        change_msg = OscMessageBuilder(address=f"/pose/{id}/time/age")
        change_msg.add_arg(float(age), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(change_msg.build()) # type: ignore

        # range [-pi, pi]
        angle_rad_values: list[float] = pose.angles.values.tolist()
        angle_rad_msg = OscMessageBuilder(address=f"/pose/{id}/angle/rad")
        for angle in angle_rad_values:
            angle_rad_msg.add_arg(angle, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(angle_rad_msg.build()) # type: ignore

        # range [-finite, finite] -> [-2pi, 2pi]
        angle_vel_values: list[float] = pose.angle_vel.values.tolist()
        angle_vel_msg = OscMessageBuilder(address=f"/pose/{id}/angle/vel")
        for angle_vel in angle_vel_values:
            angle_vel_msg.add_arg(angle_vel, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(angle_vel_msg.build()) # type: ignore

        # range [0, 1]
        mean_sym: float = pose.angle_sym.overall_symmetry()
        mean_sym_msg = OscMessageBuilder(address=f"/pose/{id}/angle/sym")
        mean_sym_msg.add_arg(float(mean_sym), OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(mean_sym_msg.build()) # type: ignore

        # range [0, 1]
        similarity_values: list[float] = m_s #pose.similarity.values.tolist()
        similarity_msg = OscMessageBuilder(address=f"/pose/{id}/similarity")
        for similarity in similarity_values:
            similarity_msg.add_arg(similarity, OscMessageBuilder.ARG_TYPE_FLOAT)
        bundle_builder.add_content(similarity_msg.build()) # type: ignore
