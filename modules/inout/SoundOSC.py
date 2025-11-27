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
from modules.pose.features.AngleSymmetry import SymmetryElement
from modules.pose.similarity import SimilarityBatch, SimilarityFeature, AggregationMethod

from modules.DataHub import DataHub, DataType

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class SoundOSCConfig:
    ip_addresses: str = field(default_factory=lambda: "127.0.0.1")
    port: int = field(default=9000)
    num_players: int = field(default=8)
    data_type: DataType = field(default=DataType.pose_I)


class SoundOSC:
    """
    Sends smooth pose data over OSC at a configurable frame rate in its own thread.
    """
    def __init__(self, data_hub: DataHub, settings: SoundOSCConfig) -> None:

        self._config: SoundOSCConfig = settings
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

                self._send_data()

    def _send_data(self) -> None:
        t = perf_counter()

        bundle_builder: OscBundleBuilder = OscBundleBuilder(IMMEDIATELY)  # type: ignore

        poses: dict[int, Frame] = self._data_hub.get_dict(self._config.data_type)
        num_players = self._config.num_players

        for id in range(num_players):
            if id not in poses:
                SoundOSC._build_inactive_message(id, bundle_builder)
            else:
                SoundOSC._build_active_message(poses[id], bundle_builder)

        similarity: SimilarityBatch | None = self._data_hub.get_item(DataType.sim_P)
        if similarity is not None:
            SoundOSC._build_similarity_message(similarity, bundle_builder, num_players)

        bundle: OscBundle = bundle_builder.build()

        # Send the bundle if it contains any messages
        if bundle_builder._contents:
            self._client.send(bundle)

        # print(f"SoundOSC: Sent OSC with {len(bundle_builder._contents)} messages in {perf_counter() - t:.4f} seconds")

    @ staticmethod
    def _build_inactive_message(id: int, bundle_builder: OscBundleBuilder) -> None:
        msg_builder = OscMessageBuilder(address=f"/pose/{id}/active")
        msg_builder.add_arg(0)
        bundle_builder.add_content(msg_builder.build()) # type: ignore


    @ staticmethod
    def _build_active_message(pose: Frame, bundle_builder: OscBundleBuilder) -> None:
        id: int = pose.track_id
        active_msg = OscMessageBuilder(address=f"/pose/{id}/active")
        active_msg.add_arg(1)
        bundle_builder.add_content(active_msg.build()) # type: ignore

        motion: float = pose.motion_time
        change_msg = OscMessageBuilder(address=f"/pose/{id}/time/motion")
        change_msg.add_arg(float(motion))
        bundle_builder.add_content(change_msg.build()) # type: ignore

        motion: float = pose.age
        change_msg = OscMessageBuilder(address=f"/pose/{id}/time/age")
        change_msg.add_arg(float(motion))
        bundle_builder.add_content(change_msg.build()) # type: ignore

        angle_msg = OscMessageBuilder(address=f"/pose/{id}/angle")
        for joint in AngleLandmark:
            angle: float | None = pose.angles.get(joint)
            angle_msg.add_arg(float(angle))
        bundle_builder.add_content(angle_msg.build()) # type: ignore

        velocity_msg = OscMessageBuilder(address=f"/pose/{id}/delta")
        for joint in AngleLandmark:
            velocity: float | None = pose.angle_vel.get(joint)
            velocity_msg.add_arg(float(velocity))
        bundle_builder.add_content(velocity_msg.build()) # type: ignore

        mean_symmetry: float = pose.angle_sym.geometric_mean()
        symmetry_msg = OscMessageBuilder(address=f"/pose/{id}/symmetry/mean")
        symmetry_msg.add_arg(float(mean_symmetry))
        bundle_builder.add_content(symmetry_msg.build()) # type: ignore

        sym_msg = OscMessageBuilder(address=f"/pose/{id}/symmetry")
        for sym_type in SymmetryElement:
            symmetry: float = pose.angle_sym[sym_type]
            sym_msg.add_arg(float(symmetry))
        bundle_builder.add_content(sym_msg.build()) # type: ignore

    @ staticmethod
    def _build_similarity_message(similarity_batch: SimilarityBatch, bundle_builder: OscBundleBuilder, num_players: int) -> None:

        for id in range(num_players):
            for other_id in range(num_players):
                if id == other_id:
                    continue
                feature: SimilarityFeature | None= similarity_batch.get_pair((id, other_id))
                similarity = feature.aggregate_similarity(AggregationMethod.HARMONIC_MEAN, exponent=2.0) if feature is not None else 0.0
                sync_msg = OscMessageBuilder(address=f"/similarity/motion/{id}/{other_id}")
                sync_msg.add_arg(float(similarity))
                bundle_builder.add_content(sync_msg.build()) # type: ignore