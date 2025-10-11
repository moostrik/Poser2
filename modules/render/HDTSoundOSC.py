import threading
import time
import traceback
from typing import Dict, List, Optional, Set
import numpy as np
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from modules.pose.Pose import Pose
from modules.pose.PoseTypes import PoseJoint
from modules.pose.smooth.PoseSmoothData import PoseSmoothData
from modules.pose.PoseAngles import POSE_ANGLE_JOINTS

from modules.utils.HotReloadMethods import HotReloadMethods

class HDTSoundOSC:
    """
    Sends smooth pose data over OSC at a configurable frame rate in its own thread.
    """
    def __init__(self, smooth_data: PoseSmoothData, ip: str = "127.0.0.1", port: int = 9000, frame_rate: float = 60.0) -> None:
        self.smooth_data: PoseSmoothData = smooth_data
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.frame_interval: float = 1.0 / frame_rate
        print(f"HDTSoundOSC: Initialized OSC client to {ip}:{port} at {frame_rate} FPS")

        self.running = False
        self._thread: Optional[threading.Thread] = None

        hot_reload = HotReloadMethods(self.__class__, True, True)


    def start(self) -> None:
        """Start the OSC sender thread"""
        if self._thread is not None and self._thread.is_alive():
            return

        self.running = True
        self._thread = threading.Thread(target=self._send_loop)
        self._thread.daemon = True  # Thread will exit when main program exits
        self._thread.start()

    def stop(self) -> None:
        """Stop the OSC sender thread"""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)  # Wait up to 1 second for thread to exit
            self._thread = None

    def _send_loop(self) -> None:
        """Main loop for sending OSC data at regular intervals"""
        while self.running:
            start_time: float = time.time()

            # Make a copy of the active tracklet IDs to avoid locks during iteration
            self._send_data()

            # Sleep for the remainder of the frame interval
            elapsed: float = time.time() - start_time
            sleep_time: float = max(0, self.frame_interval - elapsed)
            time.sleep(sleep_time)

    def _send_data(self) -> None:
        bundle_builder: OscBundleBuilder = OscBundleBuilder(IMMEDIATELY)  # type: ignore

        for id in range(self.smooth_data.num_players):
            if not self.smooth_data.get_is_active(id):
                self._build_inactive_message(id, bundle_builder)
            else:
                self._build_active_message(id, bundle_builder)

        # Send the bundle if it contains any messages
        if bundle_builder._contents:
            self.client.send(bundle_builder.build())

    def _build_inactive_message(self, id: int, bundle_builder: OscBundleBuilder) -> None:
        active_msg = OscMessageBuilder(address=f"/pose/{id}/active")
        active_msg.add_arg(0)
        bundle_builder.add_content(active_msg.build()) # type: ignore

    def _build_active_message(self, id: int, bundle_builder: OscBundleBuilder) -> None:
        active_msg = OscMessageBuilder(address=f"/pose/{id}/active")
        active_msg.add_arg(1)
        bundle_builder.add_content(active_msg.build()) # type: ignore

        motion: float = self.smooth_data.get_angular_motion(id)
        if motion is not None:
            change_msg = OscMessageBuilder(address=f"/pose/{id}/motion")
            change_msg.add_arg(float(motion))
            bundle_builder.add_content(change_msg.build()) # type: ignore

        # Smoothed angles for key joints
        for joint in POSE_ANGLE_JOINTS:
            angle: float | None = self.smooth_data.get_angle(id, joint)
            if angle is not None:
                angle_msg = OscMessageBuilder(address=f"/pose/{id}/angle/{joint.name}")
                angle_msg.add_arg(float(angle))
                bundle_builder.add_content(angle_msg.build()) # type: ignore

