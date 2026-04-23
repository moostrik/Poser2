import os
import threading
import socket
import queue
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Union

from pythonosc.udp_client import UDPClient
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle import OscBundle
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from apps.white_space.composition.output import CompositionOutput
from modules.utils.HotReloadMethods import HotReloadMethods

import logging
logger = logging.getLogger(__name__)


OscMessageList = list[Union[OscMessage, OscBundle]]


@dataclass
class LedUdpSenderSettings:
    resolution: int
    port: int
    ip_addresses: list[str]

    send_info: bool = True
    use_signed: bool = False
    mtu: int = 1500
    _chunk_size: int = field(init=False, repr=False)
    _num_chunks: int = field(init=False, repr=False)

    def __post_init__(self):
        self._chunk_size, self._num_chunks = self._calculate_optimal_chunks(self.resolution, self.mtu)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    @staticmethod
    def _calculate_optimal_chunks(byte_length: int, MTU=1500, byte_size: int = 1) -> tuple[int, int]:
        max_chunk_size: int = int((MTU - 100) / byte_size)
        if byte_length <= max_chunk_size:
            return byte_length, 1
        min_chunks = (byte_length + max_chunk_size - 1) // max_chunk_size
        for divisor in range(min_chunks, byte_length):
            if byte_length % divisor == 0:
                chunk_size = byte_length // divisor
                if chunk_size <= max_chunk_size:
                    return chunk_size, divisor
        logger.info(
            f"No perfect divisor found for {byte_length} bytes, using {min_chunks} chunks of {max_chunk_size} bytes"
        )
        return max_chunk_size, min_chunks


class LedUdpSender(threading.Thread):
    """Sends LED strip data over OSC/UDP to the installation hardware."""

    def __init__(self, settings: LedUdpSenderSettings) -> None:
        super().__init__()
        self.settings: LedUdpSenderSettings = settings
        self.running = False
        self.message_queue: queue.Queue[CompositionOutput] = queue.Queue()
        self.osc_clients: dict[str, UDPClient] = {}
        self.HotReloadStaticMethods = HotReloadMethods(self.__class__, True)

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.join()

    def run(self) -> None:
        valid_ips: list[str] = [ip for ip in self.settings.ip_addresses if self._check_ip_adress_availability(ip)]
        if not valid_ips:
            logger.info("No valid IP addresses found. Exiting LedUdpSender thread.")
            return
        for ip in valid_ips:
            self.osc_clients[ip] = UDPClient(ip, self.settings.port)
        logger.info(
            f"LED UDP SENDER: port {self.settings.port}, addresses {valid_ips}, "
            f"{self.settings._num_chunks} chunks of {self.settings._chunk_size} bytes each."
        )
        info_message: OscBundle = self._build_info_message(self.settings)
        self.osc_clients[ip].send(info_message)

        self.running = True
        while self.running:
            try:
                output: CompositionOutput = self.message_queue.get(block=True, timeout=0.1)
                message_list: Optional[OscMessageList] = self._build_data_message(output, self.settings)
                if message_list:
                    for ip in self.osc_clients:
                        for message in message_list:
                            try:
                                self.osc_clients[ip].send(message)
                            except Exception as e:
                                logger.error(f"Error sending to {ip}: {e}")
            except queue.Empty:
                continue
            except Exception:
                logger.exception("LedUdpSender error")

    def send_message(self, output: CompositionOutput) -> None:
        self.message_queue.put(output)

    @staticmethod
    def _build_info_message(settings: LedUdpSenderSettings) -> OscBundle:
        bundle = OscBundleBuilder(IMMEDIATELY)
        r_msgb = OscMessageBuilder("/WS/resolution")
        r_msgb.add_arg(settings.resolution)
        bundle.add_content(r_msgb.build())  # type: ignore
        cz_msgb = OscMessageBuilder("/WS/chunk_size")
        cz_msgb.add_arg(settings.chunk_size)
        bundle.add_content(cz_msgb.build())  # type: ignore
        cn_msgb = OscMessageBuilder("/WS/num_chunks")
        cn_msgb.add_arg(settings.num_chunks)
        bundle.add_content(cn_msgb.build())  # type: ignore
        return bundle.build()

    @staticmethod
    def _build_data_message(output: CompositionOutput, settings: LedUdpSenderSettings) -> Optional[OscMessageList]:
        try:
            if output.resolution != settings.resolution:
                raise ValueError(f"Resolution mismatch: expected {settings.resolution}, got {output.resolution}")

            message_list: OscMessageList = []
            if settings.use_signed:
                white_channel: np.ndarray = LedUdpSender.float_to_int8(output.light_0)
                blue_channel: np.ndarray  = LedUdpSender.float_to_int8(output.light_1)
            else:
                white_channel = LedUdpSender.float_to_uint8(output.light_0)
                blue_channel  = LedUdpSender.float_to_uint8(output.light_1)

            for i in range(settings.num_chunks):
                start_idx = i * settings.chunk_size
                end_idx   = min((i + 1) * settings.chunk_size, len(white_channel))

                wc_msgb = OscMessageBuilder(f"/WS/white{i}")
                wc_msgb.add_arg(white_channel[start_idx:end_idx].tobytes(), 'b')
                message_list.append(wc_msgb.build())

                bc_msgb = OscMessageBuilder(f"/WS/blue{i}")
                bc_msgb.add_arg(blue_channel[start_idx:end_idx].tobytes(), 'b')
                message_list.append(bc_msgb.build())

            return message_list
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None

    @staticmethod
    def _check_ip_adress_availability(ip_address: str) -> bool:
        if ip_address == "127.0.0.1":
            return True
        try:
            socket.inet_aton(ip_address)
        except socket.error:
            logger.warning(f"Invalid IP address format: {ip_address}")
            return False
        try:
            response = os.system(f"ping -n 1 -w 500 {ip_address} > nul 2>&1")
            if response == 0:
                return True
            logger.warning(f"IP address {ip_address} is not reachable")
            return False
        except Exception as e:
            logger.error(f"IP address {ip_address} check failed: {e}")
            return False

    @staticmethod
    def float_to_uint8(arr: np.ndarray) -> np.ndarray:
        arr_clipped: np.ndarray = np.clip(arr, 0.0, 1.0)
        return np.round(arr_clipped * 255.0).astype(np.uint8)

    @staticmethod
    def float_to_int8(arr: np.ndarray) -> np.ndarray:
        arr_clipped = np.clip(arr, 0.0, 1.0)
        return np.round(arr_clipped * 255.0 - 128.0).astype(np.int8)
