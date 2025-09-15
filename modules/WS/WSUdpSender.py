# Standard library imports
import os
import threading
import socket
import queue
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Union

# Third-party imports
from pythonosc.udp_client import UDPClient
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle import OscBundle
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

# Local application imports
from modules.WS.WSOutput import WSOutput, WS_IMG_TYPE
from modules.utils.HotReloadMethods import HotReloadMethods



OscMessageList = list[Union[OscMessage, OscBundle]]  # Type alias for OSC messages or bundles

@dataclass
class WSUdpSenderSettings:
    resolution: int
    port: int
    ip_addresses: list[str]

    send_info: bool = True  # Whether to send resolution, chunk size, and num chunks info
    use_signed: bool = False  # Whether to send data as signed (-128 to 127) instead of unsigned bytes (0-255)

    mtu: int = 1500  # Maximum Transmission Unit for UDP packets
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
    def _calculate_optimal_chunks(byte_length: int, MTU=1500, byte_size: int=1) -> tuple[int, int]:
        """
        Calculate the optimal chunk size to evenly divide an array of given length.

        Args:
            byte_length: Total length of the array to chunk
            MTU: Maximum Transmission Unit (default 1500)

        Returns:
            The optimal chunk size
        """
        # Maximum size for a single chunk (accounting for OSC overhead)
        max_chunk_size: int = int((MTU - 100) / byte_size)  # Leave space for OSC overhead

        # If the array fits in one chunk, return the array length
        if byte_length <= max_chunk_size:
            return byte_length, 1

        # Calculate minimum number of chunks needed
        min_chunks = (byte_length + max_chunk_size - 1) // max_chunk_size

        # Try to find a divisor of byte_length that results in equal chunks
        for divisor in range(min_chunks, byte_length):
            if byte_length % divisor == 0:  # Perfect division with no remainder
                chunk_size = byte_length // divisor
                if chunk_size <= max_chunk_size:
                    # print(f"Using perfect divisor: {divisor} chunks of size {chunk_size} bytes for {byte_length} bytes")
                    return chunk_size, divisor

        print(f"No perfect divisor found for {byte_length} bytes, using maximum {min_chunks} chunks with a size of {max_chunk_size} bytes, totalling {min_chunks * max_chunk_size * byte_size} bytes")
        return max_chunk_size, min_chunks




class WSUdpSender(threading.Thread):
    def __init__(self, settings: WSUdpSenderSettings) -> None:
        super().__init__()
        self.settings: WSUdpSenderSettings = settings

        self.running = False
        self.message_queue: queue.Queue[WSOutput] = queue.Queue()

        # Create an OSC client for each IP address
        self.osc_clients: dict[str, UDPClient] = {}

        self.HotReloadStaticMethods = HotReloadMethods(self.__class__, True)

    def stop(self) -> None:
        if not self.running:
            return
        """Stop the thread."""
        self.running = False
        self.join()

    def run(self) -> None:
        """Thread loop to send messages."""

        # Validate IP addresses and create OSC clients
        valid_ips: list[str] = [ip for ip in self.settings.ip_addresses if self._check_ip_adress_availability(ip)]

        if not valid_ips:
            print("No valid IP addresses found. Exiting UdpSender thread.")
            return

        for ip in valid_ips:
            self.osc_clients[ip] = UDPClient(ip, self.settings.port)
        print(f"UDP SENDER: port {self.settings.port} and addresses {valid_ips}. Data is split in {self.settings._num_chunks} chunks of {self.settings._chunk_size} bytes each.")

        info_message: OscBundle = self._build_info_message(self.settings)
        self.osc_clients[ip].send(info_message)

        self.running = True
        while self.running:
            try:
                # Block until a message is available or timeout occurs
                av_output: WSOutput = self.message_queue.get(block=True, timeout=0.1)
                message_list: Optional[OscMessageList] = self._build_data_message(av_output, self.settings)
                if message_list:
                    for ip in self.osc_clients:
                        for message in message_list:
                            try:
                                self.osc_clients[ip].send(message)
                            except Exception as e:
                                print(f"Error sending message to {ip}: {e}")
            except queue.Empty:
                continue

    def send_message(self, av_output: WSOutput):
        """Queue a message to be sent."""
        self.message_queue.put(av_output)

    @staticmethod
    def _build_info_message(settings: WSUdpSenderSettings) -> OscBundle:
        """Build an OSC message containing resolution, chunk size, and number of chunks."""
        bundle = OscBundleBuilder(IMMEDIATELY)

        r_msgb = OscMessageBuilder("/WS/resolution")
        r_msgb.add_arg(settings.resolution)  # Use "i" for integer resolution
        bundle.add_content(r_msgb.build()) # type: ignore

        cz_msgb = OscMessageBuilder("/WS/chunk_size")
        cz_msgb.add_arg(settings.chunk_size)
        bundle.add_content(cz_msgb.build())  # type: ignore

        cz_msgb = OscMessageBuilder("/WS/num_chunks")
        cz_msgb.add_arg(settings.num_chunks)
        bundle.add_content(cz_msgb.build()) # type: ignore

        return bundle.build()

    @staticmethod
    def _build_data_message(av_output: WSOutput, settings: WSUdpSenderSettings) -> Optional[OscMessageList]:
        """Send the WSOutput as OSC messages to all IP addresses using blob data."""
        try:
            if av_output.resolution != settings.resolution:
                raise Exception(f"Resolution mismatch: expected {settings.resolution}, got {av_output.resolution}")

            message_list: OscMessageList = []

            if settings.use_signed:
                white_channel: np.ndarray = WSUdpSender.float_to_int8(av_output.light_0)
                blue_channel: np.ndarray = WSUdpSender.float_to_int8(av_output.light_1)
            else:
                white_channel: np.ndarray = WSUdpSender.float_to_uint8(av_output.light_0)
                blue_channel: np.ndarray = WSUdpSender.float_to_uint8(av_output.light_1)

            for i in range(settings.num_chunks):
                start_idx: int = i * settings.chunk_size
                end_idx: int = min((i + 1) * settings.chunk_size, len(white_channel))

                white_chunk_bytes: bytes = white_channel[start_idx:end_idx].tobytes()
                wc_msgb = OscMessageBuilder(f"/WS/white{i}")
                wc_msgb.add_arg(white_chunk_bytes, 'b')
                message_list.append(wc_msgb.build())

                blue_chunk_bytes: bytes = blue_channel[start_idx:end_idx].tobytes()
                bc_msgb = OscMessageBuilder(f"/WS/blue{i}")
                bc_msgb.add_arg(blue_chunk_bytes, 'b')
                message_list.append(bc_msgb.build())

            return message_list

        except Exception as e:
            print(f"Error preparing data: {e}")
            return None

    @staticmethod
    def float_to_int8(arr: np.ndarray) -> np.ndarray:
        """
        Convert a float32 array to int8 by clipping and scaling.

        Args:
            arr_float32: Input array of type float32

        Returns:
            Array of type int8 with values scaled to -128 to 127
        """
        if arr.dtype != np.float32:
            raise ValueError("Input array must be of type float32")

        # Convert to float16 for memory efficiency, then back to float32 for calculations
        arr_clipped = np.clip(arr, 0.0, 1.0)
        arr_scaled = arr_clipped * 255.0 - 128.0    # Scale to -128 to 127 range
        # print( arr_scaled.min(), arr_scaled.max())
        return np.round(arr_scaled).astype(np.int8)
    @staticmethod

    def float_to_uint8(arr: np.ndarray) -> np.ndarray:
        """
        Convert a float32 array to int8 by clipping and scaling.

        Args:
            arr_float32: Input array of type float32

        Returns:
            Array of type int8 with values scaled to -128 to 127
        """
        if arr.dtype != np.float32:
            raise ValueError("Input array must be of type float32")

        # Convert to float16 for memory efficiency, then back to float32 for calculations
        arr_clipped: np.ndarray = np.clip(arr, 0.0, 1.0)
        arr_scaled = arr_clipped * 255.0    # Scale to 0 to 255 range
        return np.round(arr_scaled).astype(np.uint8)

    @staticmethod
    def uint8_to_int8(arr: np.ndarray) -> np.ndarray:
        """
        Convert a uint8 array to int8 by shifting values.

        Args:
            arr_uint8: Input array of type uint8

        Returns:
            Array of type int8 with values shifted to -128 to 127
        """
        if arr.dtype != np.uint8:
            raise ValueError("Input array must be of type uint8")

        # Convert uint8 to int8 by subtracting 128
        return (arr.astype(np.int8) - 128)

    @staticmethod
    def _calculate_optimal_chunks(byte_length: int, byte_size: int=1, MTU=1500) -> tuple[int, int]:
        """
        Calculate the optimal chunk size to evenly divide an array of given length.

        Args:
            byte_length: Total length of the array to chunk
            MTU: Maximum Transmission Unit (default 1500)

        Returns:
            The optimal chunk size
        """
        # Maximum size for a single chunk (accounting for OSC overhead)
        max_chunk_size: int = int((MTU - 100) / byte_size)  # Leave space for OSC overhead

        # If the array fits in one chunk, return the array length
        if byte_length <= max_chunk_size:
            return byte_length, 1

        # Calculate minimum number of chunks needed
        min_chunks = (byte_length + max_chunk_size - 1) // max_chunk_size

        # Try to find a divisor of byte_length that results in equal chunks
        for divisor in range(min_chunks, byte_length):
            if byte_length % divisor == 0:  # Perfect division with no remainder
                chunk_size = byte_length // divisor
                if chunk_size <= max_chunk_size:
                    # print(f"Using perfect divisor: {divisor} chunks of size {chunk_size} bytes for {byte_length} bytes")
                    return chunk_size, divisor

        print(f"No perfect divisor found for {byte_length} bytes, using maximum {min_chunks} chunks with a size of {max_chunk_size} bytes, totalling {min_chunks * max_chunk_size * byte_size} bytes")
        return max_chunk_size, min_chunks

    @staticmethod
    def _check_ip_adress_availability(ip_address: str) -> bool:
        """
        Check if the given IP address is valid and potentially reachable.

        Args:
            ip_address: IP address to check

        Returns:
            True if IP address is valid, False otherwise
        """
        # Special case for localhost/loopback addresses
        if ip_address == "127.0.0.1":
            return True

        try:
            socket.inet_aton(ip_address)
        except socket.error:
            print(f"WARNING: Invalid IP address format: {ip_address}")
            return False

        # For non-localhost addresses, try a simple ping test
        # This is optional and may not work on all systems
        try:
            # Use a simple ping with short timeout
            response = os.system(f"ping -n 1 -w 500 {ip_address} > nul 2>&1")
            if response == 0:
                return True
            else:
                print(f"WARNING: IP address {ip_address} is not reachable")
                return False
        except Exception as e:
            print(f"WARNING: IP address {ip_address} check failed: {e}")
            return False