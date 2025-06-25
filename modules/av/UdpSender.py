# Standard library imports
import os
import threading
import socket
import queue
import numpy as np
from typing import Optional

# Third-party imports
from pythonosc.udp_client import UDPClient
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle import OscBundle
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

# Local application imports
from modules.av.Definitions import AvOutput
from modules.utils.HotReloadStaticMethods import HotReloadStaticMethods

# Constants
UDP_MTU = 1500
UDP_PORT = 8888
UDP_IP_ADDRESSES: list[str] = ['127.0.0.1']  # Localhost

class UdpSender(threading.Thread):
    def __init__(self, resolution: int, port: int = UDP_PORT, ip_addresses: list[str] = UDP_IP_ADDRESSES, mtu: int = UDP_MTU) -> None:
        super().__init__()
        self.port: int = port
        self.ip_addresses: list[str] = ip_addresses

        self.resolution: int = resolution
        self.chunk_size: int
        self.num_chunks: int
        self.chunk_size, self.num_chunks = self._calculate_optimal_chunks(resolution, mtu)

        self.running = False
        self.message_queue: queue.Queue[AvOutput] = queue.Queue()

        # Create an OSC client for each IP address
        self.osc_clients: dict[str, UDPClient] = {}

        self.HotReloadStaticMethods = HotReloadStaticMethods(self.__class__, True)

    def stop(self) -> None:
        """Stop the thread."""
        self.running = False
        self.join()

    def run(self) -> None:
        """Thread loop to send messages."""

        # Validate IP addresses and create OSC clients
        valid_ips: list[str] = [ip for ip in self.ip_addresses if self._check_ip_adress_availability(ip)]

        if not valid_ips:
            print("No valid IP addresses found. Exiting UdpSender thread.")
            return

        for ip in valid_ips:
            self.osc_clients[ip] = UDPClient(ip, self.port)
        print(f"Starting UdpSender on port {self.port} and addresses {valid_ips}. Data is split in {self.num_chunks} chunks of {self.chunk_size} bytes each.")

        self.running = True
        while self.running:
            try:
                # Block until a message is available or timeout occurs
                av_output: AvOutput = self.message_queue.get(block=True, timeout=0.1)
                message_bundle: Optional[OscBundle] = self._build_message(av_output, self.resolution, self.chunk_size, self.num_chunks)
                if message_bundle:
                    for ip in self.osc_clients:
                        try:
                            self.osc_clients[ip].send(message_bundle)
                        except Exception as e:
                            print(f"Error sending message to {ip}: {e}")
            except queue.Empty:
                continue

    def send_message(self, av_output: AvOutput):
        """Queue a message to be sent."""
        self.message_queue.put(av_output)

    @staticmethod
    def _build_message(av_output: AvOutput, resolution: int, chunk_size: int, num_chunks: int) -> Optional[OscBundle]:
        """Send the AvOutput as OSC messages to all IP addresses using blob data."""
        try:
            if av_output.resolution != resolution:
                raise Exception(f"Resolution mismatch: expected {resolution}, got {av_output.resolution}")

            bundle = OscBundleBuilder(IMMEDIATELY)

            r_msgb = OscMessageBuilder("/WS/resolution")
            r_msgb.add_arg(av_output.resolution)  # Use "i" for integer resolution
            bundle.add_content(r_msgb.build()) # type: ignore

            cz_msgb = OscMessageBuilder("/WS/chunk_size")
            cz_msgb.add_arg(chunk_size)
            bundle.add_content(cz_msgb.build())  # type: ignore

            cz_msgb = OscMessageBuilder("/WS/num_chunks")
            cz_msgb.add_arg(num_chunks)
            bundle.add_content(cz_msgb.build()) # type: ignore

            # Convert float16 to int8 (since values are normalized 0-1)
            # white_channel: np.ndarray = ((av_output.img[0, :, 0] - 0.5) * 2 * 127).astype(np.int8)
            # blue_channel: np.ndarray =  ((av_output.img[0, :, 1] - 0.5) * 2 * 127).astype(np.int8)
            white_channel: np.ndarray = ((av_output.img[0, :, 0] - 0.5) * 2 * 127).astype(np.int8)
            blue_channel: np.ndarray =  ((av_output.img[0, :, 1] - 0.5) * 2 * 127).astype(np.int8)
            for i in range(num_chunks):
                start_idx: int = i * chunk_size
                end_idx: int = min((i + 1) * chunk_size, len(white_channel))

                white_chunk_bytes: bytes = white_channel[start_idx:end_idx].tobytes()
                wc_msgb = OscMessageBuilder(f"/WS/white/chunk/{i}")
                wc_msgb.add_arg(white_chunk_bytes)
                bundle.add_content(wc_msgb.build()) # type: ignore

                blue_chunk_bytes: bytes = blue_channel[start_idx:end_idx].tobytes()
                bc_msgb = OscMessageBuilder(f"/WS/blue/chunk/{i}")
                bc_msgb.add_arg(blue_chunk_bytes)
                bundle.add_content(bc_msgb.build()) # type: ignore

            return bundle.build()

        except Exception as e:
            print(f"Error preparing data: {e}")
            return None

    @staticmethod
    def _calculate_optimal_chunks(byte_length: int, MTU=1500) -> tuple[int, int]:
        """
        Calculate the optimal chunk size to evenly divide an array of given length.

        Args:
            byte_length: Total length of the array to chunk
            MTU: Maximum Transmission Unit (default 1500)

        Returns:
            The optimal chunk size
        """
        # Maximum size for a single chunk (accounting for OSC overhead)
        max_chunk_size: int = MTU - 100  # Leave space for OSC overhead

        # If the array fits in one chunk, return the array length
        if byte_length <= max_chunk_size:
            return 1, byte_length

        # Calculate minimum number of chunks needed
        min_chunks = (byte_length + max_chunk_size - 1) // max_chunk_size

        # Try to find a divisor of byte_length that results in equal chunks
        for divisor in range(min_chunks, byte_length):
            if byte_length % divisor == 0:  # Perfect division with no remainder
                chunk_size = byte_length // divisor
                if chunk_size <= max_chunk_size:
                    # print(f"Using perfect divisor: {divisor} chunks of size {chunk_size} bytes for {byte_length} bytes")
                    return chunk_size, divisor

        print(f"No perfect divisor found for {byte_length} bytes, using maximum {min_chunks} chunks with a size of {max_chunk_size} bytes, totalling {min_chunks * max_chunk_size} bytes")
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