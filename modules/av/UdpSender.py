import socket
import threading
import json
import queue
import numpy as np
from modules.av.Definitions import AvOutput, UDP_PORT, UDP_IP_ADDRESSES

class UdpSender(threading.Thread):
    def __init__(self, port: int = UDP_PORT, ip_addresses: list[str] = UDP_IP_ADDRESSES) -> None:
        super().__init__()
        self.port: int = port
        self.ip_addresses: list[str] = ip_addresses
        self.running = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.message_queue: queue.Queue[AvOutput] = queue.Queue()

    def send_message(self, av_output: AvOutput):
        """Queue a message to be sent."""
        self.message_queue.put(av_output)

    def run(self) -> None:
        print(f"UdpSender initialized with port {self.port} and IP addresses {self.ip_addresses}")
        """Thread loop to send messages."""
        self.running = True
        while self.running:
            try:
                # Block until a message is available or timeout occurs
                av_output: AvOutput = self.message_queue.get(timeout=0.1)
                self._send_to_all(av_output)
            except queue.Empty:
                continue

    def _send_to_all(self, av_output: AvOutput) -> None:
        """Send the serialized AvOutput to all IP addresses."""
        try:
            # Extract the first and second channels of the image
            white_channel: np.ndarray = av_output.img[0, :, 0].tolist()  # First channel
            blue_channel: np.ndarray = av_output.img[0, :, 1].tolist()   # Second channel

            # Serialize AvOutput to JSON
            serialized_data: str = json.dumps({
                "resolution": av_output.resolution,
                "angle": av_output.angle,
                "white": white_channel,  # First channel as 'white'
                "blue": blue_channel     # Second channel as 'blue'
            })
            # print(f"Serialized data: {serialized_data[:100]}...")  # Print first 100 characters for debugging

            # Send the serialized data to all IP addresses
            for ip in self.ip_addresses:
                self.sock.sendto(serialized_data.encode('utf-8'), (ip, self.port))
        except Exception as e:
            print(f"Error sending data: {e}")

    def stop(self) -> None:
        """Stop the thread."""
        self.running = False
        self.join()
        self.sock.close()