import threading
import queue
import numpy as np
from pythonosc import udp_client, osc_message
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc import osc_bundle
from pythonosc.parsing import osc_types
from modules.av.Definitions import AvOutput, UDP_PORT, UDP_IP_ADDRESSES
from modules.utils.HotReloadStaticMethods import HotReloadStaticMethods

class UdpSender(threading.Thread):
    def __init__(self, port: int = UDP_PORT, ip_addresses: list[str] = UDP_IP_ADDRESSES) -> None:
        super().__init__()
        self.port: int = port
        self.ip_addresses: list[str] = ip_addresses
        self.running = False
        self.message_queue: queue.Queue[AvOutput] = queue.Queue()

        # Create an OSC client for each IP address
        self.osc_clients = {}
        for ip in ip_addresses:
            self.osc_clients[ip] = udp_client.SimpleUDPClient(ip, port)

        self.HotReloadStaticMethods = HotReloadStaticMethods(self.__class__, True)

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
                av_output: AvOutput = self.message_queue.get(block=True, timeout=0.1)
                UdpSender._send_to_all(av_output, self.osc_clients)
            except queue.Empty:
                continue

    @staticmethod
    def _send_to_all(av_output: AvOutput, osc_clients: dict) -> None:
        """Send the AvOutput as OSC messages to all IP addresses using blob data.

        Args:
            av_output: The output data to send
            osc_clients: Dictionary of OSC clients {ip: client}
        """
        try:
            # Convert float16 to uint8 (since values are normalized 0-1)
            # Scale 0-1 range to 0-255 range
            # white_channel = (av_output.img[0, :, 0] * 511).astype(np.uint8)
            # blue_channel = (av_output.img[0, :, 1] * 511).astype(np.uint8)
            white_channel = ((av_output.img[0, :, 0] - 0.5) * 2 * 127).astype(np.int8)
            blue_channel = ((av_output.img[0, :, 1] - 0.5) * 2 * 127).astype(np.int8)

            # With uint8 (1 byte each), we can send much larger chunks with blob type
            chunk_size = 900  # Larger chunks are possible with blob data

            # Send data to all IP addresses
            for ip, client in osc_clients.items():
                try:
                    # Send resolution and angle
                    client.send_message("/WS/resolution", av_output.resolution)
                    # client.send_message("/WS/chunk_size", chunk_size)

                    # Send white channel in chunks as blobs
                    num_chunks_white = len(white_channel) // chunk_size + (1 if len(white_channel) % chunk_size else 0)
                    client.send_message("/WS/white/num_chunks", num_chunks_white)

                    for i in range(num_chunks_white):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(white_channel))

                        # Get the chunk as bytes
                        chunk_bytes = white_channel[start_idx:end_idx].tobytes()

                        # Create a message with blob data
                        msg_builder = OscMessageBuilder(f"/WS/white/chunk/{i}")
                        msg_builder.add_arg(chunk_bytes, "b")
                        msg: osc_message.OscMessage = msg_builder.build()

                        # Send the message
                        client.send(msg)

                    # Send blue channel in chunks as blobs
                    num_chunks_blue = len(blue_channel) // chunk_size + (1 if len(blue_channel) % chunk_size else 0)
                    client.send_message("/WS/blue/num_chunks", num_chunks_blue)

                    for i in range(num_chunks_blue):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(blue_channel))

                        # Get the chunk as bytes
                        chunk_bytes = blue_channel[start_idx:end_idx].tobytes()

                        # Create a message with blob data
                        msg_builder = OscMessageBuilder(f"/WS/blue/chunk/{i}")
                        msg_builder.add_arg(chunk_bytes, "b")
                        msg = msg_builder.build()

                        # Send the message
                        client.send(msg)

                except Exception as e:
                    print(f"Error sending data to {ip}: {e}")

        except Exception as e:
            print(f"Error preparing data: {e}")

    def stop(self) -> None:
        """Stop the thread."""
        self.running = False
        self.join()