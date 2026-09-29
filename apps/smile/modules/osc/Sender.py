from ipaddress import ip_address
from pythonosc.udp_client import SimpleUDPClient
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder
from threading import Thread, Lock
from queue import Queue, Empty
from time import time, sleep
from typing import List, Tuple, Union, Any, Optional

ArgValue = Union[str, bytes, bool, int, float, list]

def is_valid_ip_address(ip: str) -> bool:
    try:
        ip_address(ip)
        return True
    except ValueError:
        return False

def is_valid_port(port: int | str) -> bool:
    if type(port) == str:
        try:
            port = int(port)
        except ValueError:
            return False
    port = int(port)
    if port in range(1, 65535):
        return True
    return False

class Sender(Thread):
    def __init__(self, ip: str = "127.0.0.1", port: int | str = 8000, fps: float = 30.0) -> None:
        super().__init__()

        self.ip: str= "127.0.0.1"
        if is_valid_ip_address(ip):
            self.ip = ip
        else:
            print('ThreadedOscSender', ip, 'is not a valid ip address, defaulting to', self.ip)

        self.port: int = int(port)
        if is_valid_port(port):
            self.port = int(port)
        else:
            print('ThreadedOscSender', port, 'is not a valid port, defaulting to', self.port)

        self.client: SimpleUDPClient | None = None

        self.running: bool = False
        self.queue: Queue = Queue()
        self.timeout: float = 0.1
        self.mutex: Lock = Lock()
        self.fps: float = fps
        self.update_interval: float = 1.0 / float(self.fps)
        self.last_message_time: float = time()
        self.sending: bool = True

    def start(self) -> None:
        if self.running:
            return
        self.updateClient()
        self.running = True
        super().start()

    def stop(self) -> None:
        if self.running:
            self.running = False
            self.join()

    def run(self) -> None:
        while self.running:
            start_time = time()
            data_list = []
            try:
                # Get all available data in the queue
                while True:
                    data: Any = self.queue.get(False)
                    data_list.append(data)
            except Empty:
                pass  # Queue is empty

            if data_list:
                # Process the data and construct a message
                relevant_data = self.get_last_data(data_list)

                # Send the constructed message
                self.mutex.acquire()
                try:
                    if self.client:
                        bundle = self.process_data(relevant_data)
                        self.client.send(bundle)
                        self.last_message_time = time()  # Update last message time
                except:
                    self.mutex.release()
                    print('osc error, stopping')
                    self.stop()

                finally:
                    self.mutex.release()

            # Calculate time elapsed since the start of the loop
            elapsed_time: float = time() - start_time

            # Calculate the time to sleep to achieve the desired framerate
            sleep_time: float = max(0, self.update_interval - elapsed_time)

            # Sleep to control the framerate
            sleep(sleep_time)

    def get_last_data(self, data_list):

        # Separate data based on unique addresses
        grouped_data = {}
        for address, args in data_list:
            if address not in grouped_data:
                grouped_data[address] = []
            grouped_data[address].append(args)

        # Process each group and construct a message
        processed_data = []
        for address, args_list in grouped_data.items():
            # Your logic to handle multiple entries with the same address
            # Example: Use the latest values for each address
            latest_args = args_list[-1]
            processed_data.append((address, latest_args))

        return processed_data

    def process_data(self, data_list):
        bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

        for address, args in data_list:
            msg = osc_message_builder.OscMessageBuilder(address=address)
            msg.add_arg(args)
            bundle.add_content(msg.build()) # type: ignore

        return bundle.build()

    def send(self, address: str, args: ArgValue) -> None:
        if self.sending:
            self.queue.put((address, args))

    def set_sending(self, value: bool) -> None:
        self.sending = value

    def set_fps(self, fps: int | float) -> None:
        self.fps = fps
        self.update_interval: float = 1.0 / float(self.fps)

    def set_port(self, port: int | str) -> None:
        if not is_valid_port(port):
            return
        p = int(port)
        if self.port == p:
            return
        self.port = p
        self.updateClient()

    def set_ip(self, ip: str) -> None:
        if ip == self.ip:
            return
        if is_valid_ip_address(ip):
            self.ip = ip
        else: return
        self.updateClient()

    def updateClient(self) -> None:
        self.mutex.acquire()
        self.client = SimpleUDPClient(self.ip, self.port)
        print('Osc Client Sending to', self.ip, self.port)
        self.mutex.release()
