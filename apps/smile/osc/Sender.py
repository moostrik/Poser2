from ipaddress import ip_address
from pythonosc.udp_client import SimpleUDPClient
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder
from threading import Thread, Lock
from queue import Queue, Empty
from time import time, sleep
from typing import List, Tuple, Union, Any, Optional
import logging

from apps.smile.osc.settings import OscSettings

logger = logging.getLogger(__name__)

ArgValue = Union[str, bytes, bool, int, float, list]

def is_valid_ip_address(ip: str) -> bool:
    try:
        ip_address(ip)
        return True
    except ValueError:
        return False

def is_valid_port(port: int) -> bool:
    return port in range(1, 65536)

# addresses whose every message counts (a bang per clip start, a test bang per screen); of the others only
# the newest value is sent
EVENT_ADDRESSES: tuple[str, ...] = ('/bang', '/test_bang')
ERROR_LOG_INTERVAL: float = 10.0    # s between the log lines of a send that keeps failing

class Sender(Thread):
    def __init__(self, settings: OscSettings) -> None:
        super().__init__()
        self.settings: OscSettings = settings

        self.client: SimpleUDPClient | None = None

        self.running: bool = False
        self.queue: Queue = Queue()
        self.timeout: float = 0.1
        self.mutex: Lock = Lock()
        self.last_message_time: float = time()
        self._error_time: float = 0.0       # when a failed send was last logged

    def start(self) -> None:
        if self.running:
            return
        self.updateClient()
        self.settings.bind(OscSettings.ip, self._on_address)
        self.settings.bind(OscSettings.port, self._on_address)
        self.running = True
        super().start()

    def stop(self) -> None:
        if self.running:
            self.settings.unbind(OscSettings.ip, self._on_address)
            self.settings.unbind(OscSettings.port, self._on_address)
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
                with self.mutex:
                    try:
                        if self.client:
                            bundle = self.process_data(relevant_data)
                            self.client.send(bundle)
                            self.last_message_time = time()  # Update last message time
                    except Exception:
                        # this bundle is lost, the next one is sent: a sender that stopped here would
                        # stay silent for the rest of the run while its queue grows
                        if time() - self._error_time > ERROR_LOG_INTERVAL:
                            self._error_time = time()
                            logger.exception('osc error, bundle not sent')

            # Calculate time elapsed since the start of the loop
            elapsed_time: float = time() - start_time

            # Calculate the time to sleep to achieve the desired framerate
            update_interval: float = 1.0 / max(self.settings.fps, 1.0)
            sleep_time: float = max(0, update_interval - elapsed_time)

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
            if address in EVENT_ADDRESSES:
                # events: each one is sent
                processed_data.extend((address, args) for args in args_list)
            else:
                # states: the latest value
                processed_data.append((address, args_list[-1]))

        return processed_data

    def process_data(self, data_list):
        bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

        for address, args in data_list:
            msg = osc_message_builder.OscMessageBuilder(address=address)
            # a tuple or list goes as separate arguments (as one add_arg it would be an OSC array, which Max does not take)
            for arg in (args if isinstance(args, (tuple, list)) else (args,)):
                msg.add_arg(arg)
            bundle.add_content(msg.build()) # type: ignore

        return bundle.build()

    def send(self, address: str, args: ArgValue) -> None:
        # only while the thread takes them: nothing empties the queue otherwise
        if self.settings.active and self.running:
            self.queue.put((address, args))

    def _on_address(self, _=None) -> None:
        self.updateClient()

    def updateClient(self) -> None:
        ip: str = self.settings.ip
        port: int = self.settings.port
        if not is_valid_ip_address(ip) or not is_valid_port(port):
            logger.warning('Invalid OSC address %s:%s, keeping the current client', ip, port)
            return
        with self.mutex:
            self.client = SimpleUDPClient(ip, port)
        logger.info('Osc Client Sending to %s %s', ip, port)
