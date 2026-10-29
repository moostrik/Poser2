"""Art-Net sender: a thread that sends one DMX packet to one universe at the settings' rate. Its owner
fills the packet in `fill`, called from the thread before every send."""

from threading import Thread, Lock
from typing import Callable
import logging
import time

from stupidArtnet import StupidArtnet

from modules.settings import BaseSettings, Field, Widget
from .net_probe import is_valid_ip

logger = logging.getLogger(__name__)


class ArtNetSenderSettings(BaseSettings):
    enabled:  Field[bool]  = Field(True, widget=Widget.switch, description="Send Art-Net to the lights")
    ip:       Field[str]   = Field("127.0.0.1", widget=Widget.ip_field, description="Art-Net node IP address")
    universe: Field[int]   = Field(0, min=0, max=254, widget=Widget.number_field, description="Art-Net universe")
    fps:      Field[float] = Field(40.0, min=1.0, max=60.0, step=1.0, description="Packets sent per second")


def is_valid_universe(value: int) -> bool:
    return value in range(0, 255)


class ArtNetSender(Thread):
    def __init__(self, settings: ArtNetSenderSettings, channels: int, fill: Callable[[bytearray], None]) -> None:
        """`channels` DMX channels; `fill` writes the packet before each send, on the sender's thread."""
        super().__init__()
        self.settings: ArtNetSenderSettings = settings
        self.fill: Callable[[bytearray], None] = fill
        self.mutex: Lock = Lock()
        self.running: bool = False

        # even: StupidArtnet rounds its own size up to even and refuses a packet of another length
        self.packet_size: int = channels + channels % 2
        self.packet = bytearray(self.packet_size)
        self.sender: StupidArtnet | None = None

    def start(self) -> None:
        self._update_sender()
        self.settings.bind(ArtNetSenderSettings.ip, self._on_address)
        self.settings.bind(ArtNetSenderSettings.universe, self._on_address)
        self.settings.bind(ArtNetSenderSettings.enabled, self._on_enabled)
        self.running = True
        super().start()

    def stop(self) -> None:
        if not self.is_alive():
            return
        self.settings.unbind(ArtNetSenderSettings.ip, self._on_address)
        self.settings.unbind(ArtNetSenderSettings.universe, self._on_address)
        self.settings.unbind(ArtNetSenderSettings.enabled, self._on_enabled)
        self.running = False

        if self.sender:
            self.sender.flash_all()
            time.sleep(self._interval())
            self.sender.blackout()

        self.join()

    def run(self) -> None:
        while self.running:
            self.update()
            time.sleep(self._interval())

    def _interval(self) -> float:
        return 1.0 / max(self.settings.fps, 1.0)

    def update(self) -> None:
        self.fill(self.packet)
        if self.sender and self.settings.enabled:
            with self.mutex:
                self.sender.send(self.packet)

    def _on_address(self, _=None) -> None:
        self._update_sender()

    def _on_enabled(self, _=None) -> None:
        self.flash()

    def _update_sender(self) -> None:
        ip: str = self.settings.ip
        universe: int = self.settings.universe
        if not is_valid_ip(ip) or not is_valid_universe(universe):
            logger.warning('Invalid Art-Net address %s universe %s, keeping the current sender', ip, universe)
            return

        with self.mutex:
            if self.sender:
                self.sender.stop()
            self.sender = StupidArtnet(target_ip=ip, universe=universe, packet_size=self.packet_size, fps=int(self.settings.fps))
            logger.info('Artnet sending to %s on universe %s', ip, universe)

        self.flash()

    def flash(self) -> None:
        if not self.settings.enabled or not self.sender:
            return
        with self.mutex:
            self.sender.flash_all()
            time.sleep(self._interval())
