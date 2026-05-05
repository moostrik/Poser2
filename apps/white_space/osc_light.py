from threading import Thread, Event, Lock
from typing import Optional, Union

import numpy as np
from pythonosc.udp_client import UDPClient
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle import OscBundle
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY

from .composition import CompositionOutput
from modules.settings import BaseSettings, Field, Group, Widget
from modules.inout.network_validation import validate_connection

import logging
logger = logging.getLogger(__name__)


OscMessageList = list[Union[OscMessage, OscBundle]]


class OscLightOffsetSettings(BaseSettings):
    white_0: Field[int] = Field(0, min=-10, max=10, description="White strip 0 offset")
    white_1: Field[int] = Field(0, min=-10, max=10, description="White strip 1 offset")
    blue_0:  Field[int] = Field(0, min=-10, max=10, description="Blue strip 0 offset")
    blue_1:  Field[int] = Field(0, min=-10, max=10, description="Blue strip 1 offset")


class OscLightSettings(BaseSettings):
    ip_addresses: Field[str]  = Field("127.0.0.1", widget=Widget.ip_field,      description="Target LED receiver IP address")
    port:         Field[int]  = Field(8000, min=1024, max=65535, widget=Widget.number_field, description="Target UDP port")
    use_signed:   Field[bool] = Field(False,                                     description="Send signed int8 instead of uint8")
    resolution:   Field[int]  = Field(3600, min=256, max=4096, step=16, access=Field.INIT, visible=False, description="LED strip resolution (pixels)")
    mtu:          Field[int]  = Field(1500, min=576, max=9000,  access=Field.INIT, description="Network MTU (affects chunk size)")
    chunk_size:    Field[int]  = Field(0,    access=Field.READ,  description="Computed chunk size (bytes)")
    num_chunks:   Field[int]  = Field(0,    access=Field.READ,  description="Computed number of chunks")
    rpm:          Field[int]  = Field(0, min=0,   max=2400, description="LED rotation speed (RPM)")
    offsets:      Group[OscLightOffsetSettings] = Group(OscLightOffsetSettings)


class OscLight:
    """Sends LED strip data over OSC/UDP to the installation hardware.

    OSC address pattern (all under /WS/):
        /WS/i/res         -- info: total resolution (pixels)
        /WS/i/chunk_sz    -- info: chunk size (bytes)
        /WS/i/n_chunks    -- info: number of chunks
        /WS/o/0..3        -- offsets: white_0, white_1, blue_0, blue_1
        /WS/r/0           -- rotation: RPM
        /WS/w/{i}         -- white channel chunk i (bytes)
        /WS/b/{i}         -- blue channel chunk i (bytes)
    """

    def __init__(self, settings: OscLightSettings) -> None:
        self._config = settings
        self._chunk_size, self._num_chunks = self._calculate_optimal_chunks(settings.resolution, settings.mtu)
        self._config.chunk_size = self._chunk_size
        self._config.num_chunks = self._num_chunks

        self._latest_output: Optional[CompositionOutput] = None
        self._output_lock:   Lock  = Lock()
        self._client_lock:   Lock  = Lock()
        self._update_event:  Event = Event()
        self._client: UDPClient = UDPClient(settings.ip_addresses, settings.port)

        self._running = False
        self._thread: Optional[Thread] = None

        self._config.bind(OscLightSettings.ip_addresses, self._on_connection_change)  # type: ignore[arg-type]
        self._config.bind(OscLightSettings.port,         self._on_connection_change)  # type: ignore[arg-type]

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="OscLight")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._update_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def send_message(self, output: CompositionOutput) -> None:
        with self._output_lock:
            self._latest_output = output
        self._update_event.set()

    def _run(self) -> None:
        if not validate_connection(self._config.ip_addresses, self._config.port, "OscLight"):
            self._running = False
            return

        logger.info(
            f"OscLight: {self._config.ip_addresses}:{self._config.port}, "
            f"resolution={self._config.resolution}, "
            f"{self._num_chunks} chunks of {self._chunk_size} bytes each."
        )

        info_message = self._build_info_message(self._config.resolution, self._chunk_size, self._num_chunks)
        with self._client_lock:
            self._client.send(info_message)

        while self._running:
            self._update_event.wait()
            self._update_event.clear()
            if not self._running:
                break
            with self._output_lock:
                output = self._latest_output
            if output is None:
                continue
            message_list = self._build_data_message(output, self._config, self._chunk_size, self._num_chunks)
            if message_list:
                with self._client_lock:
                    for message in message_list:
                        try:
                            self._client.send(message)
                        except Exception as e:
                            logger.error(f"LedUdpSender send error: {e}")

    def _on_connection_change(self, _=None) -> None:
        with self._client_lock:
            self._client = UDPClient(self._config.ip_addresses, self._config.port)
        logger.info(f"OscLight: reconnected to {self._config.ip_addresses}:{self._config.port}")

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_info_message(resolution: int, chunk_size: int, num_chunks: int) -> OscBundle:
        bundle = OscBundleBuilder(IMMEDIATELY)
        r_msgb = OscMessageBuilder("/WS/i/res")
        r_msgb.add_arg(resolution)
        bundle.add_content(r_msgb.build())  # type: ignore
        cz_msgb = OscMessageBuilder("/WS/i/chunk_sz")
        cz_msgb.add_arg(chunk_size)
        bundle.add_content(cz_msgb.build())  # type: ignore
        cn_msgb = OscMessageBuilder("/WS/i/n_chunks")
        cn_msgb.add_arg(num_chunks)
        bundle.add_content(cn_msgb.build())  # type: ignore
        return bundle.build()

    @staticmethod
    def _build_data_message(
        output: CompositionOutput,
        settings: OscLightSettings,
        chunk_size: int,
        num_chunks: int,
    ) -> Optional[OscMessageList]:
        try:
            message_list: OscMessageList = []

            for addr, val in (
                ("/WS/o/0", settings.offsets.white_0),
                ("/WS/o/1", settings.offsets.white_1),
                ("/WS/o/2", settings.offsets.blue_0),
                ("/WS/o/3", settings.offsets.blue_1),
            ):
                off_msgb = OscMessageBuilder(addr)
                off_msgb.add_arg(val)
                message_list.append(off_msgb.build())

            rpm_msgb = OscMessageBuilder("/WS/r/0")
            rpm_msgb.add_arg(settings.rpm)
            message_list.append(rpm_msgb.build())

            if settings.use_signed:
                white_channel: np.ndarray = OscLight.float_to_int8(output.light_0)
                blue_channel:  np.ndarray = OscLight.float_to_int8(output.light_1)
            else:
                white_channel = OscLight.float_to_uint8(output.light_0)
                blue_channel  = OscLight.float_to_uint8(output.light_1)

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx   = min((i + 1) * chunk_size, len(white_channel))

                wc_msgb = OscMessageBuilder(f"/WS/white{i}") # -> w/{i}
                wc_msgb.add_arg(white_channel[start_idx:end_idx].tobytes(), 'b')
                message_list.append(wc_msgb.build())

                bc_msgb = OscMessageBuilder(f"/WS/blue{i}") # -> b/{i}
                bc_msgb.add_arg(blue_channel[start_idx:end_idx].tobytes(), 'b')
                message_list.append(bc_msgb.build())

            return message_list
        except Exception as e:
            logger.error(f"OscLight error preparing data: {e}")
            return None

    # ------------------------------------------------------------------
    # Chunk calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_optimal_chunks(byte_length: int, mtu: int = 1500) -> tuple[int, int]:
        max_chunk_size = (mtu - 100)
        if byte_length <= max_chunk_size:
            return byte_length, 1
        min_chunks = (byte_length + max_chunk_size - 1) // max_chunk_size
        for divisor in range(min_chunks, byte_length):
            if byte_length % divisor == 0:
                chunk_size = byte_length // divisor
                if chunk_size <= max_chunk_size:
                    return chunk_size, divisor
        logger.info(
            f"No perfect divisor found for {byte_length} bytes, "
            f"using {min_chunks} chunks of {max_chunk_size} bytes"
        )
        return max_chunk_size, min_chunks

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def float_to_uint8(arr: np.ndarray) -> np.ndarray:
        return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def float_to_int8(arr: np.ndarray) -> np.ndarray:
        return np.round(np.clip(arr, 0.0, 1.0) * 255.0 - 128.0).astype(np.int8)

