from threading import Thread, Event, Lock
from time import monotonic, perf_counter, sleep
from typing import Optional

import numpy as np
from pythonosc.udp_client import UDPClient
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

from ..light import Frame, Tick
from modules.settings import BaseSettings, Field, Group, Widget
from modules.inout.net_probe import validate_connection

import logging
logger = logging.getLogger(__name__)


OscMessageList = list[OscMessage]

# The fixture firmware hard-codes a 1200-byte body and exactly three chunks per channel, and finds
# the chunk digit at a fixed byte offset in the address. Changing `resolution` or `mtu` such that
# `_calculate_optimal_chunks` yields anything else silently breaks the installation.
FIRMWARE_CHUNK_SIZE: int = 1200
FIRMWARE_NUM_CHUNKS: int = 3

# Offsets and rpm are constant for a whole show, so they are sent on change only. This keepalive
# re-sends them anyway at a low rate so a lost config packet self-heals; the firmware ignores a
# repeat of the value it already holds.
_CONFIG_KEEPALIVE_S: float = 1.0


class OscLightOffsetSettings(BaseSettings):
    white_0: Field[int] = Field(0, min=-10, max=10, description="White strip 0 offset")
    white_1: Field[int] = Field(0, min=-10, max=10, description="White strip 1 offset")
    blue_0:  Field[int] = Field(0, min=-10, max=10, description="Blue strip 0 offset")
    blue_1:  Field[int] = Field(0, min=-10, max=10, description="Blue strip 1 offset")


class OscLightSenderSettings(BaseSettings):
    ip_addresses: Field[str]  = Field("127.0.0.1", widget=Widget.ip_field,      description="Target LED receiver IP address")
    port:         Field[int]  = Field(8000, min=1024, max=65535, widget=Widget.number_field, description="Target UDP port")
    use_signed:   Field[bool] = Field(False,                                     description="Send signed int8 instead of uint8")
    resolution:   Field[int]  = Field(3600, min=256, max=4096, step=16, access=Field.INIT, visible=False, description="LED strip resolution (pixels)")
    mtu:          Field[int]  = Field(1500, min=576, max=9000,  access=Field.INIT, description="Network MTU (affects chunk size)")
    chunk_size:    Field[int]  = Field(0,    access=Field.READ,  description="Computed chunk size (bytes)")
    num_chunks:   Field[int]  = Field(0,    access=Field.READ,  description="Computed number of chunks")
    lower_edge:   Field[float] = Field(0.35, min=0.0, max=1.0, step=0.01, description="Lamp turn-on floor: lit pixels lift to at least this; black stays off")
    curve:        Field[float] = Field(1.0,  min=0.5, max=3.0, step=0.01, description="Output gamma curve; <1 brightens mids, >1 darkens")
    startup_delay: Field[float] = Field(2.0, min=0.0, max=10.0, step=0.5, description="Hold motor rpm at 0 for this long after connect, then release to the commanded speed — forces a 0→target edge the motor controller acts on at boot")
    chunk_interval: Field[float] = Field(0.0,    min=0.0, max=0.005, step=0.0005, description="Seconds between consecutive pixel datagrams (0 = send back-to-back). Only raise this if the fixture reports dropped chunks — it adds output latency")
    offsets:      Group[OscLightOffsetSettings] = Group(OscLightOffsetSettings)


class OscLightSender:
    """Sends LED strip data over OSC/UDP to the installation hardware.

    OSC address pattern (all under /WS/):
        /WS/o/0..3        -- offsets: white_0, white_1, blue_0, blue_1 (int)
        /WS/r/0           -- rotation: RPM (int)
        /WS/white{i}      -- white channel chunk i (blob)
        /WS/blue{i}       -- blue channel chunk i (blob)

    Wire contract with the fixture firmware (`apps/white_space/data/firmware/firmware.ino`):

    * A chunk datagram is exactly 1220 bytes — a 20-byte OSC preamble (12-byte padded address,
      4-byte typetag, 4-byte blob length) followed by 1200 pixel bytes. The firmware reads exactly
      those 20 + 1200 bytes, and locates the chunk digit at a fixed offset in the address, so both
      the chunk size and the address spelling are load-bearing.
    * The firmware commits a frame when `/WS/blue2` arrives, so that message must be sent **last**.
    * Its socket receive buffer is 8 KB against a 7.4 KB frame, and it drains while it fills, so
      the burst normally fits. Config messages are kept out of it anyway (they never change), and
      `chunk_interval` can spread the six chunks further if the fixture ever reports dropped
      chunks — a chunk it misses is published as a stale third of the ring for one revolution.
    * On a clean quit, ``stop()`` ends with a **blackout** — rpm 0, an all-zero frame, rpm 0
      again — so the fixture goes dark and the motor decelerates immediately. The firmware's
      Ethernet watchdog (packet silence → motor stop + blank) covers the crash path only.
    """

    def __init__(self, settings: OscLightSenderSettings) -> None:
        self._config = settings
        self._chunk_size, self._num_chunks = self._calculate_optimal_chunks(settings.resolution, settings.mtu)
        self._config.chunk_size = self._chunk_size
        self._config.num_chunks = self._num_chunks

        if (self._chunk_size, self._num_chunks) != (FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS):
            logger.error(
                "chunking is %s x %s bytes but the fixture firmware only accepts %s x %s — "
                "the fixture will not display correctly. Check `resolution` (%s) and `mtu` (%s).",
                self._num_chunks, self._chunk_size, FIRMWARE_NUM_CHUNKS, FIRMWARE_CHUNK_SIZE,
                settings.resolution, settings.mtu,
            )

        self._latest_output: Optional[Frame] = None
        self._start_time:    Optional[float] = None   # connect time → motor-rpm startup hold (boot edge)
        self._output_lock:   Lock  = Lock()
        self._client_lock:   Lock  = Lock()
        self._update_event:  Event = Event()
        self._client: UDPClient = UDPClient(settings.ip_addresses, settings.port)

        # Config (offsets + rpm) is sent on change instead of every frame; `None` rpm forces the
        # first send. `_config_dirty` is set from the settings callback on whichever thread edits.
        self._config_dirty:   bool = True
        self._sent_rpm:       Optional[int] = None
        self._last_config_at: float = 0.0

        self._running = False
        self._thread: Optional[Thread] = None

        self._config.bind(OscLightSenderSettings.ip_addresses, self._on_connection_change)  # type: ignore[arg-type]
        self._config.bind(OscLightSenderSettings.port,         self._on_connection_change)  # type: ignore[arg-type]
        self._config.offsets.bind_all(self._on_offsets_change)

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="OscLightSender")
        self._thread.start()

    def stop(self) -> None:
        started = self._thread is not None
        self._running = False
        self._update_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        self._config.unbind(OscLightSenderSettings.ip_addresses, self._on_connection_change)  # type: ignore[arg-type]
        self._config.unbind(OscLightSenderSettings.port,         self._on_connection_change)  # type: ignore[arg-type]
        self._config.offsets.unbind_all(self._on_offsets_change)

        # Send AFTER the join: guaranteed to be the last thing on the wire (late
        # send_message() calls only store into the dead thread's slot).
        if started:
            try:
                self._send_blackout()
            except Exception as e:
                logger.error(f"Error sending light blackout: {e}")

    def send_message(self, output: Frame) -> None:
        with self._output_lock:
            self._latest_output = output
        self._update_event.set()

    def _run(self) -> None:
        if not validate_connection(self._config.ip_addresses, self._config.port, "OscLightSender"):
            self._running = False
            return

        logger.info(
            f"{self._config.ip_addresses}:{self._config.port}, "
            f"resolution={self._config.resolution}, "
            f"{self._num_chunks} chunks of {self._chunk_size} bytes each."
        )

        self._start_time = monotonic()   # begin the startup rpm-zero hold (forces a boot edge)

        while self._running:
            self._update_event.wait()
            self._update_event.clear()
            if not self._running:
                break
            # Take the frame *and* clear the slot under one lock: a plain read leaves the event set
            # when a new frame lands between the clear and the read, which resends the same frame.
            with self._output_lock:
                output, self._latest_output = self._latest_output, None
            if output is None:
                continue

            # Hold rpm at 0 for `startup_delay` after connect, then release to the commanded speed:
            # the 0→target edge is what the motor controller acts on. Sending it on change gives
            # exactly one clean edge; the keepalive below covers a lost packet.
            held = self._start_time is not None and (monotonic() - self._start_time) < self._config.startup_delay
            motor_rpm = 0 if held else int(output.motor.target_rpm)
            self._send_config(motor_rpm)

            chunks = self._build_chunk_messages(output, self._config, self._chunk_size, self._num_chunks)
            if chunks:
                self._send_paced(chunks, self._config.chunk_interval)

    def _send_config(self, motor_rpm: int) -> None:
        """Send the offsets and rpm — on change, or when the keepalive interval has elapsed.

        Keeping these five datagrams out of the per-frame burst is what buys the fixture's socket
        buffer enough headroom for the six pixel chunks.
        """
        now = monotonic()
        due = self._config_dirty or motor_rpm != self._sent_rpm or (now - self._last_config_at) >= _CONFIG_KEEPALIVE_S
        if not due:
            return
        self._config_dirty   = False
        self._sent_rpm       = motor_rpm
        self._last_config_at = now
        for message in self._build_config_messages(self._config, motor_rpm):
            self._send(message)

    def _send_paced(self, messages: OscMessageList, interval: float) -> None:
        """Send `messages` spaced `interval` seconds apart, against a fixed deadline grid.

        The fixture drains one datagram per firmware loop over SPI while its socket buffer holds
        only about three; sending back-to-back overruns it and costs a chunk.
        """
        deadline = perf_counter()
        for message in messages:
            if interval > 0.0:
                remaining = deadline - perf_counter()
                if remaining > 0.0:
                    sleep(remaining)
                deadline += interval
            self._send(message)

    def _send(self, message: OscMessage) -> None:
        with self._client_lock:
            client = self._client
        try:
            client.send(message)
        except Exception as e:
            logger.error(f"OscLightSender send error: {e}")

    def _send_blackout(self) -> None:
        """Leave the fixture dark and stopped on a clean quit; see the class docstring.
        Paced like the live burst so the final frame is not the one datagram that drops."""
        self._send_paced(self._build_blackout_messages(self._config, self._chunk_size, self._num_chunks),
                         self._config.chunk_interval)

    def _on_connection_change(self, _=None) -> None:
        with self._client_lock:
            self._client = UDPClient(self._config.ip_addresses, self._config.port)
        self._config_dirty = True   # the new endpoint has not seen the offsets or rpm yet
        self._sent_rpm     = None
        logger.info(f"reconnected to {self._config.ip_addresses}:{self._config.port}")

    def _on_offsets_change(self, _=None) -> None:
        self._config_dirty = True

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rpm_message(rpm: float) -> OscMessage:
        """Motor speed command (`/WS/r/0`)."""
        msgb = OscMessageBuilder("/WS/r/0")
        msgb.add_arg(int(rpm))
        return msgb.build()

    @staticmethod
    def _build_blackout_messages(settings: OscLightSenderSettings, chunk_size: int, num_chunks: int) -> OscMessageList:
        """The shutdown blackout: rpm 0 **first** (deceleration starts at once), the six
        all-zero pixel chunks (`/WS/blue2` last commits the dark frame), and rpm 0 once
        more **last** — a lost single rpm datagram must not leave the motor spinning
        until the firmware watchdog."""
        zero = Frame(settings.resolution, Tick(0.0, 0.0, 0.0, 0.0, 0))
        chunks = OscLightSender._build_chunk_messages(zero, settings, chunk_size, num_chunks) or []
        return [OscLightSender._build_rpm_message(0), *chunks, OscLightSender._build_rpm_message(0)]

    @staticmethod
    def _build_config_messages(settings: OscLightSenderSettings, motor_rpm: int) -> OscMessageList:
        """The four lamp-alignment offsets plus the motor speed — constant for a whole show."""
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
        message_list.append(OscLightSender._build_rpm_message(motor_rpm))
        return message_list

    @staticmethod
    def _build_chunk_messages(
        output: Frame,
        settings: OscLightSenderSettings,
        chunk_size: int,
        num_chunks: int,
    ) -> Optional[OscMessageList]:
        """The pixel payload: all white chunks, then all blue.

        Grouping each channel keeps its chunks adjacent on the wire, and leaves `/WS/blue{n-1}`
        last — the message the firmware treats as the frame's commit trigger.
        """
        try:
            # Lamp output mapping: gamma curve + turn-on floor (master brightness applied upstream).
            white_f = OscLightSender._apply_levels(output.white, settings.curve, settings.lower_edge)
            blue_f  = OscLightSender._apply_levels(output.blue,  settings.curve, settings.lower_edge)
            if settings.use_signed:
                white_channel: np.ndarray = OscLightSender.float_to_int8(white_f)
                blue_channel:  np.ndarray = OscLightSender.float_to_int8(blue_f)
            else:
                white_channel = OscLightSender.float_to_uint8(white_f)
                blue_channel  = OscLightSender.float_to_uint8(blue_f)

            message_list: OscMessageList = []
            for prefix, channel in (("white", white_channel), ("blue", blue_channel)):
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx   = min((i + 1) * chunk_size, len(channel))
                    msgb = OscMessageBuilder(f"/WS/{prefix}{i}")
                    msgb.add_arg(channel[start_idx:end_idx].tobytes(), 'b')
                    message_list.append(msgb.build())
            return message_list
        except Exception as e:
            logger.error(f"OscLightSender error preparing data: {e}")
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
    def _apply_levels(arr: np.ndarray, curve: float, lower_edge: float) -> np.ndarray:
        """Lamp output mapping: gamma `curve`, then lift lit pixels above the turn-on floor
        (`lower_edge`); true-black pixels stay off. In/out in [0,1]."""
        x = np.clip(arr, 0.0, 1.0)
        s = x ** curve
        return np.where(s > 0.0, lower_edge + (1.0 - lower_edge) * s, 0.0)

    @staticmethod
    def float_to_uint8(arr: np.ndarray) -> np.ndarray:
        return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def float_to_int8(arr: np.ndarray) -> np.ndarray:
        return np.round(np.clip(arr, 0.0, 1.0) * 255.0 - 128.0).astype(np.int8)

