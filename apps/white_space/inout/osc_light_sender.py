from threading import Thread, Event, Lock
from time import monotonic, perf_counter, sleep

import numpy as np
from pythonosc.udp_client import UDPClient
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

from ..light import LightFrame, Tick, BeamLightId, BEAM_LIGHT_CHANNEL, FIXTURE_PROJECTION_RPM, MotorSettings
from modules.settings import BaseSettings, Field, Group, Widget
from modules.inout import validate_connection

import logging
logger = logging.getLogger(__name__)


OscMessageList = list[OscMessage]

# The fixture firmware hard-codes a 1200-byte body and exactly three chunks per channel, and finds
# the chunk digit at a fixed byte offset in the address. Changing `resolution` or `mtu` such that
# `_calculate_optimal_chunks` yields anything else silently breaks the installation.
FIRMWARE_CHUNK_SIZE: int = 1200
FIRMWARE_NUM_CHUNKS: int = 3

# The firmware's readout mode follows the *commanded* rpm, switching on receipt of `/WS/r/0`
# regardless of the bar's actual speed (firmware.cpp line 475). Below FIXTURE_PROJECTION_RPM it is in
# beam mode: the four DACs are copied from four fixed pixels — these slots, as normalized azimuths
# per beam light (index = BeamLightId; the channel is BEAM_LIGHT_CHANNEL) — and every other pixel is
# never read (lines 297-305). At or above it paints the projection and never reads a fixed slot
# (lines 266-295). So a slot carries the beam light's level in beam mode and the projection otherwise.
FIRMWARE_LIGHT_SLOT_AZIMUTHS: np.ndarray = np.array([0.0, 0.5, 0.0, 0.5])

# Interlace and rpm are constant for a whole show, so they are sent on change only. This keepalive
# re-sends them anyway at a low rate so a lost config packet self-heals; the firmware ignores a
# repeat of the value it already holds.
_CONFIG_KEEPALIVE_S: float = 1.0

# How often to re-probe an unreachable endpoint at startup; each failed probe logs an error.
_CONNECT_RETRY_S: float = 5.0


class OscLightInterlaceSettings(BaseSettings):
    """Interlace: the two arms' LEDs have gaps and are mounted out of phase, so one arm's LEDs
    fill the gaps of the other's and the two halves of a revolution interlace into one image.
    Shift each beam light (pixels, 1 px = 0.1°) until the two sides paint one image: the whites
    against each other, the blues against each other, then the blue image onto the white.
    Projection mode only — the firmware ignores these below FIXTURE_PROJECTION_RPM."""
    white_0: Field[int] = Field(0, min=-10, max=10, description="Front white beam light shift (pixels)")
    white_1: Field[int] = Field(0, min=-10, max=10, description="Back white beam light shift (pixels)")
    blue_0:  Field[int] = Field(0, min=-10, max=10, description="Left blue beam light shift (pixels)")
    blue_1:  Field[int] = Field(0, min=-10, max=10, description="Right blue beam light shift (pixels)")


class OscLightSenderSettings(BaseSettings):
    ip:           Field[str]  = Field("127.0.0.1", widget=Widget.ip_field,      description="Target LED receiver IP address")
    port:         Field[int]  = Field(8000, min=1024, max=65535, widget=Widget.number_field, description="Target UDP port")
    use_signed:   Field[bool] = Field(False,                                     description="Send signed int8 instead of uint8")
    resolution:   Field[int]  = Field(3600, min=256, max=4096, step=16, access=Field.INIT, visible=False, description="Projection resolution (pixels per turn)")
    mtu:          Field[int]  = Field(1500, min=576, max=9000,  access=Field.INIT, description="Network MTU (affects chunk size)")
    chunk_size:    Field[int]  = Field(0,    access=Field.READ,  description="Computed chunk size (bytes)")
    num_chunks:   Field[int]  = Field(0,    access=Field.READ,  description="Computed number of chunks")
    lower_edge:   Field[float] = Field(0.35, min=0.0, max=1.0, step=0.01, description="LED driver turn-on floor: lit pixels lift to at least this; black stays off")
    upper_edge:   Field[float] = Field(1.0,  min=0.0, max=1.0, step=0.01, description="LED driver saturation ceiling: software 1.0 maps to this")
    curve:        Field[float] = Field(1.0,  min=0.5, max=3.0, step=0.01, description="Output gamma curve; <1 brightens mids, >1 darkens")
    startup_delay: Field[float] = Field(2.0, min=0.0, max=10.0, step=0.5, description="Motor rpm held at 0 after connect, for the 0→target edge at boot (s)")
    resync_hold:  Field[float] = Field(0.3, min=0.1, max=2.0, step=0.1, description="Rpm-0 hold after a send gap, to restart a watchdog-stopped motor (s)")
    chunk_interval: Field[float] = Field(0.0,    min=0.0, max=0.005, step=0.0005, description="Gap between pixel datagrams, only against dropped chunks (s); 0 = back-to-back")
    projection_offset: Field[float] = Field(0.0, min=0.0, max=360.0, step=0.1, description="Projection rotation that puts the authored azimuth in place in the room (°)", newline=True)
    interlace:    Group[OscLightInterlaceSettings] = Group(OscLightInterlaceSettings)


# The firmware's watchdog stops the motor after >5 revolutions of packet silence (firmware.cpp
# line 426). Assume it may have fired after this many — the margin covers a physical speed slightly
# above the ceiling's estimate.
_WATCHDOG_SAFETY_REVS: float = 4.0

# Slowest assumed spin-down (rpm/s) for the physical-speed ceiling. The real brake is ~3× the
# ~333 rpm/s spin-up (see `light/motor.py` _SIM_DECEL); staying well under it keeps the ceiling
# above the bar's true speed while it brakes down from PROJECTION.
_CEILING_DECAY_RPM_S: float = 250.0


class FixtureSpeedLatch:
    """Host-side model of the fixture's speed latch — decides the rpm to put on the wire.

    The firmware applies a received rpm only when it differs from the last one it applied
    (firmware.cpp line 543, `vRPM`), and its packet watchdog stops the motor after more than
    5 revolutions of packet silence *without recording that stop* (line 426). So after any send
    silence — simulation, a host hitch, a stalled sender — the fixture's latch is stale:
    re-sending the same rpm is a no-op and a watchdog-stopped motor never restarts.

    While the latch is untrusted — before anything has been sent, after a send gap long enough
    that the watchdog may have fired, or after `invalidate()` — `rpm_to_send` yields 0 for a
    hold period and then releases to the target, forcing the 0→target edge the firmware acts on.
    The hold is anchored at the send that begins it, not at `begin_hold`, so a hold armed while
    nothing is sent (boot with simulation on) runs in full once sending starts.

    The watchdog counts *physical* revolutions and the bar can spin faster than the last
    commanded rpm (braking down from PROJECTION), so the gap bound uses a decaying ceiling of
    the physical speed rather than the last sent value.

    Owned by the sender thread; only `invalidate()` may be called from other threads (an atomic
    flag, same pattern as the sender's `_config_dirty`).
    """

    def __init__(self) -> None:
        self._pending_hold_s: float = 0.0           # armed hold length; anchored at the next send
        self._hold_until:     float | None = None   # active hold deadline (sender-thread clock)
        self._last_send:      float | None = None   # when the fixture last got a packet from us
        self._ceiling:        float = 0.0           # upper bound on the bar's physical rpm
        self._invalidated:    bool  = False         # cross-thread resync request

    def invalidate(self) -> None:
        """Distrust the latch from any thread; the sender converts it to a hold on its next pass."""
        self._invalidated = True

    def begin_hold(self, seconds: float) -> None:
        """Arm a hold of `seconds` at rpm 0, starting at the next send. Never shortens one."""
        self._pending_hold_s = max(self._pending_hold_s, seconds)

    def note_sent(self, rpm: int, now: float) -> None:
        """Record datagrams put on the wire: any packet resets the fixture's watchdog counter."""
        if self._last_send is not None:
            self._ceiling = max(0.0, self._ceiling - _CEILING_DECAY_RPM_S * (now - self._last_send))
        self._ceiling = max(self._ceiling, float(rpm))
        self._last_send = now

    def rpm_to_send(self, target_rpm: int, resync_hold_s: float, now: float) -> int:
        """The rpm to put on the wire this frame: `target_rpm`, or 0 while re-establishing."""
        if self._invalidated:
            self._invalidated = False
            self.begin_hold(resync_hold_s)
        if self._last_send is not None and self._hold_until is None and self._pending_hold_s == 0.0:
            # Revolutions the bar may have made since our last packet, at the ceiling rate.
            unseen_revs = (now - self._last_send) * self._ceiling / 60.0
            if unseen_revs > _WATCHDOG_SAFETY_REVS:
                self.begin_hold(resync_hold_s)
        if self._pending_hold_s > 0.0:
            self._hold_until = now + self._pending_hold_s   # anchor the hold at this send
            self._pending_hold_s = 0.0
        if self._hold_until is not None:
            if now < self._hold_until:
                return 0
            self._hold_until = None
        return target_rpm


class OscLightSender:
    """Sends the projection and the beam lights over OSC/UDP to the installation hardware.

    OSC address pattern (all under /WS/):
        /WS/o/0..3        -- interlace: white_0, white_1, blue_0, blue_1 (int pixels)
        /WS/r/0           -- rotation: RPM (int)
        /WS/white{i}      -- white channel chunk i (blob)
        /WS/blue{i}       -- blue channel chunk i (blob)

    Wire contract with the fixture firmware (`apps/white_space/data/firmware/firmware.ino`):

    * A chunk datagram is exactly 1220 bytes — a 20-byte OSC preamble (12-byte padded address,
      4-byte typetag, 4-byte blob length) followed by 1200 pixel bytes. The firmware reads exactly
      those 20 + 1200 bytes, and locates the chunk digit at a fixed offset in the address, so both
      the chunk size and the address spelling are load-bearing.
    * The firmware commits a frame when `/WS/blue2` arrives, so that message must be sent **last**.
    * Its readout mode follows the **commanded** rpm (`/WS/r/0`), not the bar's speed: below
      `FIXTURE_PROJECTION_RPM` it drives the four lamps from four fixed pixel slots (pixel 0 and the
      middle pixel of each channel, `FIRMWARE_LIGHT_SLOT_AZIMUTHS`) and ignores every other pixel; at or
      above it paints the projection and ignores the slots. `_rebuild_fixture_pixels` writes the frame's
      explicit beam lights into those slots exactly when the fixture reads them, using the rpm
      this sender actually put on the wire.
    * This sender owns the fixture's two projection-mode alignments, so the frame on the board
      stays azimuth-true: `projection_offset` rotates the whole projection on its way out, and the
      four `interlace` values shift each lamp's readout (`/WS/o/0..3`). Both are ignored by the
      firmware in beam mode, so `_rebuild_fixture_pixels` never rotates the four slots.
    * Its socket receive buffer is 8 KB against a 7.4 KB frame, and it drains while it fills, so
      the burst normally fits. Config messages are kept out of it anyway (they never change), and
      `chunk_interval` can spread the six chunks further if the fixture ever reports dropped
      chunks — a chunk it misses is published as a stale third of the projection for one revolution.
    * On a clean quit, ``stop()`` ends with a **blackout** — rpm 0, an all-zero frame, rpm 0
      again — so the fixture goes dark and the motor decelerates immediately. The firmware's
      Ethernet watchdog (packet silence → motor stop + blank) covers the crash path only.
    * The firmware acts on the rpm only when it **changes** (its `vRPM` latch), and its watchdog
      stops the motor without recording that stop — so after any send silence, repeating the old
      rpm would leave a stopped motor stopped forever. `FixtureSpeedLatch` guards this: at boot,
      after a reconnect, and after any gap long enough that the watchdog may have fired, the rpm
      is held at 0 (`startup_delay` at boot, `resync_hold` otherwise) and then released, giving
      the firmware the 0→target edge it acts on.
    * While the motor is simulated (``motor.simulate``) nothing is sent at all — no rpm, no
      pixels — so a simulated motor never reaches the fixture; the flag is read live, so
      switching it off sends from the next frame — through a resync hold, since the silence has
      usually tripped the fixture's watchdog.
    """

    def __init__(self, settings: OscLightSenderSettings, motor: MotorSettings) -> None:
        self.settings: OscLightSenderSettings = settings
        self._motor = motor             # while the motor is simulated, nothing is sent
        self._chunk_size, self._num_chunks = self._calculate_optimal_chunks(settings.resolution, settings.mtu)
        self.settings.chunk_size = self._chunk_size
        self.settings.num_chunks = self._num_chunks

        if (self._chunk_size, self._num_chunks) != (FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS):
            logger.error(
                "chunking is %s x %s bytes but the fixture firmware only accepts %s x %s — "
                "the fixture will not display correctly. Check `resolution` (%s) and `mtu` (%s).",
                self._num_chunks, self._chunk_size, FIRMWARE_NUM_CHUNKS, FIRMWARE_CHUNK_SIZE,
                settings.resolution, settings.mtu,
            )

        self._latest_output: LightFrame | None = None
        self._latch = FixtureSpeedLatch()   # when the fixture's speed latch needs a 0→target edge
        self._output_lock:   Lock  = Lock()
        self._client_lock:   Lock  = Lock()
        self._update_event:  Event = Event()
        self._client: UDPClient = UDPClient(settings.ip, settings.port)

        # Config (interlace + rpm) is sent on change instead of every frame; `None` rpm forces the
        # first send. `_config_dirty` is set from the settings callback on whichever thread edits.
        self._config_dirty:   bool = True
        self._sent_rpm:       int | None = None
        self._last_config_at: float = 0.0

        self._running = False
        self._thread: Thread | None = None

        self.settings.bind(OscLightSenderSettings.ip, self._on_connection_change)
        self.settings.bind(OscLightSenderSettings.port,         self._on_connection_change)
        self.settings.interlace.bind_all(self._on_interlace_change)

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

        self.settings.unbind(OscLightSenderSettings.ip, self._on_connection_change)
        self.settings.unbind(OscLightSenderSettings.port,         self._on_connection_change)
        self.settings.interlace.unbind_all(self._on_interlace_change)

        # Send AFTER the join: guaranteed to be the last thing on the wire (late
        # submit_light_frame() calls only store into the dead thread's slot).
        if started:
            try:
                self._send_blackout()
            except Exception as e:
                logger.error(f"Error sending light blackout: {e}")

    def submit_light_frame(self, frame: LightFrame) -> None:
        with self._output_lock:
            self._latest_output = frame
        self._update_event.set()

    def _run(self) -> None:
        # No thread priority boost (see Conductor.run): a boost cannot speed this thread's Python
        # up under the GIL, it only preempts the rest of the app.
        # At boot the fixture may come up after the app: keep probing instead of giving up — a
        # sender that exits here would never send, and the fixture's watchdog would park the
        # motor (dark, stopped) for the whole session. The endpoint is read live, so a settings
        # change mid-retry is picked up.
        while self._running and not validate_connection(self.settings.ip, self.settings.port, "OscLightSender"):
            retry_at = monotonic() + _CONNECT_RETRY_S
            while self._running and monotonic() < retry_at:
                self._update_event.wait(0.5)   # frames keep setting the event; the deadline paces the probe
                self._update_event.clear()
        if not self._running:
            return

        logger.info(
            f"{self.settings.ip}:{self.settings.port}, "
            f"resolution={self.settings.resolution}, "
            f"{self._num_chunks} chunks of {self._chunk_size} bytes each."
        )

        self._latch.begin_hold(self.settings.startup_delay)   # boot edge; anchored at the first send

        while self._running:
            self._update_event.wait()
            self._update_event.clear()
            if not self._running:
                break
            # A dead sender is packet silence to the fixture (watchdog → motor parked), so an
            # unexpected error skips this frame instead of ending the thread (see Conductor.run).
            try:
                # Take the frame *and* clear the slot under one lock: a plain read leaves the event
                # set when a new frame lands between the clear and the read, which resends the
                # same frame.
                with self._output_lock:
                    output, self._latest_output = self._latest_output, None
                if output is None:
                    continue
                if self._motor.simulate:
                    continue        # a simulated motor never reaches the fixture: no rpm, no pixels

                # While the fixture's speed latch is untrusted — boot, or a send gap long enough
                # that its watchdog may have stopped the motor without recording it — hold rpm at
                # 0, then release: the 0→target edge is the only speed command the firmware acts on.
                now = monotonic()
                motor_rpm = self._latch.rpm_to_send(int(output.motor_command.target_rpm),
                                                    self.settings.resync_hold, now)
                sent = self._send_config(motor_rpm)

                # The fixture's readout mode follows the rpm we just sent — so does the slot rebuild.
                slow = motor_rpm < FIXTURE_PROJECTION_RPM
                chunks = self._build_chunk_messages(output, self.settings, self._chunk_size, self._num_chunks, slow)
                if chunks:
                    self._send_paced(chunks, self.settings.chunk_interval)
                    sent = True
                if sent:   # a pass that put nothing on the wire must not reset the latch's gap clock
                    self._latch.note_sent(motor_rpm, now)
            except Exception:
                logger.exception("Error sending light frame")

    def _send_config(self, motor_rpm: int) -> bool:
        """Send the interlace and rpm — on change, or when the keepalive interval has elapsed.
        Returns whether it sent.

        Keeping these five datagrams out of the per-frame burst is what buys the fixture's socket
        buffer enough headroom for the six pixel chunks.
        """
        now = monotonic()
        due = self._config_dirty or motor_rpm != self._sent_rpm or (now - self._last_config_at) >= _CONFIG_KEEPALIVE_S
        if not due:
            return False
        self._config_dirty   = False
        self._sent_rpm       = motor_rpm
        self._last_config_at = now
        for message in self._build_config_messages(self.settings, motor_rpm):
            self._send(message)
        return True

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
        self._send_paced(self._build_blackout_messages(self.settings, self._chunk_size, self._num_chunks),
                         self.settings.chunk_interval)

    def _on_connection_change(self, _=None) -> None:
        with self._client_lock:
            self._client = UDPClient(self.settings.ip, self.settings.port)
        self._config_dirty = True   # the new endpoint has not seen the interlace or rpm yet
        self._sent_rpm     = None
        self._latch.invalidate()    # nor do we know the speed its latch holds
        logger.info(f"reconnected to {self.settings.ip}:{self.settings.port}")

    def _on_interlace_change(self, _=None) -> None:
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
        zero = LightFrame(settings.resolution, Tick(0.0, 0.0))
        chunks = OscLightSender._build_chunk_messages(zero, settings, chunk_size, num_chunks, slow=True) or []
        return [OscLightSender._build_rpm_message(0), *chunks, OscLightSender._build_rpm_message(0)]

    @staticmethod
    def _build_config_messages(settings: OscLightSenderSettings, motor_rpm: int) -> OscMessageList:
        """The four interlace shifts plus the motor speed — constant for a whole show."""
        message_list: OscMessageList = []
        for addr, val in (
            ("/WS/o/0", settings.interlace.white_0),
            ("/WS/o/1", settings.interlace.white_1),
            ("/WS/o/2", settings.interlace.blue_0),
            ("/WS/o/3", settings.interlace.blue_1),
        ):
            off_msgb = OscMessageBuilder(addr)
            off_msgb.add_arg(val)
            message_list.append(off_msgb.build())
        message_list.append(OscLightSender._build_rpm_message(motor_rpm))
        return message_list

    @staticmethod
    def _rebuild_fixture_pixels(output: LightFrame, slow: bool, shift: int) -> tuple[np.ndarray, np.ndarray]:
        """The (white, blue) pixel channels as the fixture will read them.

        In projection mode the projection is what the fixture paints, so it goes out rotated
        by ``shift`` — the projection offset in pixels — and the beam lights are dropped (a slot
        value would only be a one-pixel blip in the image).

        In beam mode the firmware reads only the four fixed slots, so each slot is **replaced**
        by its beam light's level and the rotation is skipped: the slots are pixel positions the
        firmware indexes directly, and the pixels around them are never seen either way.

        Never mutates the shared frame.
        """
        if not slow:
            if shift:
                return np.roll(output.white, shift), np.roll(output.blue, shift)
            return output.white, output.blue
        white, blue = output.white.copy(), output.blue.copy()
        slots = (FIRMWARE_LIGHT_SLOT_AZIMUTHS * output.resolution).astype(int)
        for light in BeamLightId:
            channel = white if BEAM_LIGHT_CHANNEL[light] == 0 else blue
            channel[slots[light]] = output.beam_lights[light]
        return white, blue

    @staticmethod
    def _projection_shift(settings: OscLightSenderSettings) -> int:
        """The projection offset in whole pixels."""
        return int(round(settings.projection_offset / 360.0 * settings.resolution)) % settings.resolution

    @staticmethod
    def _build_chunk_messages(
        output: LightFrame,
        settings: OscLightSenderSettings,
        chunk_size: int,
        num_chunks: int,
        slow: bool,
    ) -> OscMessageList | None:
        """The pixel payload: all white chunks, then all blue.

        Grouping each channel keeps its chunks adjacent on the wire, and leaves `/WS/blue{n-1}`
        last — the message the firmware treats as the frame's commit trigger. ``slow`` is the
        fixture's readout mode for this frame (the rpm sent with it below `FIXTURE_PROJECTION_RPM`).
        """
        try:
            shift = OscLightSender._projection_shift(settings)
            white_src, blue_src = OscLightSender._rebuild_fixture_pixels(output, slow, shift)
            # Output mapping: gamma curve + the driver's usable window (master applied upstream).
            white_f = OscLightSender._apply_levels(white_src, settings.curve, settings.lower_edge, settings.upper_edge)
            blue_f  = OscLightSender._apply_levels(blue_src,  settings.curve, settings.lower_edge, settings.upper_edge)
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
    def _apply_levels(arr: np.ndarray, curve: float, lower_edge: float, upper_edge: float = 1.0) -> np.ndarray:
        """Output mapping: gamma `curve`, then map lit pixels into the analog LED driver's
        usable window (`lower_edge`..`upper_edge` — dark below the forward threshold,
        saturated above); true-black pixels stay off. In/out in [0,1]."""
        x = np.clip(arr, 0.0, 1.0)
        s = x ** curve
        hi = max(upper_edge, lower_edge)   # a crossed pair collapses to the floor, never inverts
        return np.where(s > 0.0, lower_edge + (hi - lower_edge) * s, 0.0)

    @staticmethod
    def float_to_uint8(arr: np.ndarray) -> np.ndarray:
        return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def float_to_int8(arr: np.ndarray) -> np.ndarray:
        return np.round(np.clip(arr, 0.0, 1.0) * 255.0 - 128.0).astype(np.int8)

