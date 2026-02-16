"""ArtNet RGBW LED bar controller using stupidArtnet.

Controls pairs of vertical RGBW LED bars with bar-fill visualization.
Each instance controls one controller (one IP) with two bars.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from threading import Thread, Event, Lock
import socket
import time

from stupidArtnet import StupidArtnet

from modules.ConfigBase import ConfigBase, config_field
from modules.inout.NetworkValidation import validate_connection

CONFIG_MODE: bool = False


class ChannelOrder(IntEnum):
    """Common RGBW channel orderings for different LED strips.

    Includes all permutations starting or ending with W.
    """
    # Ending with W
    RGBW = 0
    RBGW = 1
    GRBW = 2
    GBRW = 3
    BRGW = 4
    BGRW = 5
    # Starting with W
    WRGB = 6
    WRBG = 7
    WGRB = 8
    WGBR = 9
    WBRG = 10
    WBGR = 11


def _parse_channel_order(order_enum: ChannelOrder) -> tuple[int, int, int, int]:
    """Convert channel order enum to index tuple.

    Args:
        order_enum: ChannelOrder enum value

    Returns:
        Tuple of (r_idx, g_idx, b_idx, w_idx) for buffer positioning.
    """
    order = order_enum.name  # Get enum name (e.g., "RGBW", "GRBW")
    if len(order) != 4 or set(order) != {'R', 'G', 'B', 'W'}:
        raise ValueError(f"Invalid channel order '{order}'. Must be permutation of RGBW.")
    return (order.index('R'), order.index('G'), order.index('B'), order.index('W'))


@dataclass
class ArtNetLedConfig(ConfigBase):
    """Configuration for a single ArtNet LED bar controller.

    Fixed fields (set at init, then locked):
        ip_address: Controller IP address
        base_universe: First ArtNet universe (0-14). Second bar uses base_universe+1.
        fps: Update rate in frames per second
        num_pixels: Number of RGBW pixels per bar
        channel_order: DMX channel layout (e.g., "RGBW", "GRBW")

    Runtime fields (adjustable):
        white: Base white level for all pixels (0.0-1.0)
        color: Accent color brightness (0.0-1.0)
        bar: Bar fill level (0.0-1.0), 0=empty, 1=full
        enabled: Enable/disable output
    """

    verbose: bool = config_field(default=True, fixed=True, description="Enable detailed logging (warnings always shown)")

    # Fixed network settings
    ip_address: str = config_field(default="192.168.1.100", fixed= not CONFIG_MODE, description="Controller IP address")
    base_universe: int = config_field(default=0, fixed= not CONFIG_MODE, min=0, max=14, description="First ArtNet universe (second bar uses +1)")
    fps: int = config_field(default=40, fixed= not CONFIG_MODE, min=1, max=44, description="Update rate (frames per second)")

    # Accent color RGB components (0.0-1.0)
    r: float = config_field(default=1.0, fixed=not CONFIG_MODE, min=0.0, max=1.0, description="Red component of accent color")
    g: float = config_field(default=0.0, fixed=not CONFIG_MODE, min=0.0, max=1.0, description="Green component of accent color")
    b: float = config_field(default=0.0, fixed=not CONFIG_MODE, min=0.0, max=1.0, description="Blue component of accent color")

    # Fixed LED configuration
    num_pixels: int = config_field(default=90, fixed= not CONFIG_MODE, min=1, max=128, description="Number of RGBW pixels per bar")
    channel_order: ChannelOrder = config_field(default=ChannelOrder.RGBW, fixed= not CONFIG_MODE, description="DMX channel order (e.g., RGBW, GRBW, WRGB)")

    # Runtime controls
    enabled: bool = config_field(default=True, description="Enable ArtNet output")
    white: float = config_field(default=0.5, min=0.0, max=1.0, description="Base white intensity for all pixels")
    color: float = config_field(default=0.5, min=0.0, max=1.0, description="Accent color intensity")
    bar: float = config_field(default=0.5, min=0.0, max=1.0, description="Bar fill level (0=empty, 1=full)")


class ArtNetLed:
    """ArtNet controller for a pair of RGBW LED bars.

    Drives two vertical LED bars with identical bar-fill visualization.
    Each bar uses its own universe (base_universe and base_universe+1).
    Uses threading for continuous updates at configured FPS.

    Example:
        >>> config = ArtNetLedConfig(
        ...     ip_address="192.168.1.100",
        ...     num_pixels=90,
        ...     base_universe=0,  # Uses universes 0 and 1
        ...     channel_order=ChannelOrder.GRBW,
        ... )
        >>> controller = ArtNetLed(config)
        >>> controller.start()
        >>>
        >>> # Runtime adjustments
        >>> config.bar = 0.5    # Half-filled bar
        >>> config.white = 0.2  # Dim white base
        >>>
        >>> controller.stop()
    """

    def __init__(self, config: ArtNetLedConfig) -> None:
        self._config: ArtNetLedConfig = config
        self._lock: Lock = Lock()

        # Parse channel order
        self._channel_indices: tuple[int, int, int, int] = _parse_channel_order(config.channel_order)

        # Calculate DMX size: single bar × num_pixels × 4 channels (RGBW)
        # Both bars receive the same data on consecutive universes
        self._channels_per_pixel: int = 4
        self._total_channels: int = config.num_pixels * self._channels_per_pixel

        # Initialize ArtNet
        self._artnet: StupidArtnet = StupidArtnet(
            config.ip_address,
            config.base_universe,
            self._total_channels,
            config.fps,
            True,  # even_packet_size
            False   # broadcast
        )

        # DMX buffer (single bar, sent to both universes)
        self._buffer: bytearray = bytearray(self._total_channels)

        # Threading
        self._running: bool = False
        self._update_event: Event = Event()
        self._thread: Thread | None = None
        self._dirty: bool = True  # Frame needs recalculation

        # Periodic sending
        self._last_periodic_time: float = 0.0
        self._periodic_interval: float = 1.0  # Send at least once per second

        # Watch config changes to trigger immediate update
        self._unwatchers: list = []
        self._setup_watchers()

        print(f"ArtNetLed: Initialized for {config.ip_address} universes {config.base_universe}/{config.base_universe + 1}, "
              f"{config.num_pixels} pixels/bar")

    def start(self) -> None:
        """Start the ArtNet output thread.

        Note: Network validation happens at thread start.
        """
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._update_loop, daemon=True, name=f"ArtNetLed-{self._config.ip_address}")
        self._thread.start()
        if self._config.verbose:
            print(f"ArtNetLed: Starting output to {self._config.ip_address}...")

    def stop(self) -> None:
        """Stop the ArtNet output thread and blackout."""
        if not self._running:
            return

        self._running = False
        self._update_event.set()  # Wake up thread to exit

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Blackout using internal method
        self._blackout()

        # Clean up watchers
        for unwatch in self._unwatchers:
            unwatch()
        self._unwatchers.clear()

        if self._config.verbose:
            print(f"ArtNetLed: Stopped output to {self._config.ip_address}")

    def set_bar(self, value: float) -> None:
        """Set the bar fill level.

        Args:
            value: Bar fill level (0.0-1.0), where 0=empty, 1=full
        """
        self._config.bar = max(0.0, min(1.0, value))

    def _setup_watchers(self) -> None:
        """Set up config watchers to trigger updates on change."""
        # Global watcher: mark dirty and wake thread on any change
        self._unwatchers.append(self._config.watch(self._on_config_change))

        # Specific handlers for fields that need reinit
        self._unwatchers.append(self._config.watch(lambda val: self._validate_and_reinit_artnet(), 'ip_address'))
        self._unwatchers.append(self._config.watch(lambda val: self._reinit_artnet(), 'fps'))
        self._unwatchers.append(self._config.watch(lambda val: self._update_universe(), 'base_universe'))
        self._unwatchers.append(self._config.watch(lambda val: self._update_packet_size(), 'num_pixels'))
        self._unwatchers.append(self._config.watch(lambda val: self._reinit_channel_order(), 'channel_order'))

        # Blackout when disabled
        self._unwatchers.append(self._config.watch(lambda val: self._on_enabled_change(val), 'enabled'))

    def _on_enabled_change(self, enabled: bool) -> None:
        """Called when enabled state changes."""
        if not enabled:
            self._blackout()
            if self._config.verbose:
                print(f"ArtNetLed: Output disabled, LEDs blacked out")

    def _on_config_change(self) -> None:
        """Called on any config change - mark dirty and wake thread."""
        self._dirty = True
        self._update_event.set()

    def _validate_and_reinit_artnet(self) -> None:
        """Reinitialize ArtNet connection after IP change with validation."""
        # Validate the new IP before reinitializing
        if not validate_connection(self._config.ip_address, 5568, "ArtNetLed"):
            print(f"ArtNetLed WARNING: New IP {self._config.ip_address} is not reachable. Changes saved but output may fail.")
            # Don't return - still reinit so user can fix it later

        self._reinit_artnet()

    def _update_packet_size(self) -> None:
        """Update packet size when num_pixels changes."""
        with self._lock:
            self._total_channels = self._config.num_pixels * self._channels_per_pixel
            self._artnet.set_packet_size(self._total_channels)
            self._artnet.clear()  # Sync internal buffer to new packet_size
            self._buffer = bytearray(self._total_channels)
            if self._config.verbose:
                print(f"ArtNetLed: Updated packet size to {self._total_channels} channels ({self._config.num_pixels} pixels/bar)")

    def _update_universe(self) -> None:
        """Update universe when it changes."""
        with self._lock:
            self._artnet.set_universe(self._config.base_universe)
            if self._config.verbose:
                print(f"ArtNetLed: Updated base universe to {self._config.base_universe}/{self._config.base_universe + 1}")

    def _reinit_artnet(self) -> None:
        """Full reinitialize for IP or FPS changes (requires new instance)."""
        was_running = self._running

        # Stop everything first if running
        if was_running:
            self._running = False
            self._update_event.set()
            if self._thread:
                self._thread.join(timeout=2.0)
                self._thread = None

        with self._lock:
            # Blackout and close the old ArtNet instance
            if self._artnet is not None:
                try:
                    self._blackout()
                    self._artnet.close()
                except:
                    pass

            # Recalculate DMX size (single bar)
            self._total_channels = self._config.num_pixels * self._channels_per_pixel
            self._buffer = bytearray(self._total_channels)

            # Recreate ArtNet instance
            self._artnet = StupidArtnet(
                self._config.ip_address,
                self._config.base_universe,
                self._total_channels,
                self._config.fps,
                True,  # even_packet_size
                False   # broadcast
            )

            if self._config.verbose:
                print(f"ArtNetLed: Recreated instance for {self._config.ip_address} universes {self._config.base_universe}/{self._config.base_universe + 1}, "
                      f"{self._config.num_pixels} pixels/bar, {self._total_channels} channels, {self._config.fps} fps")

        # Restart if it was running
        if was_running:
            self._running = True
            self._thread = Thread(target=self._update_loop, daemon=True, name=f"ArtNetLed-{self._config.ip_address}")
            self._thread.start()
            if self._config.verbose:
                print(f"ArtNetLed: Restarted output thread")

    def _reinit_channel_order(self) -> None:
        """Update channel order parsing."""
        with self._lock:
            try:
                self._channel_indices = _parse_channel_order(self._config.channel_order)
                if self._config.verbose:
                    print(f"ArtNetLed: Channel order updated to {self._config.channel_order.name}")
            except ValueError as e:
                print(f"ArtNetLed: Invalid channel order - {e}")

    def _blackout(self) -> None:
        """Clear all pixels (set to 0) and send to both universes."""
        with self._lock:
            # Zero out the entire buffer
            for i in range(len(self._buffer)):
                self._buffer[i] = 0

        # Send black frame to both universes
        self._artnet.set_universe(self._config.base_universe)
        self._artnet.set(self._buffer)
        self._artnet.show()
        self._artnet.set_universe(self._config.base_universe + 1)
        self._artnet.show()

    def _set_pixel(self, pixel_index: int, r: int, g: int, b: int, w: int) -> None:
        """Set a single pixel's RGBW values in the buffer.

        Args:
            pixel_index: Pixel index (0 to num_pixels-1)
            r, g, b, w: Color values (0-255)
        """
        base: int = pixel_index * self._channels_per_pixel
        r_idx, g_idx, b_idx, w_idx = self._channel_indices

        self._buffer[base + r_idx] = r
        self._buffer[base + g_idx] = g
        self._buffer[base + b_idx] = b
        self._buffer[base + w_idx] = w

    def _compute_frame(self) -> None:
        """Compute full DMX frame for the LED bar.

        Both bars receive identical data (sent to consecutive universes).
        Bar fills from bottom (pixel 0) upward.
        """
        num_pixels: int = self._config.num_pixels
        bar_value: float = self._config.bar
        white_intensity: float = self._config.white
        color_intensity: float = self._config.color

        # Get accent color RGB (0-255 range)
        accent_r = int(self._config.r * 255)
        accent_g = int(self._config.g * 255)
        accent_b = int(self._config.b * 255)

        # Calculate how many pixels are "lit" by the bar
        lit_count: int = int(bar_value * num_pixels + 0.5)  # Round to nearest

        # Base white value for all pixels
        base_w: int = int(white_intensity * 255)

        for i in range(num_pixels):
            # Pixels 0..lit_count-1 are lit (bottom-up)
            is_lit = i < lit_count

            if is_lit:
                # Lit pixel: accent color + white
                r = int(accent_r * color_intensity)
                g = int(accent_g * color_intensity)
                b = int(accent_b * color_intensity)
                w = base_w
            else:
                # Unlit pixel: only white base
                r, g, b = 0, 0, 0
                w = base_w

            self._set_pixel(i, r, g, b, w)

    def _send_frame(self) -> None:
        """Send current buffer via ArtNet to both universes."""
        if self._config.enabled:
            # Send to first universe (left bar)
            self._artnet.set_universe(self._config.base_universe)
            self._artnet.set(self._buffer)
            self._artnet.show()
            # Send to second universe (right bar) - same data
            self._artnet.set_universe(self._config.base_universe + 1)
            self._artnet.show()

    def _update_loop(self) -> None:
        """Background thread loop for continuous updates."""
        # Validate network and IP before starting
        if not validate_connection(self._config.ip_address, 5568, "ArtNetLed"):
            self._running = False
            return

        # Blackout before starting the loop
        self._blackout()

        while self._running:
            try:
                # Periodic dirty trigger: ensure send at least once per interval
                current_time: float = time.perf_counter()
                if current_time - self._last_periodic_time >= self._periodic_interval:
                    self._dirty = True
                    self._last_periodic_time = current_time

                if self._dirty:
                    with self._lock:
                        self._compute_frame()
                    self._dirty = False
                    # Send outside lock to avoid blocking config updates during I/O
                    self._send_frame()

                # Wait for next frame or config change
                frame_time: float = 1.0 / self._config.fps
                self._update_event.wait(timeout=frame_time)
                self._update_event.clear()
            except socket.error as e:
                print(f"ArtNetLed ERROR: Socket error with exception: {e}")
                self._running = False
                break
            except Exception as e:
                print(f"ArtNetLed: Error in update loop: {e}")

        # Blackout after exiting the loop
        self._blackout()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self._running:
            self.stop()
        elif self._unwatchers:
            for unwatch in self._unwatchers:
                unwatch()
            self._unwatchers.clear()

    @property
    def config(self) -> ArtNetLedConfig:
        """Access the configuration object."""
        return self._config
