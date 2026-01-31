"""ArtNet RGBW LED bar controller using stupidArtnet.

Controls pairs of vertical RGBW LED bars with bar-fill visualization.
Each instance controls one controller (one IP) with two bars.
"""

from dataclasses import dataclass, field
from enum import Enum
from threading import Thread, Event, Lock
import ipaddress
import subprocess

from stupidArtnet import StupidArtnet

from modules.ConfigBase import ConfigBase, config_field

CONFIG_MODE: bool = True

class SelectedColor(Enum):
    """Available accent colors for the LED bar."""
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"


class ChannelOrder(Enum):
    """Common RGBW channel orderings for different LED strips."""
    RGBW = "RGBW"
    GRBW = "GRBW"
    BRGW = "BRGW"
    WRGB = "WRGB"
    RGWB = "RGWB"


def _parse_channel_order(order: str) -> tuple[int, int, int, int]:
    """Convert channel order string to index tuple.

    Args:
        order: 4-character string like "RGBW", "GRBW", etc.

    Returns:
        Tuple of (r_idx, g_idx, b_idx, w_idx) for buffer positioning.
    """
    order = order.upper()
    if len(order) != 4 or set(order) != {'R', 'G', 'B', 'W'}:
        raise ValueError(f"Invalid channel order '{order}'. Must be permutation of RGBW.")
    return (order.index('R'), order.index('G'), order.index('B'), order.index('W'))


@dataclass
class ArtNetLedConfig(ConfigBase):
    """Configuration for a single ArtNet LED bar controller.

    Fixed fields (set at init, then locked):
        ip_address: Controller IP address
        universe: ArtNet universe (0-15 typical)
        fps: Update rate in frames per second
        num_pixels: Number of RGBW pixels per bar
        channel_order: DMX channel layout (e.g., "RGBW", "GRBW")
        bar_left_inverted: Fill direction for left bar
        bar_right_inverted: Fill direction for right bar

    Runtime fields (adjustable):
        white: Base white level for all pixels (0.0-1.0)
        color: Accent color brightness (0.0-1.0)
        bar: Bar fill level (0.0-1.0), 0=empty, 1=full
        selected_color: Accent color (RED, BLUE, YELLOW)
        enabled: Enable/disable output
    """

    verbose: bool = config_field(
        default=True,
        fixed=True,
        description="Enable detailed logging (warnings always shown)"
    )

    # Fixed network settings
    ip_address: str = config_field(
        default="192.168.1.100",
        fixed= not CONFIG_MODE,
        description="Controller IP address"
    )

    universe: int = config_field(
        default=0,
        fixed= not CONFIG_MODE,
        min=0,
        max=15,
        description="ArtNet universe"
    )

    fps: int = config_field(
        default=40,
        fixed= not CONFIG_MODE,
        min=1,
        max=44,  # ArtNet spec max
        description="Update rate (frames per second)"
    )

    # Accent color RGB components (0.0-1.0)
    r: float = config_field(
        default=1.0,
        fixed=not CONFIG_MODE,
        min=0.0,
        max=1.0,
        description="Red component of accent color"
    )

    g: float = config_field(
        default=0.0,
        fixed=not CONFIG_MODE,
        min=0.0,
        max=1.0,
        description="Green component of accent color"
    )

    b: float = config_field(
        default=0.0,
        fixed=not CONFIG_MODE,
        min=0.0,
        max=1.0,
        description="Blue component of accent color"
    )
    # Fixed LED configuration
    num_pixels: int = config_field(
        default=30,
        fixed= not CONFIG_MODE,
        min=1,
        max=64,  # Max pixels for single universe (512/3 for RGB, 512/4=128 for RGBW)
        description="Number of RGBW pixels per bar"
    )

    channel_order: ChannelOrder = config_field(
        default=ChannelOrder.RGBW,
        fixed= not CONFIG_MODE,
        description="DMX channel order (e.g., RGBW, GRBW, WRGB)"
    )

    bar_left_inverted: bool = config_field(
        default=False,
        fixed= not CONFIG_MODE,
        description="Invert fill direction for left bar"
    )

    bar_right_inverted: bool = config_field(
        default=False,
        fixed= not CONFIG_MODE,
        description="Invert fill direction for right bar"
    )

    # Runtime controls
    enabled: bool = config_field(
        default=True,
        description="Enable ArtNet output"
    )

    white: float = config_field(
        default=0.5,
        min=0.0,
        max=1.0,
        description="Base white intensity for all pixels"
    )

    color: float = config_field(
        default=0.5,
        min=0.0,
        max=1.0,
        description="Accent color intensity"
    )


    bar: float = config_field(
        default=0.5,
        min=0.0,
        max=1.0,
        description="Bar fill level (0=empty, 1=full)"
    )


# Color RGB values (before applying intensity)
COLOR_MAP: dict[SelectedColor, tuple[int, int, int]] = {
    SelectedColor.RED:    (255, 0, 0),
    SelectedColor.BLUE:   (0, 0, 255),
    SelectedColor.YELLOW: (255, 255, 0),
}


class ArtNetLed:
    """ArtNet controller for a pair of RGBW LED bars.

    Drives two vertical LED bars with a bar-fill visualization.
    Uses threading for continuous updates at configured FPS.

    Example:
        >>> config = ArtNetLedConfig(
        ...     ip_address="192.168.1.100",
        ...     num_pixels=60,
        ...     channel_order=ChannelOrder.GRBW,
        ...     bar_right_inverted=True
        ... )
        >>> controller = ArtNetLed(config)
        >>> controller.start()
        >>>
        >>> # Runtime adjustments
        >>> config.bar = 0.5    # Half-filled bar
        >>> config.white = 0.2  # Dim white base
        >>> config.selected_color = SelectedColor.BLUE
        >>>
        >>> controller.stop()
    """

    def __init__(self, config: ArtNetLedConfig) -> None:
        self._config: ArtNetLedConfig = config
        self._lock: Lock = Lock()

        # Parse channel order
        self._channel_indices: tuple[int, int, int, int] = _parse_channel_order(config.channel_order.value)

        # Calculate DMX size: 2 bars × num_pixels × 4 channels (RGBW)
        self._channels_per_pixel: int = 4
        self._total_channels: int = 2 * config.num_pixels * self._channels_per_pixel

        # Initialize ArtNet
        self._artnet: StupidArtnet = StupidArtnet(
            config.ip_address,
            config.universe,
            self._total_channels,
            config.fps,
            True,  # even_packet_size
            True   # broadcast
        )

        # DMX buffer
        self._buffer: bytearray = bytearray(self._total_channels)

        # Threading
        self._running: bool = False
        self._update_event: Event = Event()
        self._thread: Thread | None = None
        self._dirty: bool = True  # Frame needs recalculation

        # IP validation
        self._valid_ip: bool = False
        self._validate_ip()

        # Watch config changes to trigger immediate update
        self._unwatchers: list = []
        self._setup_watchers()

        print(f"ArtNetLed: Initialized for {config.ip_address} universe {config.universe}, "
              f"{config.num_pixels} pixels/bar, order={config.channel_order}")

    def start(self) -> None:
        """Start the ArtNet output thread.

        Note: IP reachability check happens in background thread.
        """
        if self._running:
            return

        # Basic IP format validation only (non-blocking)
        if not self._validate_ip():
            raise RuntimeError(
                f"Cannot start ArtNetLed: Invalid IP address format: {self._config.ip_address}"
            )

        self._running = True
        self._thread = Thread(target=self._update_loop, daemon=True, name=f"ArtNetLed-{self._config.ip_address}")
        self._thread.start()
        if self._config.verbose:
            print(f"ArtNetLed: Starting output to {self._config.ip_address} (validating in background)...")

    def stop(self) -> None:
        """Stop the ArtNet output thread and blackout."""
        if not self._running:
            return

        self._running = False
        self._update_event.set()  # Wake up thread to exit

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Blackout using StupidArtnet's built-in method
        self._artnet.blackout()

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
        self._unwatchers.append(self._config.watch(lambda val: self._update_universe(), 'universe'))
        self._unwatchers.append(self._config.watch(lambda val: self._update_packet_size(), 'num_pixels'))
        self._unwatchers.append(self._config.watch(lambda val: self._reinit_channel_order(), 'channel_order'))

        # Blackout when disabled
        self._unwatchers.append(self._config.watch(lambda val: self._on_enabled_change(val), 'enabled'))

    def _on_enabled_change(self, enabled: bool) -> None:
        """Called when enabled state changes."""
        if not enabled:
            self._artnet.blackout()
            if self._config.verbose:
                print(f"ArtNetLed: Output disabled, LEDs blacked out")

    def _on_config_change(self) -> None:
        """Called on any config change - mark dirty and wake thread."""
        self._dirty = True
        self._update_event.set()

    def _ping_ip(self) -> bool:
        """Ping IP address to check if host is reachable.

        Returns:
            True if ping successful, False otherwise.
        """
        try:
            # Windows: ping -n 1 -w 500 (1 packet, 500ms timeout)
            result = subprocess.run(
                ['ping', '-n', '1', '-w', '500', self._config.ip_address],
                capture_output=True,
                timeout=2.0
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception) as e:
            return False

    def _validate_ip(self) -> bool:
        """Validate IP address format.

        Returns:
            True if IP format is valid, False otherwise.
        """
        try:
            # Validate IP format
            ipaddress.ip_address(self._config.ip_address)
            return True
        except ValueError:
            print(f"ArtNetLed ERROR: Invalid IP address format: {self._config.ip_address}")
            return False

    def _validate_and_reinit_artnet(self) -> None:
        """Reinitialize ArtNet connection after IP change."""
        if not self._validate_ip():
            self._valid_ip = False
            return
        self._reinit_artnet()

    def _update_packet_size(self) -> None:
        """Update packet size when num_pixels changes."""
        with self._lock:
            self._total_channels = 2 * self._config.num_pixels * self._channels_per_pixel
            self._artnet.set_packet_size(self._total_channels)
            self._artnet.clear()  # Sync internal buffer to new packet_size
            self._buffer = bytearray(self._total_channels)
            if self._config.verbose:
                print(f"ArtNetLed: Updated packet size to {self._total_channels} channels ({self._config.num_pixels} pixels/bar)")

    def _update_universe(self) -> None:
        """Update universe when it changes."""
        with self._lock:
            self._artnet.set_universe(self._config.universe)
            if self._config.verbose:
                print(f"ArtNetLed: Updated universe to {self._config.universe}")

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
                    self._artnet.blackout()
                    self._artnet.close()
                except:
                    pass

            # Recalculate DMX size
            self._total_channels = 2 * self._config.num_pixels * self._channels_per_pixel
            self._buffer = bytearray(self._total_channels)

            # Recreate ArtNet instance
            self._artnet = StupidArtnet(
                self._config.ip_address,
                self._config.universe,
                self._total_channels,
                self._config.fps,
                True,  # even_packet_size
                True   # broadcast
            )

            if self._config.verbose:
                print(f"ArtNetLed: Recreated instance for {self._config.ip_address}:{self._config.universe}, "
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
                self._channel_indices = _parse_channel_order(self._config.channel_order.value)
                if self._config.verbose:
                    print(f"ArtNetLed: Channel order updated to {self._config.channel_order.value}")
            except ValueError as e:
                print(f"ArtNetLed: Invalid channel order - {e}")

    def _set_pixel(self, pixel_index: int, r: int, g: int, b: int, w: int) -> None:
        """Set a single pixel's RGBW values in the buffer.

        Args:
            pixel_index: Absolute pixel index (0 to 2*num_pixels-1)
            r, g, b, w: Color values (0-255)
        """
        base: int = pixel_index * self._channels_per_pixel
        r_idx, g_idx, b_idx, w_idx = self._channel_indices

        self._buffer[base + r_idx] = r
        self._buffer[base + g_idx] = g
        self._buffer[base + b_idx] = b
        self._buffer[base + w_idx] = w

    def _compute_bar(self, bar_index: int, inverted: bool) -> None:
        """Compute pixel values for one bar.

        Args:
            bar_index: 0 for left bar, 1 for right bar
            inverted: Whether to flip fill direction
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

        # Pixel offset for this bar
        pixel_offset: int = bar_index * num_pixels

        for i in range(num_pixels):
            # Determine if this pixel is in the "lit" portion
            # Normal: pixels 0..lit_count-1 are lit (bottom-up)
            # Inverted: pixels (num_pixels-lit_count)..num_pixels-1 are lit (top-down)
            if inverted:
                is_lit = i >= (num_pixels - lit_count)
            else:
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

            self._set_pixel(pixel_offset + i, r, g, b, w)

    def _compute_frame(self) -> None:
        """Compute full DMX frame for both bars."""
        self._compute_bar(0, self._config.bar_left_inverted)
        self._compute_bar(1, self._config.bar_right_inverted)

    def _send_frame(self) -> None:
        """Send current buffer via ArtNet."""
        if self._config.enabled:
            self._artnet.set(self._buffer)
            self._artnet.show()

    def _update_loop(self) -> None:
        """Background thread loop for continuous updates."""
        # Validate IP reachability in background thread
        if not self._ping_ip():
            print(f"ArtNetLed WARNING: IP {self._config.ip_address} is not reachable (ping failed). "
                  "Will continue broadcasting - device may appear later.")
            self._valid_ip = False
        else:
            self._valid_ip = True
            if self._config.verbose:
                print(f"ArtNetLed: IP {self._config.ip_address} is reachable, starting output.")

        while self._running:
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
