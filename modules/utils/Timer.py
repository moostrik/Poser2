"""Timer class with config-based control and frame-accurate timing."""

import time
import threading
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, Set

from modules.ConfigBase import ConfigBase, config_field


class TimerState(IntEnum):
    """Timer state enum."""
    IDLE =          0
    RUNNING =       auto()
    INTERMEZZO =    auto()


@dataclass
class TimerConfig(ConfigBase):
    """Configuration for Timer with GUI-integrated controls."""

    fps: float = config_field(60.0, min=1.0, max=240.0, fixed=True, description="Update rate in frames per second")
    run: bool = config_field(False, description="Run the timer")
    duration: float = config_field(10.0, min=0.1, max=600.0, description="Timer duration in seconds")
    intermezzo: float = config_field(0.0, min=0.0, max=60.0, description="Wait duration before going idle")
    auto: bool = config_field(False, repr=False, description="Automatically restart after intermezzo")
    verbose: bool = config_field(False, repr=False, description="Print timer state and time updates")


class Timer(threading.Thread):
    """Threaded timer with config-based control and callbacks.

    Outputs elapsed time at configured FPS via callbacks.

    States:
    - IDLE: Timer not running
    - RUNNING: Timer counting down
    - INTERMEZZO: Waiting before going idle

    Example:
        >>> config = TimerConfig(duration=10.0, fps=30.0)
        >>> timer = Timer(config)
        >>> timer.add_time_callback(lambda t: print(f"Elapsed: {t:.2f}s"))
        >>> timer.add_state_callback(lambda state: print(f"State: {state.name}"))
        >>> timer.start()  # Start thread
        >>> config.run = True  # Begin timing
        >>> # Later...
        >>> config.run = False  # Stop timing
        >>> timer.stop()  # Stop thread
    """

    def __init__(self, config: TimerConfig | None = None) -> None:
        """Initialize timer with configuration.

        Args:
            config: Timer configuration. Creates default if None.
        """
        super().__init__(daemon=True, name="Timer")

        self.config: TimerConfig = config or TimerConfig()

        # Callback management
        self._time_callbacks: Set[Callable[[float], None]] = set()
        self._state_callbacks: Set[Callable[[TimerState], None]] = set()
        self._callback_lock = threading.Lock()

        # Timer state
        self._state: TimerState = TimerState.IDLE
        self._start_timestamp: float = 0.0
        self._intermezzo_start: float = 0.0
        self._stop_event = threading.Event()
        self._updating_run = False  # Flag to prevent circular updates

        # Watch config changes
        self._setup_watchers()

    def _setup_watchers(self) -> None:
        """Setup config watcher for run control."""
        self.config.watch(self._on_run_change, 'run')

    def _on_run_change(self, value: bool) -> None:
        """Handle run config change."""
        # Skip if we're updating internally
        if self._updating_run:
            return

        if value and self._state == TimerState.IDLE:
            # Start timer
            self._start_timestamp = time.time()
            self._set_state(TimerState.RUNNING)
        elif not value and self._state == TimerState.RUNNING:
            # Stop timer - enter intermezzo
            self._intermezzo_start = time.time()
            self._set_state(TimerState.INTERMEZZO)

    # Callback management - Time callbacks
    def add_time_callback(self, callback: Callable[[float], None]) -> None:
        """Register callback for time updates.

        Args:
            callback: Function to call with elapsed time in seconds.
        """
        with self._callback_lock:
            self._time_callbacks.add(callback)

    def remove_time_callback(self, callback: Callable[[float], None]) -> None:
        """Unregister time callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        with self._callback_lock:
            self._time_callbacks.discard(callback)

    def _notify_time_callbacks(self, elapsed: float) -> None:
        """Emit time callbacks with elapsed time."""
        with self._callback_lock:
            callbacks_copy = list(self._time_callbacks)

        for callback in callbacks_copy:
            try:
                callback(elapsed)
            except Exception as e:
                print(f"Timer: Error in time callback: {e}")

    # Callback management - State callbacks
    def add_state_callback(self, callback: Callable[[TimerState], None]) -> None:
        """Register callback for timer state changes.

        Args:
            callback: Function to call with new TimerState when state changes.
        """
        with self._callback_lock:
            self._state_callbacks.add(callback)

    def remove_state_callback(self, callback: Callable[[TimerState], None]) -> None:
        """Unregister state callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        with self._callback_lock:
            self._state_callbacks.discard(callback)

    def _set_state(self, new_state: TimerState) -> None:
        """Set timer state and notify callbacks."""
        if self._state != new_state:
            self._state = new_state

            # Sync run field with state (prevent circular updates)
            self._updating_run = True
            self.config.run = (new_state == TimerState.RUNNING)
            self._updating_run = False

            self._notify_state_callbacks(new_state)

    def _notify_state_callbacks(self, state: TimerState) -> None:
        """Emit state callbacks."""
        with self._callback_lock:
            callbacks_copy = list(self._state_callbacks)

        for callback in callbacks_copy:
            try:
                callback(state)
            except Exception as e:
                print(f"Timer: Error in state callback: {e}")

    def run(self) -> None:
        """Main timer loop with frame-accurate timing."""
        interval = 1.0 / self.config.fps
        next_time = time.time()

        while not self._stop_event.is_set():
            # Calculate current elapsed time for display
            if self._state == TimerState.RUNNING:
                elapsed = time.time() - self._start_timestamp
            elif self._state == TimerState.INTERMEZZO:
                elapsed = time.time() - self._intermezzo_start
            else:
                elapsed = 0.0

            # Print state and time every update (if verbose)
            if self.config.verbose:
                print(f"Timer: {self._state.name} | Time: {elapsed:.2f}s")

            # Handle intermezzo state
            if self._state == TimerState.INTERMEZZO:
                self._notify_time_callbacks(elapsed)

                if elapsed >= self.config.intermezzo:
                    # Intermezzo complete
                    if self.config.auto:
                        # Auto-restart: go back to running
                        self._start_timestamp = time.time()
                        self._set_state(TimerState.RUNNING)
                    else:
                        # No auto-restart: go to idle
                        self._set_state(TimerState.IDLE)

            # Handle running timer
            elif self._state == TimerState.RUNNING:
                self._notify_time_callbacks(elapsed)

                # Check if duration reached
                if elapsed >= self.config.duration:
                    # Duration reached - enter intermezzo
                    self._intermezzo_start = time.time()
                    self._set_state(TimerState.INTERMEZZO)

            # Frame-accurate timing
            next_time += interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Fell behind, reset timing
                next_time = time.time()

    def start_timer(self) -> None:
        """Start the timer countdown.

        Convenience method that sets config.run = True.
        """
        self.config.run = True

    def stop_timer(self) -> None:
        """Stop the timer countdown.

        Convenience method that sets config.run = False.
        """
        self.config.run = False

    def stop(self) -> None:
        """Stop the timer thread and clear callbacks."""
        self._stop_event.set()

        # Join with timeout
        if self.is_alive():
            self.join(timeout=1.0)
            if self.is_alive():
                print(f"WARNING: Timer thread did not exit cleanly within timeout")

        # Clear all callbacks
        with self._callback_lock:
            self._time_callbacks.clear()
            self._state_callbacks.clear()
