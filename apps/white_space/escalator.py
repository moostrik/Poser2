"""GhostEscalator — counts the ghosts the installation has ever made and, once that cumulative total
reaches ``high_threshold``, escalates the motor from LOW to HIGH one time.

It sits *apart from* the Ghoster: the Ghoster only announces each active ghost it commits (via
``add_new_ghost_callback``); this component owns the running total and the motor-mode decision. The switch
**latches** — it flips LOW→HIGH the first time the total reaches the threshold while the motor is in LOW,
then holds (an operator may change the mode afterward; it will not re-assert). The ``reset`` button zeroes
the total, returns the motor to LOW, and re-arms the latch.

Writes ``settings.light.motor.mode`` directly — the same ``MotorSettings`` the ``MotorController`` reads
each tick, so the change reaches the real motor. (While ``simulate != OFF`` the sim selector overrides the
commanded mode, so escalation only shows on real hardware.)
"""

from __future__ import annotations

import logging
from threading import Lock

from modules.pose.frame import Frame
from modules.settings import BaseSettings, Field, Widget

from .light.motor import MotorMode, MotorSettings

logger = logging.getLogger(__name__)


class GhostEscalatorSettings(BaseSettings):
    """Configuration for ``GhostEscalator``."""
    enabled:        Field[bool] = Field(True, description="Count ghosts and escalate the motor at the threshold")
    high_threshold: Field[int]  = Field(20, min=1, max=200, step=1, description="Ghosts ever made before the motor escalates LOW → HIGH", newline=True)
    total_ghosts:   Field[int]  = Field(0, access=Field.READ, description="Ghosts made so far (cumulative)")
    reset:          Field[bool] = Field(False, widget=Widget.button, description="Zero the count, re-arm, and return the motor to LOW")


class GhostEscalator:
    """Cumulative ghost counter that latches the motor LOW→HIGH at ``high_threshold``; see the module docstring."""

    def __init__(self, settings: GhostEscalatorSettings, motor: MotorSettings) -> None:
        self._settings = settings
        self._motor = motor
        self._lock = Lock()
        self._total = 0            # ghosts ever made (since the last reset)
        self._latched = False      # True once the one LOW→HIGH escalation has fired
        self._settings.bind(GhostEscalatorSettings.reset, self._on_reset)  # type: ignore[arg-type]

    def stop(self) -> None:
        """Teardown — unbind the reset button (no thread of its own)."""
        self._settings.unbind(GhostEscalatorSettings.reset, self._on_reset)  # type: ignore[arg-type]

    def on_new_ghost(self, _ghost: Frame) -> None:
        """A ghost was made — bump the total and, if the threshold is reached while the motor is in LOW,
        escalate to HIGH once. Subscribe to the Ghoster's ``add_new_ghost_callback``."""
        with self._lock:
            self._total += 1
            self._settings.total_ghosts = self._total
            if (self._settings.enabled and not self._latched
                    and self._total >= self._settings.high_threshold
                    and self._motor.mode == MotorMode.LOW):
                self._motor.mode = MotorMode.HIGH
                self._latched = True
                logger.info("GhostEscalator: %d ghosts made → motor LOW → HIGH", self._total)

    def _on_reset(self, _=None) -> None:
        """Zero the count, re-arm the latch, and return the motor to LOW (only if we left it HIGH)."""
        with self._lock:
            self._total = 0
            self._settings.total_ghosts = 0
            self._latched = False
            if self._motor.mode == MotorMode.HIGH:
                self._motor.mode = MotorMode.LOW
        logger.info("GhostEscalator: reset — count cleared, motor returned to LOW")
