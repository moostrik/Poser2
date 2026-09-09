"""StateMachine settings — the show's identity enums and configuration tree.

Pure data (fields, groups, enums), mirroring ``light/settings.py``'s role for its
package: ``StateId`` is the state vocabulary of ``data/STATES.md``, ``SyncMode`` the
INTRO → INTRO_PLAY sync condition, and ``StateMachineSettings`` the panel — telemetry
first, then the show timings, with the sync / manual / session corners as nested groups.
"""

from enum import IntEnum, auto

from modules.settings import BaseSettings, Field, Group, Widget


class SyncMode(IntEnum):
    """How many participants must be in sync for INTRO → INTRO_PLAY."""
    THREE         = 0   # at least 3 participants ≥ sync.threshold
    ALL_MINUS_ONE = auto()   # all but one
    ALL           = auto()   # everyone

    def required(self, participants: int) -> int:
        match self:
            case SyncMode.THREE:         return 3
            case SyncMode.ALL_MINUS_ONE: return max(participants - 1, 2)
            case _:                      return participants


class StateId(IntEnum):
    """The states of ``data/STATES.md`` (the source of truth), in narrative order — the
    value is what ``/global/state`` sends. OFF = 0 is the operational off (entered by
    pinning ``blackout``), followed by its wake transition. The *_INTRO / *_IDLE / *_PLAY
    entries are transitions promoted to states: their durations are the transition
    durations."""
    OFF        = 0
    OFF_IDLE   = auto()
    IDLE       = auto()
    IDLE_INTRO = auto()
    INTRO      = auto()
    INTRO_IDLE = auto()
    INTRO_PLAY = auto()
    PLAY       = auto()
    END        = auto()
    END_INTRO  = auto()
    END_IDLE   = auto()


class ManualSettings(BaseSettings):
    """The operator's manual corner: jump the show anywhere, freeze it in place."""
    select: Field[StateId] = Field(StateId.IDLE, description="State to jump to with the goto button")
    goto:   Field[bool]    = Field(False, widget=Widget.button, description="Jump to the selected state now")
    hold:   Field[bool]    = Field(False, description="Freeze transitions; the active state keeps updating")


class SyncSettings(BaseSettings):
    """The pose-sync condition (INTRO → INTRO_PLAY): its live telemetry and tunables."""
    similarity: Field[float]    = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Mean pose similarity")
    in_sync:    Field[int]      = Field(0, access=Field.READ, pinned=True, description="Participants currently at or above threshold")
    threshold:  Field[float]    = Field(0.75, min=0.0, max=1.0, step=0.01, widget=Widget.slider, description="A participant counts as in sync at this pose similarity")
    mode:       Field[SyncMode] = Field(SyncMode.THREE, description="INTRO → INTRO_PLAY: how many participants must be in sync (3 / all−1 / all)")


class SessionModeSettings(BaseSettings):
    """Session mode: timed overrides for the open-ended states (INTRO, PLAY) and no
    END wind-back — a session always concludes."""
    enabled:       Field[bool]  = Field(False, description="Session mode on: INTRO and PLAY gain timed exits; END only winds down")
    intro_seconds: Field[float] = Field(60.0,  min=5.0, max=600.0,  step=1.0, description="INTRO → INTRO_PLAY after this time")
    play_seconds:  Field[float] = Field(150.0, min=5.0, max=1200.0, step=1.0, description="PLAY → END after this time")


class StateMachineSettings(BaseSettings):
    """Configuration for the show StateMachine. Every timing is named for the state it
    times, unit in the name (seconds = wall clock, bars = playhead bars)."""

    # Blackout — the installation's big switch. Pinning it is OFF's entry door (from any
    # state, beating hold and goto); leaving OFF is a normal condition: once unpinned,
    # OffState exits to INTRO or IDLE depending on presence.
    blackout: Field[bool] = Field(False, pinned=True, description="Blackout: pin to switch the installation OFF (spinning low, strip dark, /global/state 0); unpin and OFF exits to INTRO (people present) or IDLE (empty)")

    # Telemetry (read-only) — the show at a glance
    current:      Field[StateId] = Field(StateId.OFF, access=Field.READ, description="Current show state")
    progress:     Field[float]     = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Active state progress")
    participants: Field[int]       = Field(0, access=Field.READ, pinned=True, description="Debounced participant count")

    # Transition-state durations — one per state. spin_down_seconds is shared (via the
    # root) into the wind_down layer, which runs the S9/S10 wall fade on it; those states'
    # exit is one playhead bar after the motor re-locks at LOW.
    spin_up_seconds:       Field[float] = Field(14.0, min=1.0, max=60.0,  step=0.5, description="INTRO_PLAY: spin-up transition (seconds) — hand-tuned to the physical spin-up (spin_down_seconds' mirror)", newline=True)
    spin_down_seconds:     Field[float] = Field(10.0, min=1.0, max=60.0,  step=0.5, description="END_INTRO/END_IDLE: wall fade towards the line (seconds) — hand-tuned to the physical spin-down (drives the wind_down layer)")
    off_idle_bars:         Field[float] = Field(1.0,  min=0.1, max=20.0,  step=0.1, description="OFF_IDLE: wake fade from dark, in playhead bars", newline=True)
    intro_idle_bars:       Field[float] = Field(1.0,  min=0.1, max=20.0,  step=0.1, description="INTRO_IDLE: playhead bars back to IDLE")
    end_bars:              Field[float] = Field(3.0,  min=0.5, max=20.0,  step=0.5, description="END: wind-down playhead bars (bidirectional ramp)")

    # Condition tunables
    count_hold_seconds: Field[float] = Field(1.0,  min=0.0, max=10.0, step=0.1, description="Participant-count debounce: a new count must persist this long before conditions see it", newline=True)

    sync:    Group[SyncSettings]        = Group(SyncSettings)
    manual:  Group[ManualSettings]      = Group(ManualSettings)
    session: Group[SessionModeSettings] = Group(SessionModeSettings)
