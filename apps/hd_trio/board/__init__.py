"""HD Trio's board: the slices only this app reads, as protocol + mixin pairs like ``modules/board``'s,
and ``RenderBoard``, the app's composition of both."""

from .sequencer_state import HasSequencerState, SequencerStateStoreMixin
from .render_board import   RenderBoard
