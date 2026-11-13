"""Deep Flow's board: the slices only this app reads, as protocol + mixin pairs like ``modules/board``'s,
and ``RenderBoard``, the app's composition of both."""

from .captures import       HasCaptures, CaptureStoreMixin
from .render_board import   RenderBoard
