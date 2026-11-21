"""GL's state per context: the context, the blend state, the clear."""

from . import blend_state
from .context import Context, context, draw_quad
from .clear import clear_color
