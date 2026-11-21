import logging

from OpenGL.GL import (GL_DEBUG_OUTPUT, GL_DEBUG_OUTPUT_SYNCHRONOUS, GL_DEBUG_SEVERITY_HIGH, GL_DEBUG_SEVERITY_NOTIFICATION,
                       GLDEBUGPROC, glDebugMessageCallback, glEnable)

logger = logging.getLogger(__name__)


def log_gl_message(source: int, kind: int, message_id: int, severity: int, length: int, message: bytes | None,
                   _user: object) -> None:
    """GL's debug callback, at the call that caused it: a high-severity message is an error with the Python stack
    of that call; a notification is left out."""
    if severity == GL_DEBUG_SEVERITY_NOTIFICATION:
        return
    text: str = message[:length].decode(errors='replace') if message else ''
    if severity == GL_DEBUG_SEVERITY_HIGH:
        logger.error("GL %s: %s", message_id, text, stack_info=True)
    else:
        logger.warning("GL %s: %s", message_id, text)


_callback = GLDEBUGPROC(log_gl_message)     # kept, so the C side's pointer stays valid


def enable_debug_output() -> None:
    """GL's debug output in the current context, synchronous, so a message is logged at the call that caused it.
    Each GL call then waits for its checks: only behind the window manager's `gl_debug`."""
    glEnable(GL_DEBUG_OUTPUT)
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS)
    glDebugMessageCallback(_callback, None)
