from OpenGL.GL import (
    GL_MODELVIEW, GL_PROJECTION, GL_VIEWPORT,
    glGetIntegerv, glLoadIdentity, glMatrixMode, glOrtho, glPopMatrix, glPushMatrix, glViewport
)


_viewport_stack: list = []


def set_view(width: int, height: int) -> None:
    """
    Set orthographic projection with top-left origin (0,0) and Y-axis pointing down.

    Args:
        width: Viewport width in pixels
        height: Viewport height in pixels
    """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glViewport(0, 0, width, height)


def push_view() -> None:
    """
    Save current view state (projection matrix, modelview matrix, and viewport).
    Must be paired with pop_view() to restore state.
    """
    viewport = glGetIntegerv(GL_VIEWPORT)
    _viewport_stack.append(viewport)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()


def pop_view() -> None:
    """
    Restore previously saved view state.
    Must be paired with push_view().
    """
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    if _viewport_stack:
        viewport = _viewport_stack.pop()
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
