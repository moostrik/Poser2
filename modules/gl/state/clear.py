from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, glClear, glClearColor, glClearDepth


def clear_color(r: float = 0, g: float = 0, b: float = 0, a: float = 0.0) -> None:
    """Clear the bound framebuffer's color to (r, g, b, a)."""
    glClearColor(r, g, b, a)
    glClear(GL_COLOR_BUFFER_BIT)


def clear_depth(depth: float = 1.0) -> None:
    """Clear the bound framebuffer's depth to `depth`."""
    glClearDepth(depth)
    glClear(GL_DEPTH_BUFFER_BIT)
