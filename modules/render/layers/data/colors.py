"""Centralized color definitions for visualization layers"""

# Pose anatomical colors (left/right/center joints)
POSE_COLOR_LEFT:   tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0)  # Orange
POSE_COLOR_RIGHT:  tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)  # Cyan
POSE_COLOR_CENTER: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # White

# Default feature colors
DEFAULT_COLORS: list[tuple[float, float, float, float]] = [
    POSE_COLOR_LEFT,   # Orange
    POSE_COLOR_RIGHT,  # Cyan
]
