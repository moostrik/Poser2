"""Centralized color definitions for visualization layers"""

# Track colors for person identification (red, blue, yellow - matched perceived brightness Y=0.33)
TRACK_COLORS: list[tuple[float, float, float, float]] = [
    (1.0, 0.15, 0.15, 1.0),    # Red    Y = 0.33
    (0.28, 0.28, 1.0, 1.0),    # Blue   Y = 0.33
    (0.36, 0.36, 0.0, 1.0),    # Yellow Y = 0.33
]

# Pose anatomical colors (left/right/center joints)
POSE_COLOR_LEFT:   tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0)  # Orange
POSE_COLOR_RIGHT:  tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)  # Cyan
POSE_COLOR_CENTER: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # White

HISTORY_COLOR: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)  # Grey

# Default feature colors
DEFAULT_COLORS: list[tuple[float, float, float, float]] = [
    POSE_COLOR_LEFT,   # Orange
    POSE_COLOR_RIGHT,  # Cyan
]

# BBox feature colors
BBOX_COLORS: list[tuple[float, float, float, float]] = [
    (1.0, 0.0, 0.0, 1.0),  # Red
    (0.0, 1.0, 0.0, 1.0),  # Green
    (1.0, 0.5, 0.0, 1.0),  # Orange
    (1.0, 1.0, 0.0, 1.0),  # Yellow
]
