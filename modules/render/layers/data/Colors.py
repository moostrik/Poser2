# Individual color constants
POSE_COLOR_LEFT:    tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0) # Orange
POSE_COLOR_RIGHT:   tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0) # Cyan
POSE_COLOR_CENTER:  tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0) # White

SIMILARITY_COLOR_LOW:    tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0) # Red
SIMILARITY_COLOR_MID:    tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0) # Green
SIMILARITY_COLOR_HIGH:   tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0) # Yellow

# Color lists for different feature types
ANGLES_COLORS: list[tuple[float, float, float, float]] = [
    POSE_COLOR_LEFT,   # Orange
    POSE_COLOR_RIGHT,  # Cyan
]

MOVEMENT_COLORS: list[tuple[float, float, float, float]] = [
    POSE_COLOR_CENTER,  # White
]

SIMILARITY_COLORS: list[tuple[float, float, float, float]] = [
    SIMILARITY_COLOR_LOW,   # Red
    SIMILARITY_COLOR_MID,   # Green
    SIMILARITY_COLOR_HIGH,  # Yellow
]

BBOX_COLORS: list[tuple[float, float, float, float]] = [
    (1.0, 0.0, 0.0, 1.0),  # Red
    (0.0, 1.0, 0.0, 1.0),  # Green
    (1.0, 0.5, 0.0, 1.0),  # Orange
    (1.0, 1.0, 0.0, 1.0),  # Yellow
]