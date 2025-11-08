"""Type aliases for pose callbacks."""

from typing import Callable

from modules.pose.Pose import Pose, PoseDict

# Type aliases
PoseCallback = Callable[[Pose], None]
PoseDictCallback = Callable[[PoseDict], None]