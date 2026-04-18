"""Blackboard — protocols and thread-safe mixins for shared runtime data."""

from modules.blackboard.frames import HasFrames, FrameStoreMixin
from modules.blackboard.windows import HasWindows, WindowStoreMixin
from modules.blackboard.images import HasImages, ImageStoreMixin
from modules.blackboard.depth_tracklets import HasDepthTracklets, DepthTrackletStoreMixin
from modules.blackboard.sequence import HasSequence, SequenceStoreMixin
