"""A camera recording's streams: the writer, the reader and the depth frame's encoding."""

from .depth_frame import DEPTH_CODEC, DEPTH_HARDWARE
from .stream_writer import StreamWriter
from .stream_reader import StreamReader, StreamFrameCallback, FrameSchedule
