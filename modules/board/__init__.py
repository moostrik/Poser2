"""Board — thread-safe shared-state store for the render subsystem.

The pipeline (pose, segmentation, cameras) produces data at input FPS.
The render loop consumes it at output FPS.  The board decouples
these cadences: producers push latest state via callbacks wired in
main.py, render layers pull current snapshots on each draw call.

Each capability (frames, windows, images, …) is a protocol + mixin pair.
Apps compose a concrete ``RenderBoard`` class from only the mixins they
need.  Render layers declare the protocol slice they require (e.g.
``HasFrames``), keeping module code independent of any specific app.
"""

from .frames import                 HasFrames, FrameStoreMixin
from .windows import                HasWindows, WindowStoreMixin
from .images import                 HasCameraImages, CameraImageStoreMixin, HasCropImages, CropImageStoreMixin, \
                                    HasSegmentationImages, SegmentationImageStoreMixin
from .depth_tracklets import        HasDepthTracklets, DepthTrackletStoreMixin
from .sequence import               HasSequence, SequenceStoreMixin
from .tracklets import              HasTracklets, TrackletStoreMixin
from .composition_output import     HasCompositionOutput, CompositionOutputStoreMixin, HasCompositionDebug, \
                                    CompositionDebugStoreMixin
