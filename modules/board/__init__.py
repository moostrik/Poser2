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
from .groups import                 HasGroups, GroupStoreMixin
from .windows import                HasWindows, WindowSource, WindowStoreMixin
from .camera_images import          HasCameraImages, CameraImageStoreMixin
from .camera_depth import           HasCameraDepth, CameraDepthStoreMixin
from .crop_images import            HasCropImages, CropImageStoreMixin
from .segmentation_images import    HasSegmentationImages, SegmentationImageStoreMixin
from .camera_tracklets import       HasCameraTracklets, CameraTrackletStoreMixin
from .sequencer_state import        HasSequencerState, SequencerStateStoreMixin
from .tracklets import              HasTracklets, TrackletStoreMixin
from .observations import           HasObservations, ObservationStoreMixin
