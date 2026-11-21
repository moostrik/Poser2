"""Board: the thread-safe store the pipeline writes and the render reads, one protocol + mixin pair per slice."""

from .frames import                 HasFrames, FrameStoreMixin
from .windows import                HasWindows, WindowSource, WindowStoreMixin
from .camera_images import          HasCameraImages, CameraImageStoreMixin
from .crop_images import            HasCropImages, CropImageStoreMixin
from .segmentation_images import    HasSegmentationImages, SegmentationImageStoreMixin
from .camera_tracklets import       HasCameraTracklets, CameraTrackletStoreMixin
from .tracklets import              HasTracklets, TrackletStoreMixin
from .observations import           HasObservations, ObservationStoreMixin
from .time_series import            HasTimeSeries, TimeSeriesStoreMixin
from .inference_frames import       HasInferenceFrames, InferenceFrameStoreMixin
