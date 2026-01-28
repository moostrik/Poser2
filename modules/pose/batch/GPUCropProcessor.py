# Standard library imports
from dataclasses import replace
from typing import Callable, TYPE_CHECKING
import time

import numpy as np
import cupy as cp

from modules.pose.features import BBox
from modules.pose.Frame import FrameDict
from modules.pose.batch.GPUFrame import GPUFrame, GPUFrameDict
from modules.pose.batch.cuda_image_ops import bilinear_resize_inplace, bgr_to_rgb_inplace
from modules.utils.PointsAndRects import Rect, Point2f
from modules.utils.PerformanceTimer import PerformanceTimer

if TYPE_CHECKING:
    from modules.cam.depthcam.Definitions import FrameType


# Type aliases for callbacks
GPUCropCallback = Callable[[FrameDict, GPUFrameDict], None]


class GPUCropProcessor:
    """GPU-based batch processor for cropping images based on pose bounding boxes.

    Uploads full frames to GPU once, then performs all cropping and resizing on GPU.
    Maintains previous frames for optical flow (re-cropped at current bbox location).
    Uses CuPy memory pooling for efficient GPU memory management.

    Output crops are at a fixed resolution defined at initialization. TRT inference
    classes are responsible for scaling to their specific model input size.
    """

    def __init__(
        self,
        crop_width: int = 384,
        crop_height: int = 512,
        crop_expansion: float = 0.0,
        max_poses: int = 4
    ) -> None:
        """Initialize GPU crop processor.

        Args:
            crop_width: Output crop width (default 384 for ULTRA resolution)
            crop_height: Output crop height (default 512 for ULTRA resolution)
            crop_expansion: Fractional expansion of bounding box (0.0 = no expansion)
            max_poses: Maximum number of simultaneous poses (for buffer pooling)
        """
        self._crop_width: int = crop_width
        self._crop_height: int = crop_height
        self._crop_scale: float = 1.0 + crop_expansion
        self._max_poses: int = max_poses
        self._aspect_ratio: float = crop_width / crop_height

        # Per-camera GPU frame storage
        self._gpu_images: dict[int, cp.ndarray] = {}  # cam_id -> full frame on GPU
        self._prev_gpu_images: dict[int, cp.ndarray] = {}  # cam_id -> previous frame on GPU

        # Callbacks
        self._callbacks: set[GPUCropCallback] = set()

        # Create dedicated CUDA stream for crop operations
        self._stream: cp.cuda.Stream = cp.cuda.Stream(non_blocking=True)

        # Performance timers and accumulators
        self._accumulated_upload_ms: float = 0.0
        self._upload_timer: PerformanceTimer = PerformanceTimer(
            name="GPUCrop Upload", sample_count=10000, report_interval=100, color="green", omit_init=10
        )
        self._process_timer: PerformanceTimer = PerformanceTimer(
            name="GPUCrop Process", sample_count=10000, report_interval=100, color="green", omit_init=10
        )

    def set_image(self, cam_id: int, frame_type: 'FrameType', image: np.ndarray) -> None:
        """Upload image from a specific camera to GPU. Only VIDEO frames are stored.

        Shifts current GPU image to previous before uploading new current.

        Args:
            cam_id: Camera identifier (used as tracklet ID in single-camera setups)
            frame_type: Type of frame (only VIDEO frames are processed)
            image: BGR uint8 image (H, W, 3) on CPU
        """
        from modules.cam.depthcam.Definitions import FrameType

        if frame_type != FrameType.VIDEO:
            return

        start = time.perf_counter()

        with self._stream:
            # Shift current to previous
            if cam_id in self._gpu_images:
                self._prev_gpu_images[cam_id] = self._gpu_images[cam_id]

            # Upload BGR frame to GPU (fast - contiguous), then convert to RGB on GPU
            gpu_img = cp.asarray(image)
            bgr_to_rgb_inplace(gpu_img, stream=self._stream)
            self._gpu_images[cam_id] = gpu_img

        # Sync to measure actual upload time
        # self._stream.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._accumulated_upload_ms += elapsed_ms

    def process(self, poses: FrameDict) -> None:
        """Process all poses: crop on GPU and notify callbacks.

        Crops current frame for each pose. For optical flow, also crops previous
        frame at the CURRENT bbox location (not the previous bbox).

        Allocates fresh crop buffers each frame - CuPy's memory pool handles
        efficient reuse when TRT classes release their references.

        Args:
            poses: Dictionary of tracklet_id -> Frame with bbox information
        """
        start = time.perf_counter()

        cropped_poses: FrameDict = {}
        gpu_frames: GPUFrameDict = {}

        # Clean up previous images for lost tracks
        lost_ids = set(self._prev_gpu_images.keys()) - set(poses.keys())
        for lost_id in lost_ids:
            # Let CuPy memory pool reclaim
            if lost_id in self._prev_gpu_images:
                del self._prev_gpu_images[lost_id]

        pose_count = 0

        with self._stream:
            for pose_id, pose in poses.items():
                if pose_id not in self._gpu_images:
                    continue

                if pose_count >= self._max_poses:
                    print(f"GPUCropProcessor: Exceeded max poses ({self._max_poses}), skipping {pose_id}")
                    continue

                try:
                    gpu_image = self._gpu_images[pose_id]
                    img_height, img_width = gpu_image.shape[:2]
                    bbox_rect = pose.bbox.to_rect()

                    # Calculate crop region (same logic as ImageProcessor)
                    crop_roi = self._calculate_crop_roi(bbox_rect, img_width, img_height)

                    # Allocate fresh crop buffer (CuPy pool handles reuse)
                    crop_buffer = cp.empty((self._crop_height, self._crop_width, 3), dtype=cp.uint8)
                    self._gpu_crop_resize(gpu_image, crop_roi, crop_buffer)

                    # Crop previous frame at CURRENT bbox location (for optical flow)
                    prev_crop: cp.ndarray | None = None
                    if pose_id in self._prev_gpu_images:
                        prev_buffer = cp.empty((self._crop_height, self._crop_width, 3), dtype=cp.uint8)
                        self._gpu_crop_resize(self._prev_gpu_images[pose_id], crop_roi, prev_buffer)
                        prev_crop = prev_buffer

                    # Normalize crop ROI for output
                    normalized_roi = crop_roi.scale(Point2f(1.0 / img_width, 1.0 / img_height))
                    cropped_poses[pose_id] = replace(pose, bbox=BBox.from_rect(normalized_roi))

                    gpu_frames[pose_id] = GPUFrame(
                        track_id=pose_id,
                        full_image=gpu_image,
                        crop=crop_buffer,
                        prev_crop=prev_crop
                    )

                    pose_count += 1

                except Exception as e:
                    print(f"GPUCropProcessor: Error processing pose {pose_id}: {e}")

        # Sync before callbacks access the data
        self._stream.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._process_timer.add_time(elapsed_ms)

        # Report accumulated upload time
        if self._accumulated_upload_ms > 0:
            self._upload_timer.add_time(self._accumulated_upload_ms)
            self._accumulated_upload_ms = 0.0

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(cropped_poses, gpu_frames)
            except Exception as e:
                print(f"GPUCropProcessor: Error in callback: {e}")

    def _calculate_crop_roi(self, bbox_rect: Rect, img_width: int, img_height: int) -> Rect:
        """Calculate crop region maintaining aspect ratio.

        Args:
            bbox_rect: Normalized bounding box (0-1 range)
            img_width: Source image width
            img_height: Source image height

        Returns:
            Crop region in pixel coordinates
        """
        image_rect = Rect(0.0, 0.0, float(img_width), float(img_height))

        # Apply expansion and convert to pixel coordinates
        roi = bbox_rect.zoom(self._crop_scale)
        roi = roi.affine_transform(image_rect)

        # Scale to cover ROI while maintaining output aspect ratio
        crop_roi = Rect(0, 0, self._crop_width, self._crop_height)
        crop_roi = crop_roi.aspect_fill(roi)

        return crop_roi

    def _gpu_crop_resize(self, src: cp.ndarray, roi: Rect, dst: cp.ndarray) -> None:
        """Crop and resize on GPU using CuPy.

        Extracts region from source, applies padding if needed, and resizes
        to destination buffer size using bilinear interpolation.

        Args:
            src: Source image on GPU (H, W, 3)
            roi: Crop region in pixel coordinates
            dst: Destination buffer on GPU (crop_height, crop_width, 3)
        """
        src_h, src_w = src.shape[:2]
        dst_h, dst_w = dst.shape[:2]

        # Clamp ROI to image bounds for valid region extraction
        x1 = max(0, int(roi.x))
        y1 = max(0, int(roi.y))
        x2 = min(src_w, int(roi.x + roi.width))
        y2 = min(src_h, int(roi.y + roi.height))

        # Check if ROI is completely outside image
        if x1 >= x2 or y1 >= y2:
            dst[:] = 0  # Black frame
            return

        # Extract valid region
        crop = src[y1:y2, x1:x2]

        # Handle padding if ROI extends beyond image
        pad_left = max(0, -int(roi.x))
        pad_top = max(0, -int(roi.y))
        pad_right = max(0, int(roi.x + roi.width) - src_w)
        pad_bottom = max(0, int(roi.y + roi.height) - src_h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cp.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                         mode='constant', constant_values=0)

        # Resize to destination size using bilinear interpolation
        # cupyx.scipy.ndimage.zoom is slow; use a simple approach with grid sampling
        crop_h, crop_w = crop.shape[:2]

        if crop_h == dst_h and crop_w == dst_w:
            # No resize needed
            dst[:] = crop
        else:
            # Bilinear resize using shared CUDA kernel
            bilinear_resize_inplace(crop, dst, self._stream)

    def add_callback(self, callback: GPUCropCallback) -> None:
        """Register callback to receive cropped poses and GPU frames."""
        self._callbacks.add(callback)

    def remove_callback(self, callback: GPUCropCallback) -> None:
        """Unregister callback."""
        self._callbacks.discard(callback)

    def reset(self) -> None:
        """Clear all stored GPU images. Pool buffers are retained."""
        self._gpu_images.clear()
        self._prev_gpu_images.clear()
