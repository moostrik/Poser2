#TODO: torch.nograd() where applicable

# Standard library imports
from dataclasses import replace
import time

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F

# Local application imports
from modules.pose.features import BBox
from modules.pose.Frame import FrameDict
from modules.pose.batch.GPUFrame import GPUFrame, GPUFrameDict, GPUFrameCallback
from modules.utils.PointsAndRects import Rect, Point2f
from modules.utils.PerformanceTimer import PerformanceTimer

from modules.cam.depthcam.Definitions import FrameType
from modules.utils.HotReloadMethods import HotReloadMethods


class GPUCropProcessorConfig:
    """Configuration for GPU-based image cropping."""

    def __init__(self, expansion: float = 0.0, output_width: int = 384, output_height: int = 512, max_poses: int = 4, enable_prev_crop: bool = True) -> None:
        self.crop_scale: float = 1.0 + expansion
        self.output_width: int = output_width
        self.output_height: int = output_height
        self.max_poses: int = max_poses
        self.enable_prev_crop: bool = enable_prev_crop  # Enable previous frame crops for optical flow
        self.verbose: bool = True


class GPUCropProcessor:
    """GPU-based batch processor for cropping images based on pose bounding boxes.

    Uploads full frames to GPU once, then performs all cropping and resizing on GPU.
    Maintains previous frames for optical flow (re-cropped at current bbox location).
    Uses PyTorch for all GPU operations with F.grid_sample for efficient crop+resize.

    Output crops are float16 RGB normalized to [0,1] at a fixed resolution.
    Full frames are stored as float16 BGR [0,1] (flip happens on crop for efficiency).
    """

    def __init__(self, config: GPUCropProcessorConfig) -> None:
        """Initialize GPU crop processor.

        Args:
            config: Configuration for GPU crop processor
        """
        self._config: GPUCropProcessorConfig = config
        self._crop_width: int = config.output_width
        self._crop_height: int = config.output_height
        self._crop_scale: float = config.crop_scale
        self._max_poses: int = config.max_poses
        self._aspect_ratio: float = config.output_width / config.output_height
        self._enable_prev_crop: bool = config.enable_prev_crop

        # Per-camera GPU frame storage (float32 BGR [0,1])
        self._gpu_images: dict[int, torch.Tensor] = {}  # cam_id -> full frame on GPU
        self._prev_gpu_images: dict[int, torch.Tensor] = {}  # cam_id -> previous frame on GPU

        # Callbacks
        self._callbacks: set[GPUFrameCallback] = set()

        # Create dedicated CUDA stream for crop operations
        self._stream: torch.cuda.Stream = torch.cuda.Stream()

        # Performance timers and accumulators
        self._accumulated_upload_ms: float = 0.0
        self._process_timer: PerformanceTimer = PerformanceTimer(name="GPU Image Upload  ", sample_count=200, report_interval=100, color="green", omit_init=25)

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def set_image(self, cam_id: int, frame_type: 'FrameType', image: np.ndarray) -> None:
        """Upload image from a specific camera to GPU. Only VIDEO frames are stored.

        Shifts current GPU image to previous before uploading new current.
        Stores as float32 RGB [0,1] - BGR→RGB conversion done once at upload.

        Args:
            cam_id: Camera identifier (used as tracklet ID in single-camera setups)
            frame_type: Type of frame (only VIDEO frames are processed)
            image: BGR uint8 image (H, W, 3) on CPU
        """
        from modules.cam.depthcam.Definitions import FrameType

        if frame_type != FrameType.VIDEO:
            return

        start = time.perf_counter()

        with torch.cuda.stream(self._stream):
            # Shift current to previous
            if cam_id in self._gpu_images:
                self._prev_gpu_images[cam_id] = self._gpu_images[cam_id]

            # Upload BGR frame to GPU, fuse all ops for better memory access
            gpu_img = torch.from_numpy(image).cuda(non_blocking=True)
            # Fused: convert dtype, normalize, flip - all in one pass
            self._gpu_images[cam_id] = gpu_img.flip(2).to(dtype=torch.float16).mul_(1.0/255.0)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._accumulated_upload_ms += elapsed_ms

    def process(self, poses: FrameDict) -> None:
        """Process all poses: crop on GPU and notify callbacks.

        Crops current frame for each pose. For optical flow, also crops previous
        frame at the CURRENT bbox location (not the previous bbox).

        Output crops are float32 RGB [0,1] in HWC format.

        Args:
            poses: Dictionary of tracklet_id -> Frame with bbox information
        """
        start = time.perf_counter()

        cropped_poses: FrameDict = {}
        gpu_frames: GPUFrameDict = {}

        # Clean up previous images for lost tracks
        lost_ids = set(self._prev_gpu_images.keys()) - set(poses.keys())
        for lost_id in lost_ids:
            if lost_id in self._prev_gpu_images:
                del self._prev_gpu_images[lost_id]

        pose_count = 0

        with torch.cuda.stream(self._stream):
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

                    # Crop and resize using grid_sample, output is float32 RGB [0,1] HWC
                    crop_tensor = self._gpu_crop_resize(gpu_image, crop_roi, img_width, img_height)

                    # Crop previous frame at CURRENT bbox location (for optical flow)
                    prev_crop: torch.Tensor | None = None
                    if self._enable_prev_crop and pose_id in self._prev_gpu_images:
                        prev_img = self._prev_gpu_images[pose_id]
                        prev_h, prev_w = prev_img.shape[:2]
                        prev_crop = self._gpu_crop_resize(prev_img, crop_roi, prev_w, prev_h)

                    # Normalize crop ROI for output
                    normalized_roi = crop_roi.scale(Point2f(1.0 / img_width, 1.0 / img_height))
                    cropped_poses[pose_id] = replace(pose, bbox=BBox.from_rect(normalized_roi))

                    # Build GPUFrame with PyTorch tensors
                    gpu_frames[pose_id] = GPUFrame(
                        track_id=pose_id,
                        full_image=gpu_image,  # float32 BGR [0,1] HWC
                        crop=crop_tensor,       # float32 RGB [0,1] HWC
                        prev_crop=prev_crop     # float32 RGB [0,1] HWC or None
                    )

                    pose_count += 1

                except Exception as e:
                    print(f"GPUCropProcessor: Error processing pose {pose_id}: {e}")

        # Sync before callbacks access the data
        self._stream.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        elapsed_ms += self._accumulated_upload_ms
        self._process_timer.add_time(elapsed_ms, report=self._config.verbose)
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

    def _gpu_crop_resize(self, src: torch.Tensor, roi: Rect, src_w: int, src_h: int) -> torch.Tensor:
        """Crop and resize on GPU using slice + F.interpolate + F.pad.

        Extracts ROI region, pads if needed, resizes using bilinear interpolation.

        Args:
            src: Source image on GPU (H, W, 3) float32 RGB [0,1]
            roi: Crop region in pixel coordinates
            src_w: Source image width
            src_h: Source image height

        Returns:
            Cropped and resized image (crop_height, crop_width, 3) float32 RGB [0,1]
        """
        # Clamp ROI to image bounds for valid region extraction
        x1 = max(0, int(roi.x))
        y1 = max(0, int(roi.y))
        x2 = min(src_w, int(roi.x + roi.width))
        y2 = min(src_h, int(roi.y + roi.height))

        # Check if ROI is completely outside image
        if x1 >= x2 or y1 >= y2:
            return torch.zeros((self._crop_height, self._crop_width, 3), device='cuda', dtype=torch.float16)

        # Extract valid region: HWC slice
        crop = src[y1:y2, x1:x2, :]

        # Calculate padding amounts
        pad_left = max(0, -int(roi.x))
        pad_right = max(0, int(roi.x + roi.width) - src_w)
        pad_top = max(0, -int(roi.y))
        pad_bottom = max(0, int(roi.y + roi.height) - src_h)

        # Convert to CHW for F.pad and F.interpolate
        crop_chw = crop.permute(2, 0, 1)  # (3, H, W)

        # Apply padding if needed (F.pad uses reverse order: left, right, top, bottom)
        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            crop_chw = F.pad(crop_chw, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        # Resize using nearest neighbor (fast, since TRT will resize again with bilinear)
        crop_h, crop_w = crop_chw.shape[1], crop_chw.shape[2]
        if crop_h != self._crop_height or crop_w != self._crop_width:
            crop_chw = F.interpolate(
                crop_chw.unsqueeze(0),
                size=(self._crop_height, self._crop_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Convert CHW -> HWC (already RGB)
        result = crop_chw.permute(1, 2, 0)

        return result

    def add_callback(self, callback: GPUFrameCallback) -> None:
        """Register callback to receive cropped poses and GPU frames."""
        self._callbacks.add(callback)

    def remove_callback(self, callback: GPUFrameCallback) -> None:
        """Unregister callback."""
        self._callbacks.discard(callback)

    def reset(self) -> None:
        """Clear all stored GPU images."""
        self._gpu_images.clear()
        self._prev_gpu_images.clear()
