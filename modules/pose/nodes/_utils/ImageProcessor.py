# Third-party imports
import cv2
import numpy as np

# Local application imports
from modules.utils.PointsAndRects import Rect, Point2f


class ImageProcessor:
    """Handles image cropping and processing for pose detection"""

    def __init__(self, expansion_width: float = 0.1, expansion_height: float = 0.1, output_width: int = 192, output_height: int = 256):
        if output_width <= 0 or output_height <= 0:
            raise ValueError(f"Output dimensions must be positive, got {output_width}x{output_height}")
        self.expansion_width: float = expansion_width
        self.expansion_height: float = expansion_height
        self.output_width: int = output_width
        self.output_height: int = output_height
        self.aspect_ratio: float = output_width / output_height

    def process_pose_image(self, roi: Rect, image: np.ndarray) -> tuple[np.ndarray, Rect]:
        """Process a pose image: extract region from normalised roi and resize to configured dimensions.

        Args:
            roi: Normalized bounding box (0-1 range) - expansion should be applied before calling this
        """
        image_rect = Rect(0.0, 0.0, float(image.shape[1]), float(image.shape[0]))

        # Convert to pixel coordinates
        roi = roi.affine_transform(image_rect)

        # Determine extraction rectangle: scales to cover ROI while maintaining output aspect ratio
        crop_roi: Rect = Rect(0, 0, self.output_width, self.output_height)
        crop_roi = crop_roi.aspect_fill(roi)

        crop_image: np.ndarray = self.extract_region(image, crop_roi)

        # Auto-select interpolation based on scaling direction
        src_h, src_w = crop_image.shape[:2]
        downsample: bool = (self.output_width < src_w or self.output_height < src_h)
        interpolation: int = cv2.INTER_AREA if downsample else cv2.INTER_CUBIC

        result_image: np.ndarray = cv2.resize(crop_image, (self.output_width, self.output_height), interpolation=interpolation)
        normalized_crop_roi: Rect = crop_roi.scale(Point2f(1.0 / image_rect.width, 1.0 / image_rect.height))

        return result_image, normalized_crop_roi

    @staticmethod
    def extract_region(image: np.ndarray, pixel_roi: Rect, border_value: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Extract a region from an image with padding if needed.

        Args:
            image: Source image (H, W, C) or (H, W)
            pixel_roi: Rectangle in pixel coordinates
            border_value: RGB color for padding

        Returns:
            Cropped image with padding applied if needed
        """

        img_rect = Rect(0.0, 0.0, float(image.shape[1]), float(image.shape[0]))

        # Get valid extraction region
        valid_region: Rect = img_rect.intersect(pixel_roi)

        # Handle empty intersection
        if valid_region.is_empty:
            return np.zeros((int(pixel_roi.height), int(pixel_roi.width), 3), dtype=np.uint8)

        # Convert to RGB if not already 3-channel
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze()  # Remove channel dimension
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Extract valid pixels
        x, y = int(valid_region.x), int(valid_region.y)
        w, h = int(valid_region.width), int(valid_region.height)
        crop = image[y:y + h, x:x + w]

        # Apply padding if ROI extends beyond image
        if not img_rect.contains_rect(pixel_roi):
            left, top, right, bottom = pixel_roi.overflow(img_rect)
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)

        return crop