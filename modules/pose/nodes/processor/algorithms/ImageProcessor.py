# Third-party imports
import cv2
import numpy as np

# Local application imports
from modules.pose.features import BBoxFeature
from modules.utils.PointsAndRects import Rect


# make undependent of BBoxFeature? and rect , then it is a proper algorithm utility

class ImageProcessor:
    """Handles image cropping and processing for pose detection"""

    def __init__(self, crop_expansion: float = 0.0, output_width: int = 192, output_height: int = 256):
        self.crop_expansion: float = crop_expansion
        self.output_width: int = output_width
        self.output_height: int = output_height
        self.aspect_ratio: float = output_width / output_height

    def process_pose_image(self, bbox: BBoxFeature, image: np.ndarray) -> np.ndarray:

        roi: Rect = bbox.to_rect()

        return self.get_cropped_image(image, roi, self.output_width, self.output_height)

    @staticmethod
    def get_crop_rect(image_width: int, image_height: int, roi: Rect, aspect_ratio: float, expansion: float = 0.0) -> Rect:
        """Calculate the crop rectangle for pose detection"""

        # Calculate the original ROI coordinates
        img_x = int(roi.x * image_width)
        img_y = int(roi.y * image_height)
        img_w = int(roi.width * image_width)
        img_h = int(roi.height * image_height)

        # Calculate dimensions that maintain aspect_ratio while covering the entire ROI
        roi_aspect: float = img_w / img_h

        if roi_aspect > aspect_ratio:
            crop_w: int = int(img_w * (1 + expansion))
            crop_h: int = int(crop_w // aspect_ratio)
        else:
            crop_h: int = int(img_h * (1 + expansion))
            crop_w: int = int(crop_h * aspect_ratio)

        # Calculate the new coordinates to center the square cutout around the original ROI
        crop_center_x: int = img_x + img_w // 2
        crop_center_y: int = img_y + img_h // 2
        crop_x: int = crop_center_x - crop_w // 2
        crop_y: int = crop_center_y - crop_h // 2

        # convert back to normalized coordinates
        norm_x: float = crop_x / image_width
        norm_y: float = crop_y / image_height
        norm_w: float = crop_w / image_width
        norm_h: float = crop_h / image_height

        return Rect(norm_x, norm_y, norm_w, norm_h)

    @staticmethod
    def get_cropped_image(image: np.ndarray, roi: Rect, output_width: int, output_height: int) -> np.ndarray:
        """Extract and resize the cropped image from the ROI"""
        image_height, image_width = image.shape[:2]
        image_channels: int = image.shape[2] if len(image.shape) > 2 else 1

        # Calculate the original ROI coordinates
        x: int = int(roi.x * image_width)
        y: int = int(roi.y * image_height)
        w: int = int(roi.width * image_width)
        h: int = int(roi.height * image_height)

        # Extract the roi without padding
        img_x: int = max(0, x)
        img_y: int = max(0, y)
        img_w: int = min(w + min(0, x), image_width - img_x)
        img_h: int = min(h + min(0, y), image_height - img_y)

        crop: np.ndarray = image[img_y:img_y + img_h, img_x:img_x + img_w]

        if image_channels == 1:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

        needs_padding = (x < 0 or y < 0 or x + w > image_width or y + h > image_height)

        if needs_padding:
            top = max(0, -y)
            bottom = max(0, y + h - image_height)
            left = max(0, -x)
            right = max(0, x + w - image_width)
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        interpolation = cv2.INTER_AREA if (output_width < w or output_height < h) else cv2.INTER_CUBIC

        return cv2.resize(crop, (output_width, output_height), interpolation=interpolation)