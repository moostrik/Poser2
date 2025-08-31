import cv2
import numpy as np
from modules.tracker.Tracklet import Tracklet, Rect


class PoseImageProcessor:
    """Handles image cropping and processing for pose detection"""

    def __init__(self, crop_expansion: float = 0.0, output_width: int = 192, output_height: int = 256):
        self.crop_expansion = crop_expansion
        self.output_width = output_width
        self.output_height = output_height

    def process_pose_image(self, tracklet: Tracklet, image: np.ndarray) -> tuple[np.ndarray, Rect]:
        """
        Process and return the pose image and crop rectangle for a pose.
        Returns tuple of (cropped_image, crop_rect).
        """
        h, w = image.shape[:2]
        roi = self.get_crop_rect(w, h, tracklet.roi, self.crop_expansion)
        cropped_image = self.get_cropped_image(image, roi, self.output_width, self.output_height)
        return cropped_image, roi

    @staticmethod
    def get_crop_rect(image_width: int, image_height: int, roi: Rect, expansion: float = 0.0) -> Rect:
        """Calculate the crop rectangle for pose detection"""
        # Calculate the original ROI coordinates
        img_x = int(roi.x * image_width)
        img_y = int(roi.y * image_height)
        img_w = int(roi.width * image_width)
        img_h = int(roi.height * image_height)

        # Determine the size of the square cutout based on the longest side of the ROI
        img_wh: int = max(img_w, img_h)
        img_wh += int(img_wh * expansion)

        # Calculate the new coordinates to center the square cutout around the original ROI
        crop_center_x: int = img_x + img_w // 2
        crop_center_y: int = img_y + img_h // 2
        crop_x: int = crop_center_x - img_wh // 2
        crop_y: int = crop_center_y - img_wh // 2
        crop_w: int = img_wh
        crop_h: int = img_wh

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
        image_channels = image.shape[2] if len(image.shape) > 2 else 1

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

        # Apply padding if the roi is outside the image bounds
        left_padding: int = -min(0, x)
        top_padding: int = -min(0, y)
        right_padding: int = max(0, x + w - image_width)
        bottom_padding: int = max(0, y + h - image_height)

        if left_padding + right_padding + top_padding + bottom_padding > 0:
            crop = cv2.copyMakeBorder(crop, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize the cutout to the desired size
        return cv2.resize(crop, (output_width, output_height), interpolation=cv2.INTER_AREA)