"""The pose monitor: the pose on the crop, the camera's frame or the view, and its features over it."""

from typing import Any

from modules import render
from modules.gl import Fbo, Texture, clear_color
from ....settings import PoseFeatures, PoseFrame, PoseMonitorSettings


class PoseMonitorLayer(render.LayerBase):
    """The pose of camera `cam_id`'s player as `settings.frame` says: on the crop with its image
    (`PoseCompositor`), on the camera's frame with its boxes (`TrackerCompositor`, no image), or on the view
    (`ViewPoseLayer`); `settings.features` the features over it. Without an image it lies over any layer of its
    frame."""

    def __init__(self, cam_id: int, board: Any, cam_image: render.LayerBase, cam_crop: render.CropImageLayer,
                 view: render.PosePlacement, settings: PoseMonitorSettings, color_settings: render.ColorSettings) -> None:
        self.settings: PoseMonitorSettings = settings
        self._frames: dict[PoseFrame, render.LayerBase] = {
            PoseFrame.CROP:   render.PoseCompositor(cam_id, board, cam_image.texture, cam_crop, settings.crop, color_settings),
            PoseFrame.CAMERA: render.TrackerCompositor(cam_id, board, cam_image.texture, settings.camera, color_settings),
            PoseFrame.VIEW:   render.ViewPoseLayer(cam_id, view, settings.view, color_settings),
        }
        self._features: dict[PoseFeatures, render.LayerBase] = {
            PoseFeatures.FRAME:  render.FeatureFrameLayer(cam_id, board, settings.data, color_settings),
            PoseFeatures.WINDOW: render.FeatureWindowLayer(cam_id, board, settings.data, color_settings),
        }
        self._fbo: Fbo = Fbo()

    @property
    def texture(self) -> Texture:
        return self._fbo

    @property
    def frame(self) -> PoseFrame:
        """Where it draws the pose now."""
        return PoseFrame(self.settings.frame)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        for layer in (*self._frames.values(), *self._features.values()):
            layer.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        for layer in (*self._frames.values(), *self._features.values()):
            layer.deallocate()

    def update(self) -> None:
        shown: list[render.LayerBase] = [self._frames[self.frame]]
        features: PoseFeatures = PoseFeatures(self.settings.features)
        if features != PoseFeatures.NONE:
            shown.append(self._features[features])
        for layer in shown:
            layer.update()
        self._fbo.begin()
        clear_color()
        for layer in shown:
            layer.draw()
        self._fbo.end()
