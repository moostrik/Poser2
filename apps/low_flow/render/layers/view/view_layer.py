"""The view stage: the flow's view of a frame, the stage before the motion.

`ViewInput` is the frame as the app hands it each tick: its color image on the camera image's view (taken into GL by
the capture's source layer), its capture time, its player's mask from the segmentation and its player's pose.
`ViewOutput` is what the color stage and the motion read. The view cuts a fixed window from the camera image
(`ViewCrop`), so the motion measures the player, not a moving window; `ViewMask` places the mask at the box its crop
was cut at, in the same window, so the image and the mask of one capture lie on each other. The view is made again
when a new capture comes in, told by its capture time, or when a setting it is made with changes (`made`).
"""

# Standard library imports
from dataclasses import dataclass
from typing import Annotated

# Third-party imports
import numpy as np

# Local application imports
from modules.flow import HasPorts, Port, TextureKind, fixed_window, place_points
from modules.gl import BlendState, Texture
from modules.inference import segmentation
from modules.pose import BBox, Frame, Points2D
from modules.render import StageLayer, StageOutput
from modules.utils import Rect
from ....settings import ViewLayerSettings
from .view_crop import ViewCrop
from .view_mask import ViewMask


@dataclass
class ViewInput:
    """One camera's frame for the view stage: `image` its color on the camera image's view; `timestamp` its capture
    time, the capture's identity (None: no frame yet); `mask` its player's segmentation, or None without a player;
    `pose` its player's, whose points the view places in its crop, or None."""
    image: Annotated[Texture, TextureKind.IMAGE]
    timestamp: float | None
    mask: segmentation.Image | None = None
    pose: Frame | None = None


@dataclass
class ViewOutput(StageOutput):
    """The view's textures, at the view's size: `image` the cropped color, `mask` where the player is."""
    image: Annotated[Texture, TextureKind.IMAGE]
    mask: Annotated[Texture, TextureKind.SCALAR]


class ViewLayer(StageLayer[ViewInput, ViewOutput]):
    """The view stage's layer. Places the pose's points in the crop (`crop_pose_points`, `lost`) for the pose drawn
    over the view. Its ports are its parts' textures under their names, the input's and the outputs. Draws
    nothing."""

    def __init__(self, settings: ViewLayerSettings) -> None:
        super().__init__(settings)
        self.settings: ViewLayerSettings = settings

        self._crop: ViewCrop = ViewCrop(settings.crop)
        self._mask: ViewMask = ViewMask()

        self._window: Rect = Rect(0.0, 0.0, 1.0, 1.0)  # the window, uv on the camera image, rows down
        self._made: tuple | None = None                 # the frame and the settings the view was made from
        self._timestamp: float | None = None
        self._pose_points: Points2D | None = None

    # ========== What it hands out ==========

    @property
    def output(self) -> ViewOutput:
        return ViewOutput(self._crop.image, self._mask.mask)

    def _components(self) -> dict[str, HasPorts]:
        return {'crop': self._crop, 'mask': self._mask}

    @property
    def texture(self) -> Texture:
        """The view's image, the shape the view's monitor is drawn in."""
        return self._crop.image

    @property
    def timestamp(self) -> float | None:
        """The capture time of the capture the view holds, None before the first."""
        return self._timestamp

    @property
    def made(self) -> tuple:
        """The capture and the settings the view was made from, so a stage after it makes its own again when they
        change; empty before the first."""
        return self._made or ()

    @property
    def lost(self) -> bool:
        """Whether the view holds no pose: the pose drawn over it is cleared."""
        return self._pose_points is None

    @property
    def crop_pose_points(self) -> Points2D | None:
        """The frame's pose's points in the view, uv with rows down; None without a pose."""
        return self._pose_points

    # ========== Lifecycle ==========

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width is not None and height is not None
        self._crop.allocate(width, height)
        self._mask.allocate(width, height)
        self._made = None

    def deallocate(self) -> None:
        self._crop.deallocate()
        self._mask.deallocate()
        super().deallocate()

    def reset(self) -> None:
        self._crop.clear()
        self._mask.clear()
        self._made = None
        self._timestamp = None
        self._pose_points = None

    def draw(self) -> None:
        """A stage draws nothing; a monitor draws its textures."""

    # ========== Processing ==========

    def _update(self) -> None:
        input: ViewInput | None = self._input
        if input is None:
            return
        if input.timestamp is None:
            if self._timestamp is not None:
                self.reset()
            return
        image: Texture = input.image
        view: Texture = self._crop.image
        if not image.allocated or not view.allocated:
            return
        s: ViewLayerSettings = self.settings
        made: tuple = (input.timestamp, image.width, image.height, view.width, view.height, self._crop.made_with)
        if made == self._made:
            return
        self._made = made
        self._timestamp = input.timestamp

        self._window = fixed_window(s.crop.center_x, s.crop.center_y, s.crop.height, view.width / view.height, image.width / image.height)

        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)
        self._crop.crop(image, self._window)
        BlendState.pop()
        self._mask.place(input.mask, self._window)

        self._pose_points = self._place_pose(input.pose)

    def _place_pose(self, pose: Frame | None) -> Points2D | None:
        if pose is None:
            return None
        bbox: Rect = pose[BBox].to_rect()
        points: Points2D = pose[Points2D]
        x, y = points.get_xy_arrays()
        view_x, view_y = place_points(np.asarray(x), np.asarray(y), bbox, self._window)
        return Points2D.from_xy_arrays(view_x, view_y, points.scores)
