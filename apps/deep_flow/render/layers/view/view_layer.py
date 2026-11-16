"""The view stage: the flow's view of a capture, the stage before the motion.

`ViewInput` is the capture as the app hands it each tick: its color image and its depth on the camera image's view
(the uploader's, taken into GL by the capture's source layers), its capture time and the pose of its player.
`ViewOutput` is what the motion and the bridge read. The view runs its steps in order, each once, each a part with
its own textures: `ViewCrop`, the box in 3D and who (the image inside its sides, and the camera's depth inside the
box on the depth's own pixels); `ViewMask`, where (that depth's outline moved onto the image's by the guided filter,
He 2010, the one outline); `ViewDepth`, how far (the depth upscaled to the view once where it is the player's and the
mask is, then filled across the mask by push-pull, so the mask and the depth agree); and `ViewColor` (the player's color
for the density, graded and premultiplied by the mask's copy, eroded and feathered). The view is made again when a new capture comes in, told by its capture time,
or when a setting it is made with changes.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Annotated

# Third-party imports
import numpy as np
from OpenGL.GL import GL_RGBA16F

# Local application imports
from modules.flow import DepthView, HasPorts, Port, TextureKind, depth_step, fixed_window, place_points, view_span
from modules.gl import Style, Texture
from modules.pose import BBox, Frame, Points2D
from modules.render import StageLayer, StageOutput
from modules.utils import Rect
from ....settings import ViewCropSettings, ViewLayerSettings
from .view_color import ViewColor
from .view_crop import ViewCrop
from .view_depth import ViewDepth
from .view_mask import ViewMask


@dataclass
class ViewInput:
    """One camera's capture for the view stage: `image` its color, `depth` its metres (0 for none, None without a
    depth camera), both on the camera image's view; `timestamp` its capture time, the capture's identity (None: no
    capture yet); `pose` its player's, whose points the view places in its crop, or None."""
    image: Annotated[Texture, TextureKind.IMAGE]
    depth: Annotated[Texture | None, TextureKind.DEPTH]
    timestamp: float | None
    pose: Frame | None = None


@dataclass
class ViewOutput(StageOutput):
    """The view's textures, at the view's size, in the order the view makes them: `image` the cropped color, `mask`
    the player's shape on the image's edges, `depth` the player's metres wherever the mask is, as the motion
    measures z from them, `color` the player's color for the density, the image graded and premultiplied by the
    mask's copy."""
    image: Annotated[Texture, TextureKind.IMAGE]
    mask: Annotated[Texture, TextureKind.SCALAR]
    depth: Annotated[DepthView, TextureKind.DEPTH]
    color: Annotated[Texture, TextureKind.IMAGE]


class ViewLayer(StageLayer[ViewInput, ViewOutput]):
    """The view stage's layer. `fov` is the camera image's horizontal field (degrees), for the metres a view width
    spans. Places the pose's points in the crop (`crop_pose_points`, `lost`) for the pose drawn over the view.
    Its ports are its parts' textures under their names, the input's, and the outputs no part holds. Draws
    nothing."""

    def __init__(self, settings: ViewLayerSettings, fov: float) -> None:
        super().__init__(settings)
        self.settings: ViewLayerSettings = settings
        self._fov: float = fov

        self._crop: ViewCrop = ViewCrop(settings.crop)
        self._mask: ViewMask = ViewMask(settings.mask)
        self._depth: ViewDepth = ViewDepth(settings.crop)
        self._color: ViewColor = ViewColor(settings.color)

        self._window: Rect = Rect(0.0, 0.0, 1.0, 1.0)  # the box's sides, uv on the camera image, rows down
        self._grid_size: tuple[int, int] = (0, 0)       # the depth's grid, its samples on the camera image
        self._made: tuple | None = None                 # the capture and the settings the view was made from
        self._timestamp: float | None = None
        self._pose_points: Points2D | None = None

    # ========== What it hands out ==========

    @property
    def output(self) -> ViewOutput:
        box: ViewCropSettings = self.settings.crop
        depth: DepthView = DepthView(self._depth.depth, view_span(self._window, self._fov), depth_step(self._window, self._grid_size),
                                     box.near, box.far)
        return ViewOutput(self._crop.image, self._mask.mask, depth, self._color.texture)

    @property
    def ports(self) -> dict[str, Port]:
        """The stage's ports, the input's depth a depth in the box's range, so a monitor draws its colormap."""
        ports: dict[str, Port] = super().ports
        if 'input.depth' in ports:
            box: ViewCropSettings = self.settings.crop
            ports['input.depth'] = Port(ports['input.depth'].texture, TextureKind.DEPTH, depth_range=(box.near, box.far))
        return ports

    def _components(self) -> dict[str, HasPorts]:
        return {'crop': self._crop, 'mask': self._mask, 'depth': self._depth, 'color': self._color}

    @property
    def texture(self) -> Texture:
        """The view's image, the shape the view's monitor is drawn in."""
        return self._crop.image

    @property
    def timestamp(self) -> float | None:
        """The capture time of the capture the view holds, None before the first."""
        return self._timestamp

    @property
    def lost(self) -> bool:
        """Whether the view holds no pose: the pose drawn over it is cleared."""
        return self._pose_points is None

    @property
    def crop_pose_points(self) -> Points2D | None:
        """The capture's pose's points in the view, uv with rows down; None without a pose."""
        return self._pose_points

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._crop.allocate(width, height)
        self._mask.allocate(width, height)
        self._depth.allocate(width, height)
        self._color.allocate(width, height, GL_RGBA16F)
        self._made = None

    def deallocate(self) -> None:
        self._crop.deallocate()
        self._mask.deallocate()
        self._depth.deallocate()
        self._color.deallocate()
        super().deallocate()

    def reset(self) -> None:
        self._crop.clear()
        self._mask.clear()
        self._depth.clear()
        self._color.clear()
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
        made: tuple = (input.timestamp, image.width, image.height, view.width, view.height, self._crop.made_with,
                       self._mask.made_with, self._color.made_with)
        if made == self._made:
            return
        self._made = made
        self._timestamp = input.timestamp

        self._window = fixed_window(s.crop.center_x, s.crop.center_y, s.crop.height, view.width / view.height, image.width / image.height)
        roi: Rect = self._window
        depth: Texture | None = input.depth if input.depth is not None and input.depth.allocated else None
        self._grid_size = (depth.width, depth.height) if depth is not None else (0, 0)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Who, where, how far, each once: the box's image and the depth inside it on the depth's own pixels, that
        # depth's outline moved onto the image's (the mask), the depth upscaled to the view where it is the player's
        # and the mask is, then filled across the mask, so the mask and the depth agree
        self._crop.crop(image, depth, roi)
        self._mask.guide(view, self._crop.depth, roi, self._grid_size[1])
        self._depth.upscale(depth, roi, self._mask.mask)

        Style.pop_style()
        self._color.process(view, self._mask.mask)

        self._pose_points = self._place_pose(input.pose)

    def _place_pose(self, pose: Frame | None) -> Points2D | None:
        if pose is None:
            return None
        bbox: Rect = pose[BBox].to_rect()
        points: Points2D = pose[Points2D]
        x, y = points.get_xy_arrays()
        view_x, view_y = place_points(np.asarray(x), np.asarray(y), bbox, self._window)
        return Points2D.from_xy_arrays(view_x, view_y, points.scores)
