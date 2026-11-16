"""The view stage: the flow's view of a capture, the stage before the motion.

`ViewInput` is the capture as the app hands it each tick: its color image and its depth on the camera image's view
(the uploader's, taken into GL by the capture's source layers), its capture time and the pose of its player.
`ViewOutput` is what the motion and the bridge read. The view runs its steps in order, each a part with its own
textures but the crop: the crop (the image cut to the view's crop at the view's size), `ViewDepth` (the depth
read into it), `ViewDepthMask` (the depth cut to the player's band, which gates z, and the mask, that cut's edge
moved onto the image's by the guided filter, He 2010), with `MASK` the depth again on the mask and cut again,
and `ColorGrade` (the color, the image graded, premultiplied by the mask). The view is made again when a new capture
comes in, told by its capture time, or when a setting it is made with changes.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Annotated

# Third-party imports
import numpy as np
from OpenGL.GL import GL_RGBA16F

# Local application imports
from modules.flow import DepthRead, DepthView, HasPorts, Port, Ports, TextureKind, depth_step, fixed_window, place_points, view_span
from modules.gl import Fbo, Style, Texture
from modules.pose import BBox, Frame, Points2D
from modules.render import StageLayer, StageOutput
from modules.utils import Rect
from ....settings import ViewDepthMaskSettings, ViewLayerSettings
from .color_grade import ColorGrade
from .view_depth import ViewDepth
from .view_depth_mask import ViewDepthMask


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
    """The view's textures, at the view's size, in the order the view makes them: `image` the cropped color,
    `depth` the metres as the motion measures z from them, `depth_mask` where the depth lies from `near` to `far`,
    `mask` the player's shape on the image's edges, `graded_image` the image graded and premultiplied by the mask,
    the bridge's density color."""
    image: Annotated[Texture, TextureKind.IMAGE]
    depth: Annotated[DepthView, TextureKind.DEPTH]
    depth_mask: Annotated[Texture, TextureKind.SCALAR]
    mask: Annotated[Texture, TextureKind.SCALAR]
    graded_image: Annotated[Texture, TextureKind.IMAGE]


class ViewLayer(StageLayer[ViewInput, ViewOutput]):
    """The view stage's layer. `fov` is the camera image's horizontal field (degrees), for the metres a view width
    spans. Places the pose's points in the crop (`crop_pose_points`, `lost`) for the pose drawn over the view.
    Its ports are its parts' textures under their names, the input's, and the outputs no part holds. Draws
    nothing."""

    def __init__(self, settings: ViewLayerSettings, fov: float) -> None:
        super().__init__(settings)
        self.settings: ViewLayerSettings = settings
        self._fov: float = fov

        self._image: Fbo = Fbo()                        # the crop's: the camera image cut to the view
        self._depth_mask: ViewDepthMask = ViewDepthMask(settings.depth_mask)
        self._depth: ViewDepth = ViewDepth(settings.depth, self._depth_mask)
        self._color_grade: ColorGrade = ColorGrade(settings.color_grade)

        self._crop: Rect = Rect(0.0, 0.0, 1.0, 1.0)  # uv on the camera image, rows down
        self._grid_size: tuple[int, int] = (0, 0)       # the depth's grid, its samples on the camera image
        self._made: tuple | None = None                 # the capture and the settings the view was made from
        self._timestamp: float | None = None
        self._pose_points: Points2D | None = None

    # ========== What it hands out ==========

    @property
    def output(self) -> ViewOutput:
        band: ViewDepthMaskSettings = self.settings.depth_mask
        depth: DepthView = DepthView(self._depth.depth, view_span(self._crop, self._fov), depth_step(self._crop, self._grid_size),
                                     band.near, band.far)
        return ViewOutput(self._image, depth, self._depth_mask.depth_mask, self._depth_mask.mask, self._color_grade.texture)

    @property
    def ports(self) -> dict[str, Port]:
        """The stage's ports, the input's depth a depth in the band, so a monitor draws its colormap."""
        ports: dict[str, Port] = super().ports
        if 'input.depth' in ports:
            band: ViewDepthMaskSettings = self.settings.depth_mask
            ports['input.depth'] = Port(ports['input.depth'].texture, TextureKind.DEPTH, depth_range=(band.near, band.far))
        return ports

    def _components(self) -> dict[str, HasPorts]:
        return {'crop': Ports({'image': Port(self._image, TextureKind.IMAGE)}),     # a step without a class of its own
                'depth': self._depth, 'depth_mask': self._depth_mask, 'color_grade': self._color_grade}

    @property
    def texture(self) -> Texture:
        """The view's image, the shape the view's monitor is drawn in."""
        return self._image

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
        self._image.allocate(width, height, GL_RGBA16F)
        self._depth.allocate(width, height)
        self._depth_mask.allocate(width, height)
        self._color_grade.allocate(width, height, GL_RGBA16F)
        self._made = None

    def deallocate(self) -> None:
        self._image.deallocate()
        self._depth.deallocate()
        self._depth_mask.deallocate()
        self._color_grade.deallocate()
        super().deallocate()

    def reset(self) -> None:
        self._image.clear()
        self._depth.clear()
        self._depth_mask.clear()
        self._color_grade.clear()
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
        if not image.allocated or not self._image.allocated:
            return
        s: ViewLayerSettings = self.settings
        made: tuple = (input.timestamp, s.crop.center_x, s.crop.center_y, s.crop.height, image.width, image.height,
                       self._image.width, self._image.height, self._depth.made_with, self._depth_mask.made_with,
                       self._color_grade.made_with)
        if made == self._made:
            return
        self._made = made
        self._timestamp = input.timestamp

        self._crop = fixed_window(s.crop.center_x, s.crop.center_y, s.crop.height, self._image.width / self._image.height, image.width / image.height)
        roi: Rect = self._crop
        depth: Texture | None = input.depth if input.depth is not None and input.depth.allocated else None
        self._grid_size = (depth.width, depth.height) if depth is not None else (0, 0)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._image.begin()
        image.draw(src=roi)
        self._image.end()

        # The depth read within a surface first, cut by the player's band, the cut's edge moved onto the image's;
        # with MASK the depth read again, an edge's side from that mask, and cut again, so the depth's edges lie on
        # the mask's
        read: DepthRead = DepthRead(s.depth.depth_read)
        self._depth.read(depth, roi, DepthRead.SURFACE)
        self._depth_mask.cut(self._depth.depth)
        self._depth_mask.guide(self._image, roi, self._grid_size[1])
        if read == DepthRead.MASK and depth is not None:
            self._depth.read(depth, roi, read)
            self._depth_mask.cut(self._depth.depth)

        Style.pop_style()
        self._color_grade.process(self._image, self._depth_mask.mask)

        self._pose_points = self._place_pose(input.pose)

    def _place_pose(self, pose: Frame | None) -> Points2D | None:
        if pose is None:
            return None
        bbox: Rect = pose[BBox].to_rect()
        points: Points2D = pose[Points2D]
        x, y = points.get_xy_arrays()
        view_x, view_y = place_points(np.asarray(x), np.asarray(y), bbox, self._crop)
        return Points2D.from_xy_arrays(view_x, view_y, points.scores)
