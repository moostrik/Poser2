"""The view stage: the flow's view of a capture, the stage before the motion.

`ViewInput` is the capture as the app hands it each tick: its color image and its depth on the camera image's view
(the uploader's, taken into GL by the capture's source layers), its capture time and the pose of its player.
`ViewOutput` is what the motion and the bridge read: the image and the depth cut to the view's fixed window at the
view's size, the depth read across a surface (`DepthUpscale`, at the view's `tolerance`), the depth mask (the view's depth cut by `near` and `far`, which gates z), the mask (the depth mask, its
edge moved onto the image's by the guided filter, He 2010), the foreground (the image where the mask is) and its
color, the foreground graded and sharpened. The view is made again
when a new capture comes in, told by its capture time, or when a setting it is made with changes.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Annotated

# Third-party imports
import numpy as np
from OpenGL.GL import GL_NEAREST, GL_R16F, GL_R32F, GL_RG32F, GL_RGBA16F, GL_RGBA32F

# Local application imports
from modules.flow import DepthRead, DepthView, DepthUpscale, TextureKind, \
                         depth_step, fixed_window, place_points, view_per_depth_pixel, view_span
from modules.gl import Fbo, SwapFbo, Style, Texture, shaders
from modules.pose import BBox, Frame, Points2D
from modules.render import StageLayer, StageOutput
from modules.utils import Rect
from ....settings import ViewLayerSettings
from .shaders import DepthMask, Grade


@dataclass
class ViewInput:
    """One camera's capture for the view stage: `image` its color, `depth` its metres (0 for none, None without a
    depth camera), both on the camera image's view; `timestamp` its capture time, the capture's identity (None: no
    capture yet); `pose` its player's, whose points the view places in its window, or None."""
    image: Annotated[Texture, TextureKind.IMAGE]
    depth: Annotated[Texture | None, TextureKind.SCALAR]
    timestamp: float | None
    pose: Frame | None = None


@dataclass
class ViewOutput(StageOutput):
    """The view's textures, at the view's size: `image` the color, `foreground` the color where the mask is,
    `depth` the metres as the motion measures z from them, `mask` the player's shape on the image's edges,
    `depth_mask` where the depth lies from `near` to `far`, `color` the foreground graded, the density's color."""
    image: Annotated[Texture, TextureKind.IMAGE]
    foreground: Annotated[Texture, TextureKind.IMAGE]
    color: Annotated[Texture, TextureKind.IMAGE]
    depth: Annotated[DepthView, TextureKind.DEPTH]
    mask: Annotated[Texture, TextureKind.SCALAR]
    depth_mask: Annotated[Texture, TextureKind.SCALAR]


class ViewLayer(StageLayer[ViewInput, ViewOutput]):
    """The view stage's layer. `fov` is the camera image's horizontal field (degrees), for the metres a view width
    spans; `settings.tolerance` tells one surface from another. Places
    the pose's points in the window (`crop_pose_points`, `lost`) for the pose drawn over the view; its last pass is
    the color: the foreground graded, sharpened and masked. Draws nothing."""

    def __init__(self, settings: ViewLayerSettings, fov: float) -> None:
        super().__init__(settings)
        self.settings: ViewLayerSettings = settings
        self._fov: float = fov

        self._color: SwapFbo = SwapFbo()                # the foreground graded, then sharpened, then masked
        self._image: Fbo = Fbo()
        self._depth: Fbo = Fbo(interpolation=GL_NEAREST)     # an edge never reads as a distance in between
        self._depth_mask: Fbo = Fbo()
        self._mask: Fbo = Fbo()
        self._foreground: Fbo = Fbo()
        self._guided: Fbo = Fbo()                       # the depth mask, its edge on the image's, before the threshold
        self._products: SwapFbo = SwapFbo()             # the guided filter's products and their means (32-bit: a variance)
        self._coefficients: SwapFbo = SwapFbo()         # its coefficients a, b and their means

        self._depth_shader: DepthUpscale = DepthUpscale()
        self._depth_mask_shader: DepthMask = DepthMask()
        self._guided_shader: shaders.GuidedFilter = shaders.GuidedFilter()
        self._threshold_shader: shaders.MaskAntialias = shaders.MaskAntialias()
        self._foreground_shader: shaders.MaskMultiply = shaders.MaskMultiply()
        self._grade_shader: Grade = Grade()
        self._sharpen_shader: shaders.Sharpen = shaders.Sharpen()
        self._color_mask_shader: shaders.MaskApply = shaders.MaskApply()

        self._window: Rect = Rect(0.0, 0.0, 1.0, 1.0)  # uv on the camera image, rows down
        self._grid_size: tuple[int, int] = (0, 0)       # the depth's grid, its samples on the camera image
        self._made: tuple | None = None                 # the capture and the settings the view was made from
        self._timestamp: float | None = None
        self._pose_points: Points2D | None = None

    # ========== What it hands out ==========

    @property
    def output(self) -> ViewOutput:
        s: ViewLayerSettings = self.settings
        depth: DepthView = DepthView(self._depth, view_span(self._window, self._fov), depth_step(self._window, self._grid_size), s.near, s.far)
        return ViewOutput(self._image, self._foreground, self._color, depth, self._mask, self._depth_mask)

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
        self._depth.allocate(width, height, GL_R32F)
        self._depth_mask.allocate(width, height, GL_R16F)
        self._mask.allocate(width, height, GL_R16F)
        self._foreground.allocate(width, height, GL_RGBA16F)
        self._guided.allocate(width, height, GL_R16F)
        self._products.allocate(width, height, GL_RGBA32F)
        self._coefficients.allocate(width, height, GL_RG32F)
        self._color.allocate(width, height, GL_RGBA16F)
        for shader in self._shaders():
            shader.allocate()
        self._made = None

    def deallocate(self) -> None:
        for fbo in (self._image, self._depth, self._depth_mask, self._mask, self._foreground, self._guided, self._products,
                    self._coefficients, self._color):
            fbo.deallocate()
        for shader in self._shaders():
            shader.deallocate()
        super().deallocate()

    def _shaders(self) -> tuple:
        return (self._depth_shader, self._depth_mask_shader, self._guided_shader, self._threshold_shader, self._foreground_shader,
                self._grade_shader, self._sharpen_shader, self._color_mask_shader)

    def reset(self) -> None:
        for fbo in (self._image, self._depth, self._depth_mask, self._mask, self._foreground, self._color):
            fbo.clear()
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
        made: tuple = (input.timestamp, s.fixed_x, s.fixed_y, s.fixed_height, s.near, s.far, s.guided_radius, s.edge_contrast,
                       s.depth_read, s.tolerance, image.width, image.height, self._image.width, self._image.height,
                       s.grade_enabled, s.exposure, s.offset, s.gamma, s.contrast, s.saturation, s.sharpen, s.use_mask)
        if made == self._made:
            return
        self._made = made
        self._timestamp = input.timestamp

        self._window = fixed_window(s.fixed_x, s.fixed_y, s.fixed_height, self._image.width / self._image.height, image.width / image.height)
        roi: Rect = self._window
        depth: Texture | None = input.depth if input.depth is not None and input.depth.allocated else None
        self._grid_size = (depth.width, depth.height) if depth is not None else (0, 0)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._image.begin()
        image.draw(src=roi)
        self._image.end()

        # The depth read within a surface first, cut by the player's range, the cut's edge moved onto the image's;
        # with MASK the depth read again, an edge's side from that mask, and cut again, so the depth's edges lie on
        # the mask's
        read: DepthRead = DepthRead(s.depth_read)
        self._read_depth(depth, roi, DepthRead.SURFACE, s)
        self._cut_and_guide(s)
        if read == DepthRead.MASK and depth is not None:
            self._read_depth(depth, roi, read, s)
            self._depth_mask.begin()
            self._depth_mask_shader.use(self._depth, s.near, s.far)
            self._depth_mask.end()

        self._foreground.begin()
        self._foreground_shader.use(self._image, self._mask)
        self._foreground.end()

        self._make_color(s)
        Style.pop_style()

        self._pose_points = self._place_pose(input.pose)

    def _make_color(self, s: ViewLayerSettings) -> None:
        """The foreground graded, sharpened where `sharpen`, with the mask as alpha where `use_mask`; the
        foreground as it is while not `grade_enabled`."""
        self._color.swap()
        self._color.begin()
        if s.grade_enabled:
            self._grade_shader.use(self._foreground, s.exposure, s.offset, s.gamma, s.contrast, s.saturation)
        else:
            self._foreground.draw()
        self._color.end()
        if not s.grade_enabled:
            return
        if s.sharpen > 0.0:
            self._color.swap()
            self._color.begin()
            self._sharpen_shader.use(self._color.back, s.sharpen)
            self._color.end()
        if s.use_mask:
            self._color.swap()
            self._color.begin()
            self._color_mask_shader.use(self._color.back, self._mask)
            self._color.end()

    def _read_depth(self, depth: Texture | None, roi: Rect, read: DepthRead, s: ViewLayerSettings) -> None:
        """The depth into the view the way `read` says; none without a depth."""
        self._depth.clear()             # without a depth: none, not the last capture's
        if depth is None:
            return
        self._depth.begin()
        self._depth_shader.use(depth, roi, self.settings.tolerance, read, mask=self._mask,
                               near=s.near, far=s.far, reach=s.guided_radius)
        self._depth.end()

    def _cut_and_guide(self, s: ViewLayerSettings) -> None:
        """The depth mask, the view's depth cut by `near` and `far`; the mask, its edge moved onto the image's."""
        self._depth_mask.begin()
        self._depth_mask_shader.use(self._depth, s.near, s.far)
        self._depth_mask.end()
        guided: Texture = self._guide_edge(s)
        self._mask.begin()
        self._threshold_shader.use(guided)
        self._mask.end()

    def _guide_edge(self, s: ViewLayerSettings) -> Texture:
        """The depth mask, its edge moved to the image's (the guided filter): the products, their box means, the
        coefficients, their box means, the output. The radius, in the depth's pixels, is drawn in the view's; the
        filter's ε is `edge_contrast` squared, a variance of the image's brightness. Off, the depth mask itself."""
        radius: int = round(s.guided_radius * view_per_depth_pixel(self._window, self._grid_size[1], self._image.height))
        if radius < 1:
            return self._depth_mask
        products, coefficients = self._products, self._coefficients
        products.begin()
        self._guided_shader.products(self._image, self._depth_mask)
        products.end()
        for fbo in (products, coefficients):
            if fbo is coefficients:
                coefficients.begin()
                self._guided_shader.coefficients(products, s.edge_contrast ** 2)
                coefficients.end()
            for horizontal in (True, False):
                fbo.swap()
                fbo.begin()
                self._guided_shader.box(fbo.back, horizontal, radius)
                fbo.end()
        self._guided.begin()
        self._guided_shader.output(coefficients, self._image)
        self._guided.end()
        return self._guided

    def _place_pose(self, pose: Frame | None) -> Points2D | None:
        if pose is None:
            return None
        bbox: Rect = pose[BBox].to_rect()
        points: Points2D = pose[Points2D]
        x, y = points.get_xy_arrays()
        view_x, view_y = place_points(np.asarray(x), np.asarray(y), bbox, self._window)
        return Points2D.from_xy_arrays(view_x, view_y, points.scores)
