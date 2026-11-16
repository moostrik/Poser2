"""ViewDepthMask: who and where. The player cut on the depth's own pixels, and that cut's outline moved onto the
image's edges by the guided filter: the mask, the view's one outline."""

# Third-party imports
from OpenGL.GL import GL_R16F, GL_RG32F, GL_RGBA32F

# Local application imports
from modules.flow import DepthCut, Port, TextureKind, view_per_depth_pixel
from modules.gl import Fbo, SwapFbo, Texture, shaders
from modules.utils import Rect
from ....settings import ViewDepthMaskSettings


class ViewDepthMask:
    """`cut` makes the cut, where the depth pixel a view pixel lies in is from `near` to `far` (who); `guide` makes
    the mask, the cut's outline moved onto the image's (where; the guided filter, He 2010). Both are its own
    textures."""

    def __init__(self, settings: ViewDepthMaskSettings) -> None:
        self.settings: ViewDepthMaskSettings = settings
        self._cut: Fbo = Fbo()
        self._mask: Fbo = Fbo()
        self._guided: Fbo = Fbo()                       # the cut, its edge on the image's, before the threshold
        self._products: SwapFbo = SwapFbo()             # the guided filter's products and their means (32-bit: a variance)
        self._coefficients: SwapFbo = SwapFbo()         # its coefficients a, b and their means
        self._cut_shader: DepthCut = DepthCut()
        self._guided_shader: shaders.GuidedFilter = shaders.GuidedFilter()
        self._threshold_shader: shaders.MaskAntialias = shaders.MaskAntialias()

    @property
    def mask(self) -> Texture:
        return self._mask

    @property
    def ports(self) -> dict[str, Port]:
        return {'cut': Port(self._cut, TextureKind.SCALAR), 'mask': Port(self._mask, TextureKind.SCALAR)}

    @property
    def made_with(self) -> tuple:
        """The settings its textures are made with, so its owner makes them again when one changes."""
        s: ViewDepthMaskSettings = self.settings
        return (s.near, s.far, s.guided_radius, s.edge_contrast)

    def allocate(self, width: int, height: int) -> None:
        for fbo in (self._cut, self._mask, self._guided):
            fbo.allocate(width, height, GL_R16F)
        self._products.allocate(width, height, GL_RGBA32F)
        self._coefficients.allocate(width, height, GL_RG32F)
        for shader in (self._cut_shader, self._guided_shader, self._threshold_shader):
            shader.allocate()

    def deallocate(self) -> None:
        for fbo in (self._cut, self._mask, self._guided, self._products, self._coefficients):
            fbo.deallocate()
        for shader in (self._cut_shader, self._guided_shader, self._threshold_shader):
            shader.deallocate()

    def clear(self) -> None:
        self._cut.clear()
        self._mask.clear()

    def cut(self, depth: Texture | None, crop: Rect) -> None:
        """The cut: the player on `depth`'s own pixels (metres on the camera image, 0 for none) across `crop`;
        none without a depth."""
        self._cut.clear()
        if depth is None:
            return
        self._cut.begin()
        self._cut_shader.use(depth, crop, self.settings.near, self.settings.far)
        self._cut.end()

    def guide(self, image: Texture, crop: Rect, depth_rows: int) -> None:
        """The mask: the cut, its edge moved to `image`'s, thresholded. The radius, in the depth's pixels
        (`depth_rows` of them down the camera image), is drawn in the view's, across `crop`; the filter's ε is
        `edge_contrast` squared, a variance of the image's brightness. With no radius, the cut itself."""
        s: ViewDepthMaskSettings = self.settings
        guided: Texture = self._cut
        radius: int = round(s.guided_radius * view_per_depth_pixel(crop, depth_rows, image.height))
        if radius >= 1:
            products, coefficients = self._products, self._coefficients
            products.begin()
            self._guided_shader.products(image, self._cut)
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
            self._guided_shader.output(coefficients, image)
            self._guided.end()
            guided = self._guided
        self._mask.begin()
        self._threshold_shader.use(guided)
        self._mask.end()
