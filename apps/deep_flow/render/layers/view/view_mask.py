"""ViewMask: where. The crop's depth's outline moved onto the image's edges by the guided filter: the mask, the view's
one outline."""

# Third-party imports
from OpenGL.GL import GL_R16F, GL_RG32F, GL_RGBA32F

# Local application imports
from modules.flow import Port, TextureKind, view_per_depth_pixel
from modules.gl import Fbo, SwapFbo, Texture, shaders
from modules.utils import Rect
from ....settings import ViewMaskSettings


class ViewMask:
    """`guide` makes the mask: where the crop's depth is (above 0), its outline moved onto the image's (the guided
    filter, He 2010), thresholded at 0.5 and antialiased over one pixel. Its own texture."""

    def __init__(self, settings: ViewMaskSettings) -> None:
        self.settings: ViewMaskSettings = settings
        self._mask: Fbo = Fbo()
        self._guided: Fbo = Fbo()                       # the crop's depth's presence, its edge on the image's, before the threshold
        self._products: SwapFbo = SwapFbo()             # the guided filter's products and their means (32-bit: a variance)
        self._coefficients: SwapFbo = SwapFbo()         # its coefficients a, b and their means
        self._guided_shader: shaders.GuidedFilter = shaders.GuidedFilter()
        self._threshold_shader: shaders.MaskAntialias = shaders.MaskAntialias()

    @property
    def mask(self) -> Texture:
        return self._mask

    @property
    def ports(self) -> dict[str, Port]:
        return {'mask': Port(self._mask, TextureKind.SCALAR)}

    @property
    def made_with(self) -> tuple:
        """The settings its texture is made with, so its owner makes it again when one changes."""
        s: ViewMaskSettings = self.settings
        return (s.guided_radius, s.edge_contrast)

    def allocate(self, width: int, height: int) -> None:
        for fbo in (self._mask, self._guided):
            fbo.allocate(width, height, GL_R16F)
        self._products.allocate(width, height, GL_RGBA32F)
        self._coefficients.allocate(width, height, GL_RG32F)
        for shader in (self._guided_shader, self._threshold_shader):
            shader.allocate()

    def deallocate(self) -> None:
        for fbo in (self._mask, self._guided, self._products, self._coefficients):
            fbo.deallocate()
        for shader in (self._guided_shader, self._threshold_shader):
            shader.deallocate()

    def clear(self) -> None:
        self._mask.clear()

    def guide(self, image: Texture, depth: Texture, crop: Rect, depth_rows: int) -> None:
        """The mask: where `depth` (the crop's, metres, 0 outside the box) is, its edge moved to `image`'s,
        thresholded. The radius, in the depth's pixels (`depth_rows` of them down the camera image), is drawn in the
        view's, across `crop`; the filter's ε is `edge_contrast` squared, a variance of the image's brightness. With
        no radius the filter gives the presence itself."""
        s: ViewMaskSettings = self.settings
        radius: int = max(0, round(s.guided_radius * view_per_depth_pixel(crop, depth_rows, image.height)))
        products, coefficients = self._products, self._coefficients
        products.begin()
        self._guided_shader.products(image, depth, presence=True)
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
        self._mask.begin()
        self._threshold_shader.use(self._guided)
        self._mask.end()
