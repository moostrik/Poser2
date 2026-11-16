"""ViewDepthMask: the view's depth cut to the player's band, and that cut's edge moved onto the image's."""

# Third-party imports
from OpenGL.GL import GL_R16F, GL_RG32F, GL_RGBA32F

# Local application imports
from modules.flow import Port, TextureKind, view_per_depth_pixel
from modules.gl import Fbo, SwapFbo, Texture, shaders
from modules.utils import Rect
from ....settings import ViewDepthMaskSettings
from .shaders import DepthMask


class ViewDepthMask:
    """`cut` makes the depth mask, where a depth lies from `near` to `far`; `guide` makes the mask, the depth
    mask's edge moved onto the image's (the guided filter, He 2010). Both are its own textures."""

    def __init__(self, settings: ViewDepthMaskSettings) -> None:
        self.settings: ViewDepthMaskSettings = settings
        self._depth_mask: Fbo = Fbo()
        self._mask: Fbo = Fbo()
        self._guided: Fbo = Fbo()                       # the depth mask, its edge on the image's, before the threshold
        self._products: SwapFbo = SwapFbo()             # the guided filter's products and their means (32-bit: a variance)
        self._coefficients: SwapFbo = SwapFbo()         # its coefficients a, b and their means
        self._cut_shader: DepthMask = DepthMask()
        self._guided_shader: shaders.GuidedFilter = shaders.GuidedFilter()
        self._threshold_shader: shaders.MaskAntialias = shaders.MaskAntialias()

    @property
    def depth_mask(self) -> Texture:
        return self._depth_mask

    @property
    def mask(self) -> Texture:
        return self._mask

    @property
    def ports(self) -> dict[str, Port]:
        return {'depth_mask': Port(self._depth_mask, TextureKind.SCALAR), 'mask': Port(self._mask, TextureKind.SCALAR)}

    @property
    def made_with(self) -> tuple:
        """The settings its textures are made with, so its owner makes them again when one changes."""
        s: ViewDepthMaskSettings = self.settings
        return (s.near, s.far, s.guided_radius, s.edge_contrast)

    def allocate(self, width: int, height: int) -> None:
        for fbo in (self._depth_mask, self._mask, self._guided):
            fbo.allocate(width, height, GL_R16F)
        self._products.allocate(width, height, GL_RGBA32F)
        self._coefficients.allocate(width, height, GL_RG32F)
        for shader in (self._cut_shader, self._guided_shader, self._threshold_shader):
            shader.allocate()

    def deallocate(self) -> None:
        for fbo in (self._depth_mask, self._mask, self._guided, self._products, self._coefficients):
            fbo.deallocate()
        for shader in (self._cut_shader, self._guided_shader, self._threshold_shader):
            shader.deallocate()

    def clear(self) -> None:
        self._depth_mask.clear()
        self._mask.clear()

    def cut(self, depth: Texture) -> None:
        """The depth mask: where `depth` lies from `near` to `far`."""
        self._depth_mask.begin()
        self._cut_shader.use(depth, self.settings.near, self.settings.far)
        self._depth_mask.end()

    def guide(self, image: Texture, crop: Rect, depth_rows: int) -> None:
        """The mask: the depth mask, its edge moved to `image`'s, thresholded. The radius, in the depth's pixels
        (`depth_rows` of them down the camera image), is drawn in the view's, across `crop`; the filter's ε is
        `edge_contrast` squared, a variance of the image's brightness. With no radius, the depth mask itself."""
        s: ViewDepthMaskSettings = self.settings
        guided: Texture = self._depth_mask
        radius: int = round(s.guided_radius * view_per_depth_pixel(crop, depth_rows, image.height))
        if radius >= 1:
            products, coefficients = self._products, self._coefficients
            products.begin()
            self._guided_shader.products(image, self._depth_mask)
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
