"""Renders centered and rotated mask with temporal blending and blur, its edge guided by the camera image."""

# Third-party imports
from OpenGL.GL import GL_R16F, GL_RG32F, GL_RGBA16F, GL_RGBA32F

# Local application imports
from modules.settings import Field, BaseSettings
from ..layer_base import LayerBase, HasCaptureTime
from .center_geometry import CenterGeometry, SourceSpace
from ...shaders import DrawRoi, GuidedFilter, MaskAA, MaskBlend, MaskBlur, MaskDilate, MaskErode
from modules.gl import Fbo, SwapFbo, Texture, Style
from modules.utils import Rect

_FULL_ROI: Rect = Rect(0.0, 0.0, 1.0, 1.0)


class CenterMaskLayerSettings(BaseSettings):
    """Configuration for CenterMaskLayer temporal blending and blur."""
    blend_factor:   Field[float] = Field(0.2, min=0.0, max=1.0, description="Temporal blending strength (0=static, 1=instant)")
    dilation_steps: Field[int]   = Field(0, min=-10, max=10, description="Mask dilation iterations (positive=expand, negative=shrink)")
    blur_steps:     Field[int]   = Field(0, min=0, max=10, description="Number of blur iterations")
    blur_radius:    Field[float] = Field(1.0, min=0.0, max=20.0, description="Blur kernel radius in pixels")
    guided_radius:  Field[float] = Field(0.0, min=0.0, max=8.0, step=0.25, newline=True, description="How far the image moves the mask's edge to its own, in the source mask's pixels; 0 = off")
    edge_contrast:  Field[float] = Field(0.03, min=0.001, max=0.3, step=0.001, description="The image's brightness spread (0 to 1) above which the mask's edge follows the image's")


class CenterMaskLayer(LayerBase):
    """Renders mask image cropped and rotated with temporal blending and blur. `space` says whether
    the mask is on the pose's crop or on the camera image. `source` is the layer whose texture it
    draws; `timestamp` is that frame's capture time, forwarded. `guide`, an image in the same space, moves
    the mask's edge to where its own edges run (a guided filter, He et al. 2010) over `guided_radius` of the
    source's pixels, where the image's brightness spreads more than `edge_contrast`, so a mask coarser than
    the image, the depth's, takes the image's outline instead of its own pixels. `unrefined` is the source's
    mask on the view as it comes, before the guide and every step after it."""

    def __init__(self, cam_id: int, geometry: CenterGeometry, source: LayerBase, space: SourceSpace, settings: CenterMaskLayerSettings | None = None,
                 guide: LayerBase | None = None) -> None:
        self._cam_id: int = cam_id
        self._geometry: CenterGeometry = geometry
        self._source: LayerBase = source
        self._space: SourceSpace = space
        self._guide: LayerBase | None = guide
        self._timestamp: float | None = None     # the capture time of the frame drawn last

        # Configuration
        self.settings: CenterMaskLayerSettings = settings or CenterMaskLayerSettings()

        # FBOs
        self._roi_fbo: Fbo = Fbo()  # Single temp buffer for ROI
        self._unrefined_fbo: Fbo = Fbo()                # the source's mask on the view as it comes
        self._blend_fbo: SwapFbo = SwapFbo()
        self._mask_blur_fbo: SwapFbo = SwapFbo()  # Used for AA output, dilation, and blur
        self._mask_fbo: Fbo = Fbo()  # Output, one buffer however many passes swapped the blur buffer
        self._guide_fbo: Fbo = Fbo()                    # the guide's ROI, the mask's view
        self._products_fbo: SwapFbo = SwapFbo()         # the guided filter's products and their means (32-bit: a variance)
        self._coefficients_fbo: SwapFbo = SwapFbo()     # its coefficients a, b and their means

        # Shaders
        self._roi_shader = DrawRoi()
        self._mask_blend_shader = MaskBlend()
        self._mask_AA_shader = MaskAA()
        self._mask_dilate_shader = MaskDilate()
        self._mask_erode_shader = MaskErode()
        self._mask_blur_shader = MaskBlur()
        self._guided_shader = GuidedFilter()

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._mask_fbo

    @property
    def unrefined(self) -> Texture:
        """The source's mask on the view as it comes: before the guide, the blend, the AA, the erosion and the blur."""
        return self._unrefined_fbo

    @property
    def timestamp(self) -> float | None:
        """The capture time of the camera frame the mask it holds was made from, its source's; None without one."""
        return self._timestamp

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._roi_fbo.allocate(width, height, GL_R16F)
        self._unrefined_fbo.allocate(width, height, GL_R16F)
        self._blend_fbo.allocate(width, height, GL_R16F)
        self._mask_blur_fbo.allocate(width, height, GL_R16F)
        self._mask_fbo.allocate(width, height, GL_R16F)
        if self._guide is not None:
            self._guide_fbo.allocate(width, height, GL_RGBA16F)
            self._products_fbo.allocate(width, height, GL_RGBA32F)
            self._coefficients_fbo.allocate(width, height, GL_RG32F)

        self._roi_shader.allocate()
        self._mask_blend_shader.allocate()
        self._mask_AA_shader.allocate()
        self._mask_dilate_shader.allocate()
        self._mask_erode_shader.allocate()
        self._mask_blur_shader.allocate()
        self._guided_shader.allocate()

    def deallocate(self) -> None:
        self._roi_fbo.deallocate()
        self._unrefined_fbo.deallocate()
        self._blend_fbo.deallocate()
        self._mask_blur_fbo.deallocate()
        self._mask_fbo.deallocate()
        self._guide_fbo.deallocate()
        self._products_fbo.deallocate()
        self._coefficients_fbo.deallocate()

        self._roi_shader.deallocate()
        self._mask_blend_shader.deallocate()
        self._mask_AA_shader.deallocate()
        self._mask_dilate_shader.deallocate()
        self._mask_erode_shader.deallocate()
        self._mask_blur_shader.deallocate()
        self._guided_shader.deallocate()

    def update(self) -> None:
        """Render the mask centered by CenterGeometry, in the mask's space."""
        if self._geometry.lost:
            self._blend_fbo.clear_all()
            self._mask_blur_fbo.clear_all()
            self._mask_fbo.clear()
            self._unrefined_fbo.clear()
            self._timestamp = None
            return

        source: Texture = self._source.texture
        if not self._geometry.ready or not source.allocated:
            return
        self._timestamp = self._source.timestamp if isinstance(self._source, HasCaptureTime) else None

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # The source on the view, kept as it comes, and the ROI buffer the steps refine
        roi = self._geometry.source_roi(self._space, source)
        self._unrefined_fbo.begin()
        self._roi_shader.use(source, *roi)
        self._unrefined_fbo.end()
        self._roi_fbo.begin()
        self._roi_shader.use(self._unrefined_fbo, _FULL_ROI)
        self._roi_fbo.end()
        self._guide_edge(self._view_per_source_pixel(roi[0], source))

        # Temporal blending
        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._mask_blend_shader.use(
            self._blend_fbo.back,
            self._roi_fbo,
            self.settings.blend_factor
        )
        self._blend_fbo.end()

        # AA directly to blur buffer
        self._mask_blur_fbo.begin()
        self._mask_AA_shader.use(self._blend_fbo)
        self._mask_blur_fbo.end()

        # Multi-pass dilation or erosion
        steps = self.settings.dilation_steps
        if steps > 0:
            for _ in range(steps):
                self._mask_blur_fbo.swap()
                self._mask_blur_fbo.begin()
                self._mask_dilate_shader.use(self._mask_blur_fbo.back, 1.0)
                self._mask_blur_fbo.end()
        elif steps < 0:
            for _ in range(-steps):
                self._mask_blur_fbo.swap()
                self._mask_blur_fbo.begin()
                self._mask_erode_shader.use(self._mask_blur_fbo.back, 1.0)
                self._mask_blur_fbo.end()

        # Multi-pass blur
        for i in range(self.settings.blur_steps):
            self._mask_blur_fbo.swap()
            self._mask_blur_fbo.begin()
            self._mask_blur_shader.use(
                self._mask_blur_fbo.back,
                True,
                self.settings.blur_radius
            )
            self._mask_blur_fbo.end()

            self._mask_blur_fbo.swap()
            self._mask_blur_fbo.begin()
            self._mask_blur_shader.use(
                self._mask_blur_fbo.back,
                False,
                self.settings.blur_radius
            )
            self._mask_blur_fbo.end()

        # Copy to the output, so readers holding `texture` see this frame's result
        self._mask_fbo.begin()
        self._roi_shader.use(self._mask_blur_fbo, _FULL_ROI)
        self._mask_fbo.end()

        Style.pop_style()

    def _view_per_source_pixel(self, roi: Rect, source: Texture) -> float:
        """The view's pixels one pixel of the source spans: the view's height over the ROI's height in the source's pixels."""
        source_rows: float = roi.height * source.height
        return self._roi_fbo.height / source_rows if source_rows > 0.0 else 1.0

    def _guide_edge(self, view_per_source_pixel: float) -> None:
        """The ROI's mask, its edge moved to the guide's (the guided filter): the products, their box means, the
        coefficients, their box means, the output, into the ROI buffer. The radius, in the source's pixels, is
        drawn in the view's; the filter's ε is `edge_contrast` squared, a variance of the image's brightness."""
        radius: int = round(self.settings.guided_radius * view_per_source_pixel)
        guide: LayerBase | None = self._guide
        if guide is None or radius < 1 or not self._guide_fbo.allocated or not guide.texture.allocated:
            return
        image: Texture = guide.texture
        self._guide_fbo.begin()
        self._roi_shader.use(image, *self._geometry.source_roi(self._space, image))
        self._guide_fbo.end()

        products, coefficients = self._products_fbo, self._coefficients_fbo
        products.begin()
        self._guided_shader.products(self._guide_fbo, self._roi_fbo)
        products.end()
        for fbo in (products, coefficients):
            if fbo is coefficients:
                coefficients.begin()
                self._guided_shader.coefficients(products, self.settings.edge_contrast ** 2)
                coefficients.end()
            for horizontal in (True, False):
                fbo.swap()
                fbo.begin()
                self._guided_shader.box(fbo.back, horizontal, radius)
                fbo.end()
        self._roi_fbo.begin()
        self._guided_shader.output(coefficients, self._guide_fbo)
        self._roi_fbo.end()
