"""ColorGrade: the view's last pass, its foreground graded, sharpened and masked into the color, the density's."""

# Local application imports
from modules.gl import SwapFbo, Style, Texture, shaders
from ....settings import ColorGradeSettings
from .shaders import Grade


class ColorGrade:
    """`process` grades a foreground (`Grade`), sharpens it where `sharpen` and takes its mask as alpha where
    `use_mask`, into its own buffer; the foreground as it is while not `enabled`. `texture` holds the result."""

    def __init__(self, settings: ColorGradeSettings) -> None:
        self.settings: ColorGradeSettings = settings
        self._fbo: SwapFbo = SwapFbo()
        self._grade_shader: Grade = Grade()
        self._sharpen_shader: shaders.Sharpen = shaders.Sharpen()
        self._mask_shader: shaders.MaskApply = shaders.MaskApply()

    @property
    def texture(self) -> Texture:
        return self._fbo        # the SwapFbo itself: always its newest image, safe to keep

    @property
    def made_with(self) -> tuple:
        """The settings the result is made with, so its owner makes it again when one changes."""
        s: ColorGradeSettings = self.settings
        return (s.enabled, s.use_mask, s.exposure, s.offset, s.gamma, s.contrast, s.saturation, s.sharpen)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        for shader in (self._grade_shader, self._sharpen_shader, self._mask_shader):
            shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        for shader in (self._grade_shader, self._sharpen_shader, self._mask_shader):
            shader.deallocate()

    def clear(self) -> None:
        self._fbo.clear()

    def process(self, source: Texture, mask: Texture) -> None:
        s: ColorGradeSettings = self.settings
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.swap()
        self._fbo.begin()
        if s.enabled:
            self._grade_shader.use(source, s.exposure, s.offset, s.gamma, s.contrast, s.saturation)
        else:
            source.draw()
        self._fbo.end()
        if s.enabled and s.sharpen > 0.0:
            self._fbo.swap()
            self._fbo.begin()
            self._sharpen_shader.use(self._fbo.back, s.sharpen)
            self._fbo.end()
        if s.enabled and s.use_mask:
            self._fbo.swap()
            self._fbo.begin()
            self._mask_shader.use(self._fbo.back, mask)
            self._fbo.end()
        Style.pop_style()
