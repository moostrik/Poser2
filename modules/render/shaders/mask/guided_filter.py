"""Guided Filter shader: a mask's edge moved to where a guide image's edges run (He, Sun and Tang 2010)."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

PRODUCTS: int = 0
BOX: int = 1
COEFFICIENTS: int = 2
OUTPUT: int = 3


class GuidedFilter(Shader):
    """The guided filter's passes, which the caller chains: `products`, `box` (twice, then again on the
    coefficients), `coefficients`, `output`."""

    def products(self, guide: Texture, mask: Texture) -> None:
        """(I, p, I·I, I·p): the guide's luminance and the mask."""
        self.use(PRODUCTS, guide, mask)

    def box(self, field: Texture, horizontal: bool, radius: int) -> None:
        """Every channel's mean over `radius` pixels each side, along one axis."""
        self.use(BOX, field, horizontal=horizontal, radius=radius)

    def coefficients(self, means: Texture, epsilon: float) -> None:
        """(a, b) from the products' means; `epsilon` the guide's variance below which the mask is only averaged."""
        self.use(COEFFICIENTS, means, epsilon=epsilon)

    def output(self, coefficients: Texture, guide: Texture) -> None:
        """The mask, mean a · I + mean b, from the coefficients' means and the guide."""
        self.use(OUTPUT, coefficients, guide)

    def use(self, stage: int, tex0: Texture, tex1: Texture | None = None, horizontal: bool = True, radius: int = 1,
            epsilon: float = 1e-3) -> None:
        if not self.allocated or not self.shader_program or not tex0.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        if tex1 is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, tex1.tex_id)
            glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("stage"), stage)
        glUniform2f(self.get_uniform_loc("direction"), *((1.0, 0.0) if horizontal else (0.0, 1.0)))
        glUniform1i(self.get_uniform_loc("radius"), radius)
        glUniform1f(self.get_uniform_loc("epsilon"), epsilon)
        draw_quad()
