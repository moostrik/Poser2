"""Push-pull shader: a depth's holes filled from the depth around them (Gortler et al. 1996), its passes."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture

PUSH: int = 0
PULL: int = 1
LAST: int = 2


class PushPull(Shader):
    """The passes, which the caller chains over its pyramid: `push` a level down at a time, then `pull` from the
    coarsest up, then `last` onto the read."""

    def push(self, finer: Texture, first: bool) -> None:
        """A level down: each pixel the mean of the pairs (depth × weight, weight) it covers; `first`, `finer` is
        the read, meters in red."""
        self._use(PUSH, (finer,), first)

    def pull(self, level: Texture, coarser: Texture) -> None:
        """`level`'s push plus (1 − its weight) times the `coarser` pulled level, read bilinearly."""
        self._use(PULL, (level, coarser))

    def last(self, read: Texture, coarser: Texture, mask: Texture) -> None:
        """The depth, meters in red and its confidence in green: `read`'s where it has one, else the `coarser` pulled
        level's depth and a confidence of 0; 0 and 0 where `mask` is 0."""
        self._use(LAST, (read, coarser, mask))

    def _use(self, stage: int, textures: tuple[Texture, ...], first: bool = False) -> None:
        if not self.allocated or not self.shader_program or not all(t.allocated for t in textures):
            return
        glUseProgram(self.shader_program)
        for unit, texture in zip((GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2), textures):
            glActiveTexture(unit)
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
        glUniform1i(self.get_uniform_loc("stage"), stage)
        glUniform1i(self.get_uniform_loc("first"), int(first))
        draw_quad()
