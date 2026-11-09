"""AddToFaces3D - what is added to the velocity on the cells' faces, in place: a force at the cells, the
boundary's drag, the body force."""
from __future__ import annotations

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader
from .face_binding import bind_faces, bind_face_writes

FORCE: int = 0
DRAG: int = 1
BODY_FORCE: int = 2


class AddToFaces3D(ComputeShader):
    """Each face its two cells' force × dt; or pulled toward the body's velocity; or the body's acceleration
    added; a face held at 0 left."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4
    _cells: tuple[int, int, int] = (0, 0, 0)    # the cell counts of the call in progress

    def add_force(self, faces: Texture3D, force: Texture3D, strength: float, obstacle: Texture3D,
                  closed: tuple[bool, bool, bool], has_obstacles: bool) -> None:
        """`force` (RGBA at the cells, widths/s²) × `strength` (dt) added, each face its two cells' mean."""
        if not self._begin(FORCE, faces, (force.width, force.height, force.depth), obstacle, closed, has_obstacles):
            return
        self.bind_texture_3d(0, force, 'uForce')
        glUniform1f(self.get_uniform_loc('uStrength'), strength)
        self._dispatch()

    def drag(self, faces: Texture3D, cells: tuple[int, int, int], velocity: Texture, target_layer: float,
             spread: float, gain: float, share: float, weight: float, floor: float, depth_scale: float,
             obstacle: Texture3D, closed: tuple[bool, bool, bool], has_obstacles: bool) -> None:
        """Each face in the slab pulled toward the body's velocity (widths/s) × `gain` by `share` × its presence
        × `weight`, never past it."""
        if not self._begin(DRAG, faces, cells, obstacle, closed, has_obstacles):
            return
        self._bind_input(velocity, target_layer, spread, depth_scale)
        glUniform1f(self.get_uniform_loc('uGain'), gain)
        glUniform1f(self.get_uniform_loc('uShare'), share)
        glUniform1f(self.get_uniform_loc('uWeight'), weight)
        glUniform1f(self.get_uniform_loc('uFloor'), floor)
        self._dispatch()

    def add_acceleration(self, faces: Texture3D, cells: tuple[int, int, int], acceleration: Texture,
                         target_layer: float, spread: float, strength: float, depth_scale: float,
                         obstacle: Texture3D, closed: tuple[bool, bool, bool], has_obstacles: bool) -> None:
        """The body's acceleration (widths/s²) × `strength` (weight × gain × dt) added in the slab."""
        if not self._begin(BODY_FORCE, faces, cells, obstacle, closed, has_obstacles):
            return
        self._bind_input(acceleration, target_layer, spread, depth_scale)
        glUniform1f(self.get_uniform_loc('uStrength'), strength)
        self._dispatch()

    def _begin(self, mode: int, faces: Texture3D, cells: tuple[int, int, int], obstacle: Texture3D,
               closed: tuple[bool, bool, bool], has_obstacles: bool) -> bool:
        if not self.allocated or not self.shader_program:
            return False
        glUseProgram(self.shader_program)
        bind_faces(self, faces, cells, closed, has_obstacles, obstacle)
        bind_face_writes(self, faces, in_place=True)
        glUniform1i(self.get_uniform_loc('uMode'), mode)
        self._cells = cells
        return True

    def _bind_input(self, texture: Texture, target_layer: float, spread: float, depth_scale: float) -> None:
        glActiveTexture(int(GL_TEXTURE0) + 1)
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)
        glUniform1i(self.get_uniform_loc('uInput'), 1)
        glUniform1f(self.get_uniform_loc('uTargetLayer'), target_layer)
        glUniform1f(self.get_uniform_loc('uSpread'), spread)
        glUniform1f(self.get_uniform_loc('uDepthScale'), depth_scale)

    def _dispatch(self) -> None:
        self.dispatch(self._cells[0] + 1, self._cells[1] + 1, self._cells[2] + 1)
