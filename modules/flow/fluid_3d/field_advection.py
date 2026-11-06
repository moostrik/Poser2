"""How the 3D fluid carries a field one step: semi-Lagrangian, or MacCormack on top of it."""

from OpenGL.GL import GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import Texture3D, SwapTexture3D
from .fluid_settings import Advection
from .shaders import Advect3D, MacCormack3D

_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


class FieldAdvection:
    """The advection's shaders, shared by the fluid's fields and the grid density. Each field owns its scratch
    volume, the MacCormack predictor, allocated at its first MacCormack step at the field's size."""

    def __init__(self) -> None:
        self._advect_shader: Advect3D = Advect3D()
        self._maccormack_shader: MacCormack3D = MacCormack3D()

    def allocate(self) -> None:
        self._advect_shader.allocate()
        self._maccormack_shader.allocate()

    def deallocate(self) -> None:
        self._advect_shader.deallocate()
        self._maccormack_shader.deallocate()

    def reload(self) -> None:
        self._advect_shader.reload()
        self._maccormack_shader.reload()

    def carry(self, advection: Advection, field: SwapTexture3D, scratch: Texture3D, velocity: Texture3D | None,
              obstacle: Texture3D, aspect: float, z_extent: float, step: float, dissipation: float,
              internal_format: int, has_obstacles: bool, evaporation: float = 0.0) -> bool:
        """Carry `field` one step along `velocity` (None: the field's own previous buffer, the velocity's
        self-advection). Semi-Lagrangian into the field; or its step into `scratch` as the predictor and the
        MacCormack correction into the field. Returns whether `scratch` was (re)allocated, so its owner sets
        its walls."""
        field.swap()
        source: Texture3D = field.back
        along: Texture3D = velocity if velocity is not None else source
        if advection != Advection.MACCORMACK:
            self._advect_shader.advect(source, field, along, obstacle, aspect, z_extent, step, dissipation,
                                       internal_format=internal_format, has_obstacles=has_obstacles,
                                       evaporation=evaporation)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            return False
        allocated: bool = False
        if not scratch.allocated or (scratch.width, scratch.height, scratch.depth) != (field.width, field.height, field.depth):
            scratch.deallocate()
            scratch.allocate(field.width, field.height, field.depth, internal_format)
            allocated = True
        self._advect_shader.advect(source, scratch, along, obstacle, aspect, z_extent, step, 1.0,
                                   internal_format=internal_format, has_obstacles=has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._maccormack_shader.use(source, scratch, field, along, obstacle, aspect, z_extent, step, dissipation,
                                    internal_format=internal_format, has_obstacles=has_obstacles,
                                    evaporation=evaporation)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        return allocated
