"""3D fluid simulation compute shaders (GL_TEXTURE_2D_ARRAY density variant).

Density-specific shaders are defined locally. Base 3D shaders are re-exported
from fluid3d.shaders so FluidFlow3DArray can import everything from one place.
"""

# Density-specific shaders (GL_TEXTURE_2D_ARRAY bindings)
from .Advect3DDensity import Advect3DDensity
from .Buoyancy3DDensity import Buoyancy3DDensity
from .Inject3DDensity import Inject3DDensity
from .InjectChannel3DDensity import InjectChannel3DDensity
from .Composite3DDensity import Composite3DDensity

# Base 3D shaders (re-exported from fluid3d)
from ...fluid3d.shaders import \
    Advect3D, Divergence3D, Gradient3D, \
    JacobiPressure3D, JacobiDiffusion3D, \
    VorticityCurl3D, VorticityForce3D, Buoyancy3D, \
    Inject3D, InjectChannel3D, \
    Clamp3D, Dampen3D, Composite3D, Add3D, \
    Blit3D, InjectBinary3D
