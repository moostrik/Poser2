"""3D fluid simulation compute shaders.

Simulation fields are GL_TEXTURE_3D; density is GL_TEXTURE_2D_ARRAY
(optimal per-layer 2D tiling) with dedicated density shaders.
"""

from .Advect3D import Advect3D
from .Divergence3D import Divergence3D
from .Gradient3D import Gradient3D
from .JacobiPressure3D import JacobiPressure3D
from .JacobiDiffusion3D import JacobiDiffusion3D
from .VorticityCurl3D import VorticityCurl3D
from .VorticityForce3D import VorticityForce3D
from .Buoyancy3D import Buoyancy3D
from .Inject3D import Inject3D
from .InjectChannel3D import InjectChannel3D
from .InjectVelocityZ3D import InjectVelocityZ3D
from .Dampen3D import Dampen3D
from .Add3D import Add3D
from .InjectBinary3D import InjectBinary3D

# Density-specific shaders (GL_TEXTURE_2D_ARRAY bindings)
from .Advect3DDensity import Advect3DDensity
from .Buoyancy3DDensity import Buoyancy3DDensity
from .Inject3DDensity import Inject3DDensity
from .InjectChannel3DDensity import InjectChannel3DDensity
from .Composite3DDensity import Composite3DDensity
