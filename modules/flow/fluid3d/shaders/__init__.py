"""3D fluid simulation compute shaders.

Simulation fields are GL_TEXTURE_3D; density is GL_TEXTURE_2D_ARRAY
(optimal per-layer 2D tiling) with dedicated density shaders.
"""

from .advect_3d import Advect3D
from .divergence_3d import Divergence3D
from .gradient_3d import Gradient3D
from .jacobi_pressure_3d import JacobiPressure3D
from .jacobi_diffusion_3d import JacobiDiffusion3D
from .vorticity_curl_3d import VorticityCurl3D
from .vorticity_force_3d import VorticityForce3D
from .buoyancy_3d import Buoyancy3D
from .inject_3d import Inject3D
from .inject_channel_3d import InjectChannel3D
from .inject_flow_3d import InjectFlow3D
from .dampen_3d import Dampen3D
from .add_3d import Add3D
from .inject_binary_3d import InjectBinary3D

# Density-specific shaders (GL_TEXTURE_2D_ARRAY bindings)
from .advect_3d_density import Advect3DDensity
from .buoyancy_3d_density import Buoyancy3DDensity
from .inject_3d_density import Inject3DDensity
from .inject_channel_3d_density import InjectChannel3DDensity
from .composite_3d_density import Composite3DDensity
