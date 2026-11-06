"""3D fluid simulation compute shaders. Every field is a GL_TEXTURE_3D on one grid."""

from .advect_3d import Advect3D
from .divergence_3d import Divergence3D
from .gradient_3d import Gradient3D
from .jacobi_pressure_3d import JacobiPressure3D
from .jacobi_diffusion_3d import JacobiDiffusion3D
from .vorticity_curl_3d import VorticityCurl3D
from .vorticity_force_3d import VorticityForce3D
from .buoyancy_3d import Buoyancy3D
from .inject_3d import Inject3D
from .boundary_velocity_3d import BoundaryVelocity3D
from .body_force_acceleration_3d import BodyForceAcceleration3D
from .dampen_3d import Dampen3D
from .add_3d import Add3D
from .inject_binary_3d import InjectBinary3D
from .composite_3d import Composite3D
