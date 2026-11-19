"""The 2D fluid's shaders."""

from .advect import Advect
from .divergence import Divergence
from .gradient import Gradient
from .jacobi_pressure import JacobiPressure
from .jacobi_pressure_compute import JacobiPressureCompute
from .jacobi_diffusion import JacobiDiffusion
from .jacobi_diffusion_compute import JacobiDiffusionCompute
from .vorticity_curl import VorticityCurl
from .vorticity_force import VorticityForce
from .buoyancy import Buoyancy
from .inject_velocity_2d import InjectVelocity2D, VelocityInjection
from .inject_density_2d import InjectDensity2D
from .inject_scalar_2d import InjectScalar2D, ScalarInjection
from .density_to_rgb_2d import DensityToRgb2D
