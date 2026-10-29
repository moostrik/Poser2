"""Fluid simulation shaders."""

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
from .add_boolean import AddBoolean


