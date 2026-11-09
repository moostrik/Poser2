"""3D fluid simulation compute shaders. Every field is a GL_TEXTURE_3D on one grid."""

from .advect_3d import Advect3D
from .maccormack_3d import MacCormack3D
from .divergence_3d import Divergence3D
from .gradient_3d import Gradient3D
from .jacobi_pressure_3d import JacobiPressure3D
from .sor_pressure_3d import SorPressure3D
from .prolong_3d import Prolong3D
from .residual_restrict_3d import ResidualRestrict3D
from .coarse_pressure_3d import CoarsePressure3D
from .vorticity_curl_3d import VorticityCurl3D
from .vorticity_force_3d import VorticityForce3D
from .buoyancy_3d import Buoyancy3D
from .inject_3d import Inject3D, InjectMode
from .inject_amount_3d import InjectAmount3D
from .velocity_average_3d import VelocityAverage3D
from .noise_coordinates_3d import NoiseCoordinates3D
from .turbulent_advect_3d import TurbulentAdvect3D
from .inject_binary_3d import InjectBinary3D
from .composite_3d import Composite3D, CompositeLight, view_march, OPACITY_LIMIT
from .light_volume_3d import LightVolume3D
from .light_blur_3d import LightBlur3D
from .composite_over import CompositeOver
from .knee import Knee

# The particle carrier
from .particle_emit import ParticleEmit
from .particle_ring import ParticleRing, RING_BYTES
from .particle_advect import ParticleAdvect
from .particle_sprites import ParticleSprites
from .particle_composite import ParticleComposite
from .particle_weighted import ParticleWeighted
from .particle_surface import ParticleSurface
from .particle_surface_smooth import ParticleSurfaceSmooth
from .particle_surface_shade import ParticleSurfaceShade

# The velocity on the faces
from .advect_faces_3d import AdvectFaces3D, SEMI_LAGRANGIAN, PREDICTOR, CORRECTION
from .faces_to_centers_3d import FacesToCenters3D
from .viscosity_faces_3d import ViscosityFaces3D
from .add_to_faces_3d import AddToFaces3D
