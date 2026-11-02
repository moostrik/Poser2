"""Bridge-specific shaders."""

from .density_bridge_shader import DensityBridgeShader
from .depth_dead_zone import DepthDeadZone
from .depth_smooth import DepthSmooth
from .depth_velocity_shader import DepthVelocityShader
from .range_toward import RangeToward
from .kinetic_curl import KineticCurl
from .kinetic_emission import KineticEmission
from .kinetic_hold import KineticHold
from .kinetic_input import KineticInput
from .kinetic_onset import KineticOnset
from .kinetic_patch import KineticPatch
from .kinetic_projection import KineticProjection, Projection
from .kinetic_reservoir import KineticReservoir
from .kinetic_strip_shader import KineticStripShader
from .kinetic_summary import KineticSummary
from .kinetic_velocity import KineticVelocity
from .speed_3d import Speed3D
from .temperature_bridge_shader import TemperatureBridgeShader
