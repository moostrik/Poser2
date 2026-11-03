"""Bridge-specific shaders."""

from .density_bridge_shader import DensityBridgeShader
from .kinetic_advect import KineticAdvect
from .kinetic_curl import KineticCurl
from .kinetic_emission import KineticEmission
from .kinetic_hold import KineticHold
from .kinetic_onset import KineticOnset
from .kinetic_patch import KineticPatch
from .kinetic_projection import KineticProjection, Projection
from .kinetic_reservoir import KineticReservoir
from .kinetic_strip_shader import KineticStripShader
from .kinetic_summary import KineticSummary
from .kinetic_velocity import KineticVelocity
from .shape_velocity import ShapeVelocity
from .temperature_bridge_shader import TemperatureBridgeShader
