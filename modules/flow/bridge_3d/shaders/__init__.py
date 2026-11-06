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
from .kinetic_velocity import KineticVelocity
from .scale_velocity import ScaleVelocity
from .velocity_event import Event, VelocityEvent
from .velocity_event_region import VelocityEventRegion
from .velocity_event_region_mean import VelocityEventRegionMean
from .velocity_trail import VelocityTrail
from .temperature_bridge_shader import TemperatureBridgeShader
