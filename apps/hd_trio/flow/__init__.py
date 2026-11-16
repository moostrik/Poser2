"""hd_trio's own copy of the flow's 2D system: its motion (`GradientFlow`), bridge (`bridge_2d`) and fluid
(`Fluid2D`)."""
from .gradient_flow import GradientFlow, GradientFlowSettings
from . import bridge_2d
from .fluid_2d import Fluid2D, Fluid2DSettings
