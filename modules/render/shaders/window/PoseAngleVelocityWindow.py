from .WindowShaderBase import WindowShaderBase


class PoseAngleVelocityWindow(WindowShaderBase):
    """Shader for visualizing angular velocity trajectories over time.

    Displays angular velocity (±π range) as horizontal time series.
    Orange/cyan alternating color scheme for velocity visualization.
    """
