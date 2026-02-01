from .WindowShaderBase import WindowShaderBase


class PoseAngleWindow(WindowShaderBase):
    """Shader for visualizing angle trajectories over time.

    Displays angle values (±π range) as horizontal time series.
    Green color scheme for angle visualization.
    """
