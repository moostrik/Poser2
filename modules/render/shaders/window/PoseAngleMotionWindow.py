from .WindowShaderBase import WindowShaderBase


class PoseAngleMotionWindow(WindowShaderBase):
    """Shader for visualizing angular motion magnitude over time.

    Displays motion intensity [0, 3] range as horizontal time series.
    Red-orange color scheme for motion visualization.
    """
