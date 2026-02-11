"""MSColorMaskLayer - Cross-camera mask visualization based on similarity and motion."""

# Standard library imports
from __future__ import annotations
from dataclasses import dataclass

# Third-party imports
from OpenGL.GL import *  # type: ignore
from pytweening import *    # type: ignore
import numpy as np

# Local application imports
from modules.gl import Fbo, Texture, Style
from modules.render.layers.LayerBase import LayerBase, Blit
from modules.DataHub import DataHub, Stage
from modules.pose.Frame import Frame

from modules.render.shaders.hdt.MSColorMask import MSColorMask

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class MSColorMaskLayerConfig:
    """Configuration for MSColorMaskLayer."""
    num_players: int = 3
    blend_mode: Style.BlendMode = Style.BlendMode.ALPHA
    similarity_threshold: float = 0.33
    motion_exponent: float = 1.5


class MSColorMaskLayer(LayerBase):
    """Cross-camera mask visualization based on similarity and motion.

    Aggregates masks from all cameras and blends them based on:
    - Own camera: weighted by angle_motion.value
    - Other cameras: weighted by similarity * motion_gate

    Outputs non-premultiplied RGBA where each camera's mask is colorized
    with its assigned track color.

    Usage:
        # In RenderManager, after creating all CentreMaskLayers:
        mask_layers = {i: centre_mask_layer[i] for i in range(num_cams)}
        sync_mask = MSColorMaskLayer(cam_id, data_hub, mask_layers, colors, config)

        # Each frame:
        sync_mask.update()
        output_texture = sync_mask.texture
    """

    def __init__(
        self,
        cam_id: int,
        data_hub: DataHub,
        mask_textures: dict[int, Texture],
        colors: list[tuple[float, float, float, float]],
        config: MSColorMaskLayerConfig | None = None
    ) -> None:
        """Initialize MSColorMaskLayer.

        Args:
            cam_id: Camera ID for this layer's output
            data_hub: Data hub for pose data (similarity, motion)
            mask_textures: Dict of mask textures (cam_id -> Texture)
            colors: Per-camera colors (passed from RenderManager)
            config: Layer configuration
        """
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._mask_textures: dict[int, Texture] = mask_textures
        self._colors: list[tuple[float, float, float, float]] = colors
        self.config: MSColorMaskLayerConfig = config or MSColorMaskLayerConfig()

        self._fbo: Fbo = Fbo()
        self._shader: MSColorMask = MSColorMask()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Output texture (non-premultiplied RGBA)."""
        return self._fbo.texture

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate output buffer.

        Args:
            width: Output width
            height: Output height
            internal_format: GL format for output (e.g., GL_RGBA16F)
        """
        fmt = internal_format if internal_format is not None else GL_RGBA16F
        self._fbo.allocate(width, height, fmt)
        self._shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._fbo.deallocate()
        self._shader.deallocate()

    def reset(self) -> None:
        """Reset layer state."""
        self._fbo.clear()

    # ========== Processing ==========

    def update(self) -> None:
        """Update mask blend based on similarity and motion from pose data."""
        # Get pose data for this camera
        pose: Frame | None = self._data_hub.get_pose(Stage.LERP, self._cam_id)

        # Extract similarity and motion data
        num_players = self.config.num_players
        similarities: np.ndarray = pose.similarity.values if pose is not None else np.zeros(num_players, dtype=np.float32)
        motion_gates: np.ndarray = pose.motion_gate.values if pose is not None else np.zeros(num_players, dtype=np.float32)
        motion: float = pose.angle_motion.value if pose is not None else 0.0
        motion = easeInOutSine(motion)

        # Apply similarity threshold and exponent
        threshold = self.config.similarity_threshold
        exponent = self.config.motion_exponent
        similarities = np.clip((similarities - threshold) / (1.0 - threshold), 0.0, 1.0)
        similarities = np.power(similarities, exponent)

        # Calculate weights for each camera
        weights: list[float] = []
        masks: list[Texture] = []
        colors: list[tuple[float, float, float, float]] = []

        for cam_id in sorted(self._mask_textures.keys()):
            mask_tex = self._mask_textures[cam_id]
            if not mask_tex.allocated:
                continue

            masks.append(mask_tex)
            colors.append(self._colors[cam_id % len(self._colors)])

            if cam_id == self._cam_id:
                # Own camera: weight by motion
                weights.append(motion)
            else:
                # Other cameras: weight by similarity * motion_gate
                weight = float(similarities[cam_id] * motion_gates[cam_id])
                weights.append(weight)

        # Render blended output
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._fbo.begin()
        self._shader.use(masks, colors, weights)
        self._fbo.end()

        Style.pop_style()

    # ========== Rendering ==========

    def draw(self) -> None:
        """Draw with configured blend mode."""
        Style.push_style()
        Style.set_blend_mode(self.config.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        Style.pop_style()
