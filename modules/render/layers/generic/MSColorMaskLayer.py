"""MSColorMaskLayer - Cross-camera mask visualization based on similarity and motion."""

# Standard library imports
from __future__ import annotations
from dataclasses import dataclass

# Third-party imports
from OpenGL.GL import *  # type: ignore
from pytweening import *    # type: ignore
import numpy as np
from time import time
import math

# Local application imports
from modules.gl import Fbo, SwapFbo, Texture, Style
from modules.render.layers.LayerBase import LayerBase, Blit
from modules.DataHub import DataHub, Stage
from modules.pose.Frame import Frame

from modules.render.shaders.hdt.MSColorMask import MSColorMask
from modules.render.shaders.generic.AddDodgeBlend import AddDodgeBlend
from modules.render.shaders.generic.Tint import Tint

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class MSColorMaskLayerConfig:
    """Configuration for MSColorMaskLayer."""
    num_players: int = 3
    blend_mode: Style.BlendMode = Style.BlendMode.ALPHA
    similarity_threshold: float = 0.33
    motion_exponent: float = 1.5
    # AddDodgeBlend parameters
    dodge_intensity: float = 0.5
    add_curve: float = 2.0
    dodge_curve: float = 1.5
    opacity_curve: float = 0.3


class MSColorMaskLayer(LayerBase):
    """Cross-camera mask visualization with per-camera coloring pipeline.

    Pipeline:
        1. Tint own camera's mask with track color → _fbo
        2. AddDodgeBlend foreground onto tinted mask → _blend_fbo
        3. MSColorMask composites own styled result with other cameras → _fbo

    Each camera's contribution is colored exactly once.
    """

    def __init__(
        self,
        cam_id: int,
        data_hub: DataHub,
        mask_textures: dict[int, Texture],
        frg_texture: Texture,
        colors: list[tuple[float, float, float, float]],
        config: MSColorMaskLayerConfig | None = None
    ) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._mask_textures: dict[int, Texture] = mask_textures
        self._frg_texture: Texture = frg_texture
        self._colors: list[tuple[float, float, float, float]] = colors
        self.config: MSColorMaskLayerConfig = config or MSColorMaskLayerConfig()

        self._tint_fbos: list[Fbo] = [Fbo() for _ in range(self.config.num_players)]
        self._blend_fbo: SwapFbo = SwapFbo()
        self._output_fbo: Fbo = Fbo()

        # Shaders
        self._tint: Tint = Tint()
        self._add_dodge: AddDodgeBlend = AddDodgeBlend()
        self._shader: MSColorMask = MSColorMask()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    # ========== Output Access ==========

    @property
    def texture(self) -> Texture:
        """Output texture (non-premultiplied RGBA)."""
        return self._output_fbo

    # ========== Lifecycle Methods ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        fmt = internal_format if internal_format is not None else GL_RGBA16F
        self._output_fbo.allocate(width, height, fmt)
        self._blend_fbo.allocate(width, height, fmt)
        for fbo in self._tint_fbos:
            fbo.allocate(width, height, fmt)
        self._tint.allocate()
        self._add_dodge.allocate()
        self._shader.allocate()

    def deallocate(self) -> None:
        self._output_fbo.deallocate()
        self._blend_fbo.deallocate()
        for fbo in self._tint_fbos:
            fbo.deallocate()
        self._tint.deallocate()
        self._add_dodge.deallocate()
        self._shader.deallocate()

    def reset(self) -> None:
        self._output_fbo.clear()

    # ========== Processing ==========

    def update(self) -> None:
        """Update mask blend based on similarity and motion from pose data.

        Pipeline:
            Step 1: Tint own mask with track color → _fbo
            Step 2: AddDodgeBlend(tinted mask, foreground) → _blend_fbo
            Step 3: MSColorMask(own styled, other masks, colors, weights) → _fbo
        """
        # Get pose data for this camera
        pose: Frame | None = self._data_hub.get_pose(Stage.LERP, self._cam_id)
        active_poses = self._data_hub.get_pose_count(Stage.LERP)

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
        motion_similarities = similarities * motion_gates

        # Foreground blend: 0 if single person, otherwise Nth highest similarity
        lowest_similarity: float = 0.0
        if active_poses >= 2:
            sorted_similarities = np.sort(motion_similarities)  # Ascending
            lowest_similarity = float(sorted_similarities[-(active_poses - 1)])

        foreground_blend: float = (lowest_similarity - 0.25) * 2.0
        foreground_blend = max(0.0, min(1.0, foreground_blend))


        other_cam_ids: list[int] = []
        for cam_id in sorted(self._mask_textures.keys()):
            if cam_id != self._cam_id:
                other_cam_ids.append(cam_id)

        # ---- Render pipeline ----
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._add_dodge.reload()
        self._shader.reload()

        # Step 1: Tint all cameras' masks
        for cam_id, mask_tex in self._mask_textures.items():
            color = self._colors[cam_id % len(self._colors)]
            self._tint_fbos[cam_id].begin()
            self._tint.use(mask_tex, color[0], color[1], color[2], 1.0)
            self._tint_fbos[cam_id].end()

        # Step 2: AddDodgeBlend own camera's tinted mask with foreground
        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._add_dodge.use(
            self._tint_fbos[self._cam_id],
            self._frg_texture,
            foreground_blend,
            self.config.dodge_intensity,
            self.config.add_curve,
            self.config.dodge_curve,
            self.config.opacity_curve
        )
        self._blend_fbo.end()

        # Step 3: Composite - own (dodged) + others (tinted)
        styled_textures: list[Texture] = [self._blend_fbo.texture]
        weights: list[float] = [1.0]

        for cam_id in other_cam_ids:
            styled_textures.append(self._tint_fbos[cam_id])
            weights.append(float(motion_similarities[cam_id]))

        self._output_fbo.begin()
        self._shader.use(styled_textures, weights)
        self._output_fbo.end()

        Style.pop_style()

    # ========== Rendering ==========

    def draw(self) -> None:
        """Draw with configured blend mode."""
        Style.push_style()
        Style.set_blend_mode(self.config.blend_mode)
        Blit.use(self._output_fbo)
        Style.pop_style()
