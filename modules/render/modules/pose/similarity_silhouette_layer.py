"""SimilaritySilhouetteLayer - Cross-camera mask visualization based on similarity and motion."""

# Third-party imports
from OpenGL.GL import GL_RGBA16F
from pytweening import easeInOutSine, easeInSine
import numpy as np

# Local application imports
from modules.gl import Fbo, SwapFbo, Texture, BlendState, shaders
from modules.board import HasFrames
from modules.pose import Similarity, MotionGate, AngleMotion, Frame
from modules.settings import Field, BaseSettings
from modules.utils import Color

from ...layer_base import LayerBase
from ...color_settings import ColorSettings
from .shaders import LayerComposite, AddMix


class SimilaritySilhouetteLayerSettings(BaseSettings):
    """Configuration for SimilaritySilhouetteLayer."""
    stage:                  Field[int]   = Field(0, description="Pipeline stage for pose data")
    max_players:            Field[int]   = Field(3, access=Field.INIT)
    blend_mode:             Field[BlendState.BlendMode] = Field(BlendState.BlendMode.ALPHA)
    similarity_threshold:   Field[float] = Field(0.1, min=0.0, max=1.0, description="Similarity below this contributes nothing")
    motion_exponent:        Field[float] = Field(1.0, min=0.0, max=5.0, description="Similarity response curve")
    similarity_scale:       Field[float] = Field(1.0, min=0.0, max=1.0, description="0 = own silhouette only, 1 = full cross-camera response", access=Field.READWRITE)
    foreground_threshold:   Field[float] = Field(0.2, min=0.0, max=0.95, description="The players' lowest similarity below which the foreground does not show")
    # LayerComposite parameters
    layered:                Field[float] = Field(0.8, min=0.0, max=1.0, description="0 = additive, 1 = own in front")


class SimilaritySilhouetteLayer(LayerBase):
    """Cross-camera mask visualization with per-camera coloring pipeline.

    Pipeline:
        1. Tint own camera's mask with track color → _fbo
        2. AddMix foreground onto tinted mask → _blend_fbo
        3. LayerComposite composites own styled result with other cameras → _fbo

    Each camera's contribution is colored exactly once.
    """

    def __init__(self, cam_id: int, board: HasFrames, foreground_texture: Texture, mask_textures: dict[int, Texture], settings: SimilaritySilhouetteLayerSettings, color_settings: ColorSettings) -> None:
        self._cam_id: int = cam_id
        self._board: HasFrames = board
        self._mask_textures: dict[int, Texture] = mask_textures
        self._foreground_texture: Texture = foreground_texture
        self.settings: SimilaritySilhouetteLayerSettings = settings
        self._color_settings: ColorSettings = color_settings

        self._tint_fbos: list[Fbo] = [Fbo() for _ in range(self.settings.max_players)]
        self._blend_fbo: SwapFbo = SwapFbo()
        self._output_fbo: Fbo = Fbo()

        # Shaders
        self._color_fill: shaders.ColorFill = shaders.ColorFill()
        self._add_mix: AddMix = AddMix()
        self._shader: LayerComposite = LayerComposite()

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
        self._color_fill.allocate()
        self._add_mix.allocate()
        self._shader.allocate()

    def deallocate(self) -> None:
        self._output_fbo.deallocate()
        self._blend_fbo.deallocate()
        for fbo in self._tint_fbos:
            fbo.deallocate()
        self._color_fill.deallocate()
        self._add_mix.deallocate()
        self._shader.deallocate()

    def reset(self) -> None:
        self._output_fbo.clear()

    # ========== Processing ==========

    def update(self) -> None:
        """Update mask blend based on similarity and motion from pose data.

        Pipeline:
            Step 1: Tint own mask with track color → _fbo
            Step 2: AddMix(tinted mask, foreground) → _blend_fbo
            Step 3: LayerComposite(own styled, other masks, colors, weights) → _fbo
        """
        # Get pose data for this camera
        pose: Frame | None = self._board.get_frame(self.settings.stage, self._cam_id)
        active_poses = len(self._board.get_frames(self.settings.stage))

        # Extract similarity and motion data
        max_players = self.settings.max_players
        similarities: np.ndarray = np.nan_to_num(pose[Similarity].values) if pose is not None else np.zeros(max_players, dtype=np.float32)
        motion_gates: np.ndarray = pose[MotionGate].values if pose is not None else np.zeros(max_players, dtype=np.float32)
        motion: float = pose[AngleMotion].value if pose is not None else 0.0
        motion = easeInOutSine(motion)

        # Apply similarity threshold and exponent
        threshold = self.settings.similarity_threshold
        exponent = self.settings.motion_exponent
        similarities = np.clip((similarities - threshold) / (1.0 - threshold), 0.0, 1.0)
        similarities = np.power(similarities, exponent)
        motion_similarities = similarities * self.settings.similarity_scale

        # Foreground blend: 0 if single person, otherwise Nth highest similarity
        lowest_similarity: float = 0.0
        if active_poses >= 2:
            sorted_similarities = np.sort(motion_similarities)  # Ascending
            lowest_similarity = float(sorted_similarities[-(active_poses - 1)])

        foreground_threshold = self.settings.foreground_threshold
        foreground_blend: float = (lowest_similarity - foreground_threshold) / (1.0 - foreground_threshold)
        foreground_blend = max(0.0, min(1.0, foreground_blend))
        foreground_blend = easeInSine(foreground_blend)

        other_cam_ids: list[int] = []
        for cam_id in sorted(self._mask_textures.keys()):
            if cam_id != self._cam_id:
                other_cam_ids.append(cam_id)

        # ---- Render pipeline ----
        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)


        # Step 1: Tint all cameras' masks
        colors = self._color_settings.track_colors
        for cam_id, mask_tex in self._mask_textures.items():
            color = colors[cam_id % len(colors)]
            self._tint_fbos[cam_id].begin()
            self._color_fill.use(mask_tex, Color(color.r, color.g, color.b))
            self._tint_fbos[cam_id].end()

        # Step 2: AddMix own camera's tinted mask with foreground
        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._add_mix.use(
            self._tint_fbos[self._cam_id],
            self._foreground_texture,
            foreground_blend,
        )
        self._blend_fbo.end()

        # Step 3: Composite - own (dodged) + others (tinted)
        styled_textures: list[Texture] = [self._blend_fbo]
        weights: list[float] = [motion]

        for cam_id in other_cam_ids:
            styled_textures.append(self._tint_fbos[cam_id])
            weights.append(float(motion_similarities[cam_id]))

        self._output_fbo.begin()
        self._shader.use(styled_textures, weights, self.settings.layered)
        self._output_fbo.end()

        BlendState.pop()

    # ========== Rendering ==========

    def draw(self) -> None:
        """Draw with configured blend mode."""
        BlendState.push()
        BlendState.set_blend_mode(self.settings.blend_mode)
        self._output_fbo.draw()
        BlendState.pop()
