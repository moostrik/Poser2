"""HD Trio show stages — per-stage composition and settings control."""

import math
from collections.abc import Callable
from typing import cast

from pytweening import *  # type: ignore

from modules.board import HasFrames
from modules.pose.features import MotionTime
from modules.render.layers import LayerBase
from modules.render import layers as ls

from .settings import Layers, RenderSettings, ShowStage, Stage


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _lerp(a: float, b: float, t: float, ease: Callable[[float], float] = linear) -> float:
    return a + (b - a) * ease(t)


def _fade_in(progress: float, start: float = 0.0, end: float = 1.0,
             ease: Callable[[float], float] = linear) -> float:
    """Ramp 0→1 over [start, end] of stage progress."""
    if end == start:
        return 1.0 if progress >= start else 0.0
    return ease(_clamp((progress - start) / (end - start)))


def _fade_out(progress: float, start: float = 0.0, end: float = 1.0,
              ease: Callable[[float], float] = linear) -> float:
    """Ramp 1→0 over [start, end] of stage progress."""
    return 1.0 - _fade_in(progress, start, end, ease)


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------

class StageLayer:
    """Base for per-stage orchestration — settings control + composition."""

    def __init__(self, cam_id: int, board: HasFrames, settings: RenderSettings,
                 layers: dict[Layers, LayerBase]) -> None:
        self.cam_id = cam_id
        self.board: HasFrames = board
        self.settings = settings
        self.layers = layers
        self._start_mt: float = 0.0

    def enter(self) -> None:
        """Called once when this stage becomes active."""

    def update(self, progress: float) -> None:
        """Called every frame with stage_progress in [0, 1]."""

    def exit(self) -> None:
        """Called once when leaving this stage."""

    # -- convenience ----------------------------------------------------------

    def compose(self, entries: list[tuple[Layers, float]]) -> None:
        """Compose layers into this camera's CompositeLayer.

        Args:
            entries: List of (layer_enum, opacity) pairs.
        """
        composite = cast(ls.CompositeLayer, self.layers[Layers.composite])
        composite.compose([
            (self.layers[layer].texture, alpha)
            for layer, alpha in entries
            if layer in self.layers
        ])

    def set_geom_stage(self, stage: Stage) -> None:
        """Select which pipeline stage the centre geometry reads poses from."""
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = stage

    def set_similarity_scale(self, value: float) -> None:
        """Set how strongly other cameras contribute to the fluid and the colour mask.

        Shared from RenderSettings, so both layers follow one write. Settings values
        persist across stages, so every stage states its own.
        """
        self.settings.similarity_scale = value

    # -- motion time ----------------------------------------------------------

    def mark_motion_time(self) -> None:
        """Snapshot MotionTime as the zero point for `_motion_alpha`."""
        self._start_mt = self._get_motion_time()

    def _motion_alpha(self, threshold: float) -> float:
        """Fraction of `threshold` MotionTime accumulated since enter()."""
        return _clamp((self._get_motion_time() - self._start_mt) / threshold)

    def _get_motion_time(self) -> float:
        pose = self.board.get_frame(Stage.LERP, self.cam_id)
        if pose is None:
            return 0.0
        v = pose[MotionTime].value
        return v if not math.isnan(v) else 0.0


# ---------------------------------------------------------------------------
#  Concrete stages
# ---------------------------------------------------------------------------

class WelcomeInStage(StageLayer):
    """Attract state fades away, the centre pose appears."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.LERP)
        self.set_similarity_scale(0.0)

    def update(self, progress: float) -> None:
        out_alpha = _fade_out(progress)
        self.compose([
            (Layers.centre_pose, easeInOutSine(_fade_in(progress))),
            (Layers.fluid, out_alpha),
            (Layers.color_mask, out_alpha),
        ])


class WelcomeStage(StageLayer):
    """Centre pose alone, while the welcome is spoken."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.LERP)
        self.set_similarity_scale(0.0)

    def update(self, progress: float) -> None:
        self.compose([(Layers.centre_pose, 1.0)])


class MovementStage(StageLayer):
    """"Move and a silhouette appears" — the mask opens up as the visitor moves."""

    def enter(self) -> None:
        self.mark_motion_time()
        self.set_geom_stage(Stage.LERP)
        self.set_similarity_scale(0.0)

    def update(self, progress: float) -> None:
        # MotionTime accumulated since enter() before the mask envelope is fully open.
        # Not wall-clock seconds and not the sequencer's stage duration — a separate
        # "visitor has moved enough" threshold, so the fade completes on whichever
        # comes first: the stage running out or the visitor moving.
        #
        # Opening on progress alone reveals nothing: MSColorMaskLayer already weights
        # the own-camera slot by AngleMotion, so a motionless visitor stays dark at
        # any alpha. The max() only guarantees no pop into WHITE_POSE.
        motion_time_duration = 6.0
        progress_alpha: float = _fade_in(progress, 0.0, 1.0)
        mask_alpha: float = easeInOutSine(max(progress_alpha, self._motion_alpha(motion_time_duration)))
        self.compose([
            (Layers.centre_pose, 1.0),
            (Layers.color_mask, mask_alpha),
        ])


class WhitePoseStage(StageLayer):
    """Brings in the white example figure. Intro playback starts here (see render.py)."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.LERP)
        self.set_similarity_scale(0.0)

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.centre_pose, 1.0),
            (Layers.color_mask, 1.0),
            (Layers.intro_pose, _fade_in(progress)),
        ])


class FluidStage(StageLayer):
    """Brings in the fluid, alongside the white example figure."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.LERP)
        self.set_similarity_scale(0.0)
        cast(ls.FluidLayer, self.layers[Layers.fluid]).reset()

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.centre_pose, 1.0),
            (Layers.color_mask, 1.0),
            (Layers.fluid, easeOutSine(_fade_in(progress))),
            (Layers.intro_pose, 1.0),
        ])


class PracticeStage(StageLayer):
    """Everything on — the visitor copies the white example."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.LERP)
        self.set_similarity_scale(0.0)

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.centre_pose, 1.0),
            (Layers.color_mask, 1.0),
            (Layers.fluid, 1.0),
            (Layers.intro_pose, 1.0),
        ])


class EnjoyInStage(StageLayer):
    """Guidance withdraws — centre pose and white example fade out."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.SMOOTH)
        self.set_similarity_scale(0.0)

    def update(self, progress: float) -> None:
        progress_alpha: float = _fade_in(progress, 0.0, 1.0)
        out_alpha = easeOutQuad(1.0 - progress_alpha)
        self.compose([
            (Layers.centre_pose, out_alpha),
            (Layers.color_mask, 1.0),
            (Layers.fluid, 1.0),
            (Layers.intro_pose, out_alpha),
        ])


class EnjoyStage(StageLayer):
    """The cross-camera response reveals itself — moving in sync starts to show."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.SMOOTH)
        # enter() runs before the layer updates and update() after, so the held
        # value is stated here too — otherwise the transition frame would still
        # carry the previous stage's scale.
        self.set_similarity_scale(0.0)

    def update(self, progress: float) -> None:
        self.set_similarity_scale(easeInOutSine(_fade_in(progress, 0.0, 1.0)))
        self.compose([
            (Layers.fluid, 1.0),
            (Layers.color_mask, 1.0),
        ])


class PlayStage(StageLayer):
    """Free play — fluid and mask at full cross-camera response."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.SMOOTH)
        self.set_similarity_scale(1.0)

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.fluid, 1.0),
            (Layers.color_mask, 1.0),
        ])


class ConclusionStage(StageLayer):
    def enter(self) -> None:
        self.set_geom_stage(Stage.SMOOTH)
        self.set_similarity_scale(1.0)

    def update(self, progress: float) -> None:

        alpha = _fade_in(progress, 1.0, 0.0)
        fluid_alpha = easeOutSine(alpha)
        mask_alpha = easeInQuint(alpha)

        self.compose([
            (Layers.fluid, fluid_alpha),
            (Layers.color_mask, mask_alpha),
        ])


class BlackoutStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([])


class IdleStage(StageLayer):
    """Attract state — parked here whenever the show is stopped."""

    def enter(self) -> None:
        self.set_geom_stage(Stage.SMOOTH)
        self.set_similarity_scale(1.0)

    def update(self, progress: float) -> None:
        alpha = easeInOutSine(_fade_in(progress, 0.0, 1.0))

        self.compose([
            (Layers.fluid, alpha),
            (Layers.color_mask, alpha),
        ])


# ---------------------------------------------------------------------------
#  Stage registry
# ---------------------------------------------------------------------------

# Must stay total over ShowStage — render.py falls back to composing nothing
# for a stage with no entry here.
STAGES: dict[ShowStage, type[StageLayer]] = {
    ShowStage.WELCOME_IN: WelcomeInStage,
    ShowStage.WELCOME:    WelcomeStage,
    ShowStage.MOVEMENT:   MovementStage,
    ShowStage.WHITE_POSE: WhitePoseStage,
    ShowStage.FLUID:      FluidStage,
    ShowStage.PRACTICE:   PracticeStage,
    ShowStage.ENJOY_IN:   EnjoyInStage,
    ShowStage.ENJOY:      EnjoyStage,
    ShowStage.PLAY:       PlayStage,
    ShowStage.CONCLUSION: ConclusionStage,
    ShowStage.BLACKOUT:   BlackoutStage,
    ShowStage.IDLE:       IdleStage,
}
