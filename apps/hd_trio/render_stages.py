"""HD Trio show stages — per-stage composition and settings control."""

import math
from collections.abc import Callable
from typing import cast

from pytweening import *  # type: ignore

from modules.data_hub import DataHub, Stage
from modules.pose.features import MotionTime
from modules.render.layers import LayerBase
from modules.render import layers as ls

from .settings import Layers, RenderSettings, ShowStage


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

    def __init__(self, cam_id: int, data_hub: DataHub, settings: RenderSettings,
                 layers: dict[Layers, LayerBase]) -> None:
        self.cam_id = cam_id
        self.data_hub = data_hub
        self.settings = settings
        self.layers = layers

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


# ---------------------------------------------------------------------------
#  Concrete stages
# ---------------------------------------------------------------------------

class StartStage(StageLayer):

    def enter(self) -> None:
        self._start_mt = self._get_motion_time()
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = Stage.LERP

    def update(self, progress: float) -> None:
        motion_time_duration = 6.0  # movement threshold before centre pose fully visible
        progress_alpha: float = _fade_in(progress, 0.0, 1.0)
        eased_alpha: float = easeInOutSine(max(progress_alpha, self._motion_alpha(motion_time_duration)))
        self.compose([(Layers.centre_pose, eased_alpha)])

    def _motion_alpha(self, threshold: float) -> float:
        return _clamp((self._get_motion_time() - self._start_mt) / threshold)

    def _get_motion_time(self) -> float:
        pose = self.data_hub.get_pose(Stage.LERP, self.cam_id)
        if pose is None:
            return 0.0
        v = pose[MotionTime].value
        return v if not math.isnan(v) else 0.0


class IntroInStage(StageLayer):
    def enter(self) -> None:
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = Stage.LERP

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.centre_pose, 1.0),
            (Layers.intro_pose, _fade_in(progress)),
        ])


class IntroStage(StageLayer):
    def enter(self) -> None:
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = Stage.LERP

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.centre_pose, 1.0),
            (Layers.intro_pose, 1.0),
        ])


class IntroOutStage(StageLayer):
    def enter(self) -> None:
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = Stage.LERP

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.centre_pose, 1.0),
            (Layers.intro_pose, _fade_out(progress)),
        ])


class PlayInStage(StageLayer):
    def enter(self) -> None:
        self._start_mt = self._get_motion_time()
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = Stage.SMOOTH
        cast(ls.FluidLayer, self.layers[Layers.fluid]).reset()

    def update(self, progress: float) -> None:
        motion_time_duration = 2.0  # movement threshold before centre pose fully visible
        progress_alpha: float = _fade_in(progress, 0.0, 1.0)
        eased_alpha: float = easeInOutSine(max(progress_alpha, self._motion_alpha(motion_time_duration)))
        # can we store the movement time of the last stage and use it here to fade out the centre pose?
        self.compose([
            (Layers.centre_pose, 1.0 - eased_alpha),
            (Layers.fluid, eased_alpha),
            (Layers.color_mask, eased_alpha),
        ])

    def _motion_alpha(self, threshold: float) -> float:
        return _clamp((self._get_motion_time() - self._start_mt) / threshold)

    def _get_motion_time(self) -> float:
        pose = self.data_hub.get_pose(Stage.LERP, self.cam_id)
        if pose is None:
            return 0.0
        v = pose[MotionTime].value
        return v if not math.isnan(v) else 0.0


class PlayStage(StageLayer):
    def enter(self) -> None:
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = Stage.SMOOTH

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.fluid, 1.0),
            (Layers.color_mask, 1.0),
        ])


class ConclusionStage(StageLayer):
    def enter(self) -> None:
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = Stage.SMOOTH

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.fluid, 1.0),
            (Layers.color_mask, 1.0),
        ])


class IdleStage(StageLayer):
    def enter(self) -> None:
        cast(ls.CentreGeometry, self.layers[Layers.centre_geom]).config.stage = Stage.SMOOTH

    def update(self, progress: float) -> None:
        self.compose([
            (Layers.fluid, 1.0),
        ])


# ---------------------------------------------------------------------------
#  Stage registry
# ---------------------------------------------------------------------------

STAGES: dict[ShowStage, type[StageLayer]] = {
    ShowStage.START:      StartStage,
    ShowStage.INTRO_IN:   IntroInStage,
    ShowStage.INTRO:      IntroStage,
    ShowStage.INTRO_OUT:  IntroOutStage,
    ShowStage.PLAY_IN:    PlayInStage,
    ShowStage.PLAY:       PlayStage,
    ShowStage.CONCLUSION: ConclusionStage,
    ShowStage.IDLE:       IdleStage,
}
