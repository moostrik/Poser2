"""HD Trio show stages — per-stage composition and settings control."""

from collections.abc import Callable
from typing import cast

from pytweening import *  # type: ignore

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

    def __init__(self, settings: RenderSettings,
                 L: dict[Layers, dict[int, LayerBase]], num_cams: int) -> None:
        self.settings = settings
        self.L = L
        self.num_cams = num_cams

    def enter(self) -> None:
        """Called once when this stage becomes active."""

    def update(self, progress: float) -> None:
        """Called every frame with stage_progress in [0, 1]."""

    def exit(self) -> None:
        """Called once when leaving this stage."""

    # -- convenience ----------------------------------------------------------

    def compose(self, entries: list[tuple[Layers, float]]) -> None:
        """Compose layers into each camera's CompositeLayer.

        Args:
            entries: List of (layer_enum, opacity) pairs.
        """
        for i in range(self.num_cams):
            composite = cast(ls.CompositeLayer, self.L[Layers.composite][i])
            tex_alpha_pairs = [
                (self.L[layer][i].texture, alpha)
                for layer, alpha in entries
                if i in self.L[layer]
            ]
            composite.compose(tex_alpha_pairs)


# ---------------------------------------------------------------------------
#  Concrete stages
# ---------------------------------------------------------------------------

class StartStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([])
        # self.compose([(Layers.centre_pose, 1.0)])


class IntroInStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([
            (Layers.centre_pose, _fade_out(progress, 0.0, 0.5)),
            (Layers.intro_pose, _fade_in(progress)),
            (Layers.fluid, _fade_in(progress)),
            (Layers.color_mask, _fade_in(progress, 0.5, 1.0)),
        ])


class IntroStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([
            (Layers.centre_pose, 1.0),
            (Layers.intro_pose, 1.0),
            # (Layers.fluid, 1.0),
            # (Layers.color_mask, 1.0),
        ])


class IntroOutStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([
            (Layers.intro_pose, 1.0 - _clamp(progress)),
            (Layers.fluid, 1.0 - _clamp(progress)),
            (Layers.color_mask, 1.0 - _clamp(progress)),
        ])


class PlayInStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([])


class PlayStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([
            (Layers.fluid, 1.0),
            (Layers.color_mask, 1.0),
        ])


class ConclusionStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([
            (Layers.fluid, 1.0),
            (Layers.color_mask, 1.0),
        ])


class IdleStage(StageLayer):
    def update(self, progress: float) -> None:
        self.compose([
            (Layers.fluid, 1.0),
            (Layers.color_mask, 1.0),
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
