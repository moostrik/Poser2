"""The pressure's reach: `pressure.reach`, a share from 0 to 1, as the screened Poisson equation's term, shared by the
2D and the 3D fluid. How and why: FLUID_3D.md's "The pressure"."""


def screening(share: float, longest: float) -> float:
    """The screened equation's 1 / reach² (1/widths²) for the reach `share` s of a field whose longest side is
    `longest` widths: reach = longest × s / (1 − s), so 1 (or more) is everywhere, 0; s above 0 (the panel's log slider
    from 0.001); no pressure at all is `pressure.enabled` off, which the caller skips."""
    if share >= 1.0:
        return 0.0
    ratio: float = (1.0 - share) / (longest * share)
    return ratio * ratio
