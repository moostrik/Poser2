"""The bridge's mix (numpy reference of weighted_sum.frag)."""

import numpy as np


def weighted_sum(sources: list[np.ndarray], weights: list[float], lengths: bool = False, capped: bool = False) -> np.ndarray:
    """Each source (..., 3, or ..., 1 for a scalar such as the depth mask) times its weight, summed (..., 3);
    with `lengths`, each source's length times its weight, for an amount; with `capped`, held at 1, a presence.
    A source at weight 0 adds nothing. Mirrors weighted_sum.frag."""
    shape: tuple[int, ...] = sources[0].shape[:-1] + (3,)
    total: np.ndarray = np.zeros(shape)
    for source, weight in zip(sources, weights):
        if weight == 0.0:
            continue
        value: np.ndarray = np.zeros(shape)
        value[..., :source.shape[-1]] = source
        total = total + (np.linalg.norm(value, axis=-1, keepdims=True) if lengths else value) * weight
    return np.minimum(total, 1.0) if capped else total
