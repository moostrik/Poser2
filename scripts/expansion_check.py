"""The expansion through the projection: after it, the water's divergence is φ, in a closed box and an open one.

The fluid at deep_flow's preset, the water at rest, a disc of full presence at z position 0.5 injected as the
expansion at `--rate` (1/s); one projection. Per wall it prints φ's sum over the water's cells (0 in a closed box,
the mean taken off), the divergence's mean, and the error, the RMS of the divergence less φ over φ's RMS. The
solve's own floor beside it: the same box, no source, the swinging body's velocity dragged in and projected
once, the RMS of the divergence after over before. The error with a source should be of the floor's order.

    python scripts/expansion_check.py [--rate 0.5] [--set path=value ...]
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from OpenGL.GL import glFinish  # noqa: E402

import monitor_harness as mh  # noqa: E402
import maccormack_check as mc  # noqa: E402
from modules.flow import Advection, Fluid3D, divergence_target  # noqa: E402

W, H = mc.W, mc.H


def disc() -> tuple[np.ndarray, np.ndarray]:
    """The presence, 1 in a disc, and its z position, 0.5 at weight 1 in it, (H, W, 4) each."""
    presence = np.zeros((H, W, 4), np.float32)
    z_position = np.zeros((H, W, 4), np.float32)
    y, x = np.mgrid[0:H, 0:W]
    inside = (x - W * 0.5) ** 2 + (y - H * 0.5) ** 2 < (W * 0.15) ** 2
    presence[inside, 0] = 1.0
    z_position[inside, 0] = 0.5
    z_position[inside, 1] = 1.0
    return presence, z_position


def divergence(fluid: Fluid3D) -> np.ndarray:
    """The faces' divergence per cell (depth, height, width), 1/s, from the faces as the solve left them."""
    glFinish()
    faces = fluid._faces.volume.read_to_numpy()
    assert faces is not None
    f = faces.astype(np.float64)
    cx, cy, cz = fluid._cell
    return ((f[:-1, :-1, 1:, 0] - f[:-1, :-1, :-1, 0]) / cx + (f[:-1, 1:, :-1, 1] - f[:-1, :-1, :-1, 1]) / cy
            + (f[1:, :-1, :-1, 2] - f[:-1, :-1, :-1, 2]) / cz)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def fluid_at(wall: str, rate: float, overrides: list[str]) -> Fluid3D:
    fluid = Fluid3D(mc.fluid_settings(Advection.SEMI_LAGRANGIAN, [f'wall={wall}', f'input.expansion_rate={rate}', *overrides]))
    fluid.allocate(2160, 3840)
    fluid.update(1.0 / mc.FPS)      # the deferred walls applied
    fluid.reset()
    glFinish()
    return fluid


def with_source(wall: str, rate: float, overrides: list[str]) -> None:
    fluid = fluid_at(wall, rate, overrides)
    presence, place = disc()
    presence_texture, place_texture = mh.texture_of(presence), mh.texture_of(place)
    try:
        fluid.add_expansion(presence_texture, 1.0, place_texture)
        glFinish()
        expansion = fluid._expansion_vol.read_to_numpy()
        assert expansion is not None
        expansion = expansion.astype(np.float64).reshape(fluid._depth, fluid._simulation_height, fluid._simulation_width, -1)[..., 0]
        phi = divergence_target(expansion, rate, closed=wall == 'CLOSED')
        fluid._enforce_incompressibility()
        div = divergence(fluid)
        print(f'{wall:<7} source  φ sum {phi.sum():+.2e}  presence {expansion.sum():8.1f} cells  divergence mean {div.mean():+.2e}  '
              f'error RMS(div − φ) / RMS(φ) {rms(div - phi) / max(rms(phi), 1e-12):.4f}', flush=True)
    finally:
        presence_texture.deallocate()
        place_texture.deallocate()
        fluid.deallocate()


def floor_without(wall: str, overrides: list[str]) -> None:
    fluid = fluid_at(wall, 0.0, overrides)
    body = mh.texture_of(mc.body(0.4, True))
    middle = np.zeros((H, W, 4), np.float32)
    middle[..., 0], middle[..., 1] = 0.5, 1.0
    place = mh.texture_of(middle)
    try:
        fluid.drag_velocity(body, 1.0, place)
        before = divergence(fluid)
        fluid._enforce_incompressibility()
        after = divergence(fluid)
        print(f'{wall:<7} floor   RMS(div after) / RMS(div before) {rms(after) / max(rms(before), 1e-12):.4f}', flush=True)
    finally:
        body.deallocate()
        place.deallocate()
        fluid.deallocate()


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rate', type=float, default=0.5)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    mh.context()
    for wall in ('CLOSED', 'OPEN'):
        with_source(wall, args.rate, args.set)
        floor_without(wall, args.set)


if __name__ == '__main__':
    main()
