"""The 3D motion's and bridge's GL shaders against their numpy references, on random fields: the largest
difference of each. The z position's passes: the motion's (`ZPosition`), the lowpass's step (`LowpassStep`), the
bandpass's trails (`BandpassTrails`) and an output's mix (`ZPositionMix`); and the lowpass's blur (`LowpassBlur`).

No app and no camera: a hidden window gives the GL context.

    python scripts/bridge_shader_check.py
"""

import sys
from pathlib import Path

import numpy as np
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_FLOAT, GL_RG, GL_RG16F, GL_RG32F, GL_RGBA, GL_RGBA16F, GL_RGBA32F, GL_TEXTURE_2D, glBindTexture, glTexImage2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.gl import Fbo, Texture  # noqa: E402
from modules.flow import (advect, blur_dispatches, gaussian_blur, gaussian_weights, trail_acceleration, velocity_trail,  # noqa: E402
                          z_position, z_position_mix, z_position_trail, ZPositionMix)
from modules.flow.bridge_3d.shaders import BandpassTrails, LowpassBlur, LowpassStep  # noqa: E402
from modules.flow.motion.shaders import ZPosition  # noqa: E402
from depth_velocity_compare import context, read_back  # noqa: E402

HALF_FLOAT: float = 2e-3    # the relative step of RGBA16F near 1, and some


def texture(array: np.ndarray, internal_format: int) -> Texture:
    """A float texture of `array` (H, W, C), C 2 or 4."""
    h, w, channels = array.shape
    result = Texture()
    result.allocate(w, h, internal_format)
    glBindTexture(GL_TEXTURE_2D, result.tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, GL_RG if channels == 2 else GL_RGBA, GL_FLOAT,
                 np.ascontiguousarray(array, dtype=np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)
    return result


def report(name: str, got: np.ndarray, expected: np.ndarray, tolerance: float = HALF_FLOAT) -> bool:
    diff = float(np.abs(got - expected).max())
    ok = diff < tolerance
    print(f"{name}: max diff {diff:.2e} {'ok' if ok else 'FAIL'}")
    return ok


def check_z_position(rng: np.random.Generator) -> bool:
    h, w = 37, 29
    disparity = np.where(rng.uniform(size=(h, w)) < 0.2, 0.0, 1.0 / rng.uniform(0.05, 2.0, (h, w)))
    mask = rng.uniform(size=(h, w))
    track = texture(np.stack((np.zeros_like(disparity), disparity, np.zeros_like(disparity), np.zeros_like(disparity)), -1), GL_RGBA32F)
    depth_mask = (rng.uniform(size=(h, w)) > 0.3).astype(np.float64)
    mask_texture = texture(np.stack((mask,) * 4, -1), GL_RGBA16F)
    depth_mask_texture = texture(np.stack((depth_mask,) * 4, -1), GL_RGBA16F)
    shader = ZPosition()
    shader.allocate()
    out = Fbo()
    out.allocate(w, h, GL_RG16F)
    passed = True
    for name, m, d in (('with both masks', mask_texture, depth_mask_texture), ('with the depth mask', None, depth_mask_texture),
                       ('without', None, None)):
        out.begin()
        shader.use(track, 0.1, 1.7, m, d)
        out.end()
        expected = z_position(disparity, 0.1, 1.7, mask if m is not None else None, depth_mask if d is not None else None)
        passed &= report(f"motion z position, {name}", read_back(out)[..., :2], expected, 4e-3)
    return passed


def check_lowpass_step(rng: np.random.Generator) -> bool:
    h, w = 45, 33
    previous = rng.uniform(-1.0, 1.0, (h, w, 4)) * [1, 1, 1, 0]
    previous_z = rng.uniform(0.0, 1.0, (h, w, 2))
    velocity = rng.uniform(-1.0, 1.0, (h, w, 4)) * [1, 1, 1, 0]
    current = rng.uniform(0.0, 1.0, (h, w, 2))
    trail_in, z_in = texture(previous, GL_RGBA16F), texture(previous_z, GL_RG32F)
    for t in (trail_in, z_in):
        t.set_wrap(GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0))
    velocity_in, current_in = texture(velocity, GL_RGBA16F), texture(current, GL_RG32F)
    trail_out, z_out, rate_out = texture(np.zeros_like(previous), GL_RGBA16F), texture(np.zeros_like(previous_z), GL_RG32F), texture(np.zeros_like(previous), GL_RGBA16F)
    shader = LowpassStep()
    shader.allocate()
    carry, attack, release, dt = (0.02, 0.02 * w / h), 0.3, 0.9, 1.0 / 60.0
    shader.use(trail_in, z_in, velocity_in, current_in, trail_out, z_out, rate_out, carry, attack, release, 1.0 / dt)
    # the advect as the shader reads it: the trail's own xy at the pixel carries both
    displacement = previous[..., :2] * np.array(carry)
    moved = advect(previous, displacement, 1.0)
    moved_z = advect(previous_z, displacement, 1.0)
    trail = velocity_trail(moved[..., :3], velocity[..., :3], attack, release)
    inner = (slice(2, -2), slice(2, -2))        # the reference clamps at the edge where the trail's border is 0
    linear = 1e-2       # a linear read between texels weighs them in 1/256 steps, on values up to 2 apart
    passed = report("lowpass step, trail", read_back(trail_out)[..., :3][inner], trail[inner], linear)
    passed &= report("lowpass step, z position", read_back(z_out)[..., :2][inner],
                     z_position_trail(moved_z, moved[..., :3], velocity[..., :3], current, attack, release)[inner], linear)
    passed &= report("lowpass step, rate", (read_back(rate_out)[..., :3] * dt)[inner], (trail_acceleration(moved[..., :3], trail, dt) * dt)[inner], linear)
    return passed


def check_bandpass_trails(rng: np.random.Generator) -> bool:
    h, w = 31, 27
    velocity = rng.uniform(-1.0, 1.0, (h, w, 4))
    fast, slow = rng.uniform(-1.0, 1.0, (h, w, 4)), rng.uniform(-1.0, 1.0, (h, w, 4))
    fast_z, slow_z, current = rng.uniform(0.0, 1.0, (h, w, 2)), rng.uniform(0.0, 1.0, (h, w, 2)), rng.uniform(0.0, 1.0, (h, w, 2))
    ins = [texture(velocity, GL_RGBA16F), texture(fast, GL_RGBA16F), texture(slow, GL_RGBA16F), texture(fast_z, GL_RG32F), texture(slow_z, GL_RG32F)]
    outs = [texture(np.zeros_like(fast), GL_RGBA16F), texture(np.zeros_like(slow), GL_RGBA16F),
            texture(np.zeros_like(fast_z), GL_RG32F), texture(np.zeros_like(slow_z), GL_RG32F)]
    shader = BandpassTrails()
    shader.allocate()
    a, b = 0.4, 0.85
    shader.use(*ins, texture(current, GL_RG32F), *outs, a, b)
    speed = np.linalg.norm(velocity[..., :3], axis=-1, keepdims=True)
    passed = report("bandpass trails, fast", read_back(outs[0]), a * fast + (1 - a) * velocity)
    passed &= report("bandpass trails, slow", read_back(outs[1]), b * slow + (1 - b) * velocity)
    passed &= report("bandpass trails, fast z position", read_back(outs[2])[..., :2], a * fast_z + (1 - a) * speed * current)
    passed &= report("bandpass trails, slow z position", read_back(outs[3])[..., :2], b * slow_z + (1 - b) * speed * current)
    return passed


def check_z_position_mix(rng: np.random.Generator) -> bool:
    h, w = 23, 19
    readings = [rng.uniform(-1.0, 1.0, (h, w, 4)) for _ in range(6)]
    trails = tuple(rng.uniform(-1.0, 1.0, (h, w, 4)) for _ in range(3))
    z_positions = tuple(rng.uniform(0.0, 1.0, (h, w, 2)) for _ in range(3))
    shares = [0.8, 0.3, 0.5, 0.0, 1.0, 0.6]
    shader = ZPositionMix()
    shader.allocate()
    out = Fbo()
    out.allocate(w, h, GL_RG32F)
    out.begin()
    shader.use([(texture(r, GL_RGBA16F), share) for r, share in zip(readings, shares)],
               tuple(texture(t, GL_RGBA16F) for t in trails), tuple(texture(z, GL_RG32F) for z in z_positions))
    out.end()
    expected = z_position_mix([r[..., :3] for r in readings], shares, tuple(t[..., :3] for t in trails), z_positions)
    return report("z position mix", read_back(out)[..., :2], expected, 2e-2)


def check_lowpass_blur(rng: np.random.Generator) -> bool:
    shader = LowpassBlur()
    shader.allocate()
    passed = True
    h, w = 61, 45      # not a multiple of the tile
    for sigma in (0.4, 1.0, 2.0, 2.6, 5.0):
        fields = [rng.uniform(-1.0, 1.0, (h, w, 4)), rng.uniform(-1.0, 1.0, (h, w, 4)), rng.uniform(0.0, 1.0, (h, w, 2))]
        formats = [GL_RGBA16F, GL_RGBA16F, GL_RG32F]
        sources = [texture(f, fmt) for f, fmt in zip(fields, formats)]
        targets = [texture(np.zeros_like(f), fmt) for f, fmt in zip(fields, formats)]
        count, each = blur_dispatches(sigma, LowpassBlur.MAX_RADIUS)
        weights = gaussian_weights(each).tolist()
        expected = [f.copy() for f in fields]
        for step in range(count):
            shader.use(sources if step % 2 == 0 else targets, targets if step % 2 == 0 else sources, weights)
            expected = [gaussian_blur(e, each) for e in expected]
        result = targets if count % 2 == 1 else sources
        for name, out, exp in zip(('first', 'second', 'third'), result, expected):
            got = read_back(out)[..., :exp.shape[2]]
            diff = float(np.abs(got - exp).max())
            ok = diff < HALF_FLOAT
            passed &= ok
            print(f"lowpass blur, sigma {sigma:.1f} px in {count} dispatch(es), {name} field: max diff {diff:.2e} {'ok' if ok else 'FAIL'}")
    return passed


def main() -> None:
    context(False, 64, 64)
    rng = np.random.default_rng(7)
    passed = check_lowpass_blur(rng)
    passed &= check_z_position(rng)
    passed &= check_lowpass_step(rng)
    passed &= check_bandpass_trails(rng)
    passed &= check_z_position_mix(rng)
    print('all ok' if passed else 'FAILED')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
