# GL plan

## Findings to look into

Samplers with no texture bound at a draw or dispatch, seen by the -sim binding probe before and after H1; no fix
designed.

- `LucasKanadeTile.estimate` (deep_flow, unit 2): at the coarsest level `estimate` is None and unit 2 gets texture 0;
  the shader reads it only under `has_estimate` (seen: `lucas_kanade_tile.py`, `.comp`). Harmless, as the motion
  thread meant it.
- `AdvectFaces3D.uPred` (deep_flow, unit 8): the MacCormack predictor, bound only when one is given; the shader reads
  it only in mode 2, the correction (`fetch_pred` and `sample_pred`, under `uMode == 2`), so the predictor pass and the
  semi-Lagrangian step leave unit 8 as it was (seen: `advect_faces_3d.py`, `.comp`). Harmless.
- `DensityColor.mask` (hd_trio, unit 1): hd_trio builds its `BridgeInput` without a mask, so unit 1 gets texture 0
  and the shader reads it only under `has_mask` (seen: `hd_trio_render.py`, `density_color.py`, `.frag`). Harmless to
  the GL; whether hd_trio's density should be gated by a player mask is hd_trio's question (open).

A probe that skips a sampler whose guard is off would show only real misses; a note, not a step.
