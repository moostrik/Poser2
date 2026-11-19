# The 2D bridge

Every quantity and setting here is in a meaningful unit ([FLOW.md](FLOW.md#the-2d-and-3d-flow-systems)).

The 2D bridge is the 3D bridge on a 2D velocity: it turns the motion's x/y velocity into what the 2D fluid takes,
the same readings, outputs, settings and names, read on the image plane. What each reading and output is, and why,
is in [BRIDGE_3D.md](BRIDGE_3D.md); this document holds what the 2D does differently. How the stages fit together is
in [FLOW.md](FLOW.md), what is next in superficial_flow's plan
([SUPERFICIAL_FLOW_PLAN.md](../../apps/superficial_flow/SUPERFICIAL_FLOW_PLAN.md)).

## What it is

`Bridge2D` (`bridge_2d/bridge_2d.py`) runs on the `Bridge` both bridges share (`bridge/bridge.py`): when each filter
runs, the mixes' order, the merge of a tick's frames, the ports, the timers. Its settings are `BridgeSettings`
(`bridge/bridge_settings.py`), the 3D's: `filters` (`lowpass`, `bandpass`), then `velocity` (with its `peak`),
`acceleration`, `density`, `temperature`, `expansion`. Its layer is `Bridge2DLayer`.

| Part             | 2D                   | 3D                               |
|------------------|----------------------|----------------------------------|
| lowpass          | `VelocityLowpass2D`  | `VelocityLowpass`                |
| bandpass         | `VelocityBandpass2D` | `VelocityBandpass`               |
| the mixes        | `WeightedSum2D`      | `WeightedSum`, `ZPositionMix`    |
| the peak's count | `SpeedCount2D`       | `HistogramCount` (the monitors') |

Each 2D filter has its own kernels, the 3D's twins without z: the lowpass `LowpassStep2D` and `LowpassBlur2D`, the
bandpass `BandpassTrails2D`, `BandpassChanges2D`, `BandpassAccelerations2D` and its three region passes
(`VelocityBandpassRegion2D`, `…Mean2D`, `…Speed2D`). Both share `FrameMax` with `LargerAmount`, the math
(`decay_weight`, the blur's weights, `step_peak`, `band_width`) and the peak's Python.

## How it differs from the 3D

- Every velocity and acceleration is RG16F, x and y; the 3D's is RGBA16F, x, y and z. Its kernels read and write
  RG and carry no third component, so they cost what x/y costs.
- It has no z positions: no output's `_z_position`, no trail's z position, no `ZPositionMix`.
- The mask's share in an amount needs only the mask; the 3D's needs the motion's z position too.
- The region of a start holds the moving velocity and its share in RGB of an RGBA16F texture, as RGB16F is not sure
  to be a render target.

## Checked

On the GPU, `Bridge2D` against `Bridge3D` on one velocity sequence (128 × 224, 90 steps at 50 fps, the 3D's z 0),
2026-11-19 **(seen)**: with `advect`, `attack` and `blur` on, every reading at share 0.5 in every output, the tick's
merge every seventh step and the peak on, all 22 readings and outputs x/y equal (largest difference 0) and the peak
the same (1.297 widths/s), the mask's share 0; with the mask's share 0.5, the 2D's density is the 3D's without it
plus the mask times 0.5, exactly.
