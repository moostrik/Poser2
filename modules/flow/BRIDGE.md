# The 3D bridge

Every quantity and setting here is in a meaningful unit ([FLOW.md](FLOW.md#the-2d-and-3d-flow-systems)).

The bridge turns the motion's velocity into what the fluid takes, in units of the body: how fast it
moves, the dye it brings, its warmth. How the stages fit together is in [FLOW.md](FLOW.md), what
the fluid does with each output in [FLUID_3D.md](FLUID_3D.md#the-inputs), what is open in
[FLOW.md's Open](FLOW.md#open). "Code" is what the files show; **(deduction)** marks what is
inferred from them, **(site fact)** what the operator saw. The 2D bridge (`bridge_2d/`) gives the
boundary only and stays as on `main` until it is rebuilt on the Lucas–Kanade flow.

## The filters

The bridge reads two things from a movement: the motion itself, a velocity in its direction that
is stronger for fast than for slow and coasts after a stop; and its changes, the velocity changes
where a move starts and where it stops. The motion wants a long, soft reading, the changes a
short, sharp one, so two filters, named by what they do, read the same velocity side by side
(`bridge_3d/`):

| Filter             | Reads                                                                  | Gives                                    | Panel group         |
|--------------------|------------------------------------------------------------------------|------------------------------------------|---------------------|
| `VelocityLowpass`  | the flow's velocity through one trail: advect, attack/smooth, blur     | `trail`                                  | `readings.lowpass`  |
| `VelocityBandpass` | the flow's velocity through a fast and a slow trail, and their regions | `change`, `start`, `bow`, `stop`, `wake` | `readings.bandpass` |

A reading is what a filter makes from the motion. Each of the six comes as a velocity (widths/s)
and as an acceleration (widths/s²):

| Reading                        | As a velocity, widths/s                                                     | As an acceleration, widths/s²                                                                   |
|--------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `trail`                        | the motion, held and smoothed                                               | the trail's rate, its Ema step over dt: (velocity − trail) / τ for a short frame, τ the `attack` where the speed rises, else the `smooth`; a spike at a start, a brake over the coast at a stop; advected and blurred as the trail is |
| `change`                       | (fast − slow) / `step_peak`: the whole change of speed, signed, before the direction and the regions | (fast − slow) over the band's width: the acceleration itself, signed, braking at a stop |
| `start`, `bow`, `stop`, `wake` | the change on a direction and split by the region, at its height: the speed that started or stopped | the same pulse over the band's width instead of `step_peak` |

The band's width is what (fast − slow) adds up to over time, × dt, per unit of a change of speed:
(`slow_smooth` − `fast_smooth`) for a short frame, and for the two Emas as stepped,
dt × (w_slow / (1 − w_slow) − w_fast / (1 − w_fast)) (`band_width`), which stays exact where a frame
is longer than the fast trail (25 fps against 25 ms: 3.4% short with the plain difference). So each
acceleration × dt adds its Δv once, whatever the frame rate and the trails' times, for a sudden or
a gradual change; the velocity's height reads the Δv only for a sudden one. The trail's rate adds
up to the trail's change the same way (`modules/flow/tests/test_bridge_accelerations.py`). A filter
writes its accelerations only while an output takes one or a monitor shows one of its ports.

Every reading's acceleration integrates to the Δv over a change; they differ in their window: the
lowpass's `attack` and `smooth`, the bandpass's `fast_smooth` and `slow_smooth`.

The fluid makes its own swirl from what it is given, so the bridge gives no curl of its own.

### The trail

One unit-gain Ema per pixel: a rising 3D speed follows over `attack`, a falling one over `smooth`,
so a flick reaches the fluid at full strength and a stop coasts. Both filters treat x, y and z
alike: z is a little noisier than x/y, and its one time constant is the motion's (the depth
velocity's `smooth`), which every reader of z then shares, as a z time per filter would add the
same lag in more places. `advect` moves the trail with its own motion before each step, so the
average follows a moving limb instead of a fixed pixel. `blur_radius` and `blur_steps` spread it in
space; the ring it leaves around the body is part of the look **(site fact)**.

### The changes

The difference of the fast and the slow trail along the motion's direction rises where a move
starts and falls where it stops; it lasts about `slow_smooth`, as the slow trail catches up. It is
divided by the trails' step peak (`step_peak`, 0.58 at 25 and 150 ms), so a limb at speed v that
starts or stops at once reads v: the changes are in the motion's own unit, image widths per
second, a velocity change Δv, and a gradual start or stop reads less, by how much of its speed
changes within the trails' times. A pixel cannot tell a move starting from a limb arriving at it,
nor a move stopping from a limb leaving it: its speed rises or falls either way. The region around
it can, so the region splits each pixel's rise and fall in two, by the region weight
(`start_region_weight`, `stop_region_weight`; 0 gives all of it to the start and the stop):

| Part   | Is                                               | Along                 |
|--------|--------------------------------------------------|-----------------------|
| start  | the share of a rise its region shares: a part of the body speeding up | the fast trail's direction |
| bow    | the rest of the rise: a limb arriving, the wave in front of it        | the fast trail's direction |
| stop   | the share of a fall its region shares: a part of the body slowing down | the slow trail's direction, the motion it ended |
| wake   | the rest of the fall: a limb leaving, behind it                        | the slow trail's direction |

Start + bow is a pixel's whole rise, stop + wake its whole fall. What a region shares:

- A rise reads the region's mean velocity over its moving pixels (those the motion gives a
  velocity), taken from the flow and then smoothed per region by the fast and the slow trail.
  A limb arriving adds moving pixels at its own speed and leaves that mean as it was; a limb
  speeding up raises it; jitter on a still body cancels in it. Two other readings fail: over all
  of a region's pixels an arriving limb raises the mean like a speedup, and taken of the two
  trails the slow trail's lag at the pixels a limb arrives at does too.
- A fall reads the region's mean speed of the two trails, which jitter never lowers and a turning
  limb keeps, where a mean velocity would shrink and read as a stop. The trails lag at the pixels
  a limb leaves, so a leaving limb lowers it too: part of a wake reads as a stop.

The changes read x, y and z alike. The bandpass filters in time only: noise within a frame is the
motion's to filter, and the trail's blur is a look. A change takes no hold: it lasts about
`slow_smooth` by itself, and what a stop injects is the speed it ended, as a hold would lift steady
motion as much as the stops. No blur and no advect: the changes stay sharp. The bandpass runs
while an output takes one of its readings or a monitor shows one of its ports.

### The pulse's area

The bandpass's pulse is the fast lowpass minus the slow one, two unit-gain Emas of the velocity.
An Ema lags its input by its time constant times the change, so for any change of speed, sudden
or gradual, the integral of their difference is (`slow_smooth` − `fast_smooth`) × Δv, 0.125 s at
25 and 150 ms; the height reads Δv only for a sudden change (`step_peak`). Dividing the pulse by
that time makes it an acceleration whose integral is the Δv, so × dt in the fluid adds the Δv
exactly once at any frame rate, with no history and no number from the bridge in the fluid. The
motion's `smooth` on z is one more linear filter in the chain, so the areas hold for z too; the
heights read z low by `smooth` against `fast_smooth`. The bandpass divides by the band's width at
the frame's dt and the lowpass uses the Ema's real step, the same integral, exact per frame.

## The mix

What the fluid takes is two systems, the two a solver tells apart, each mixing the six readings,
each reading at a share, 0 to 1, as the light renderer mixes its layers (`Bridge3DLayer`,
`WeightedSum`). The panel shows three subpanels: `readings` (`lowpass`, `bandpass`), `boundary` and
`body_force`. Each output's group holds the six shares as knobs on the first row (`MixSettings`:
`trail`, `change`, `start`, `bow`, `stop`, `wake`), and a density's own settings on the second
(`saturation`, `brightness`). The density and the temperature are colored by a density and a
temperature bridge per side, with the same color.

| Subpanel     | Groups                                   | Readings enter as                                       | Is                                                   |
|--------------|------------------------------------------|---------------------------------------------------------|------------------------------------------------------|
| `readings`   | `lowpass`, `bandpass`                    |                                                         | the filters' settings                                |
| `boundary`   | `velocity`, `density`, `temperature`     | velocities                                              | the body in the water: what the water is dragged toward |
| `body_force` | `acceleration`, `density`, `temperature` | accelerations; the density and the temperature from velocities, a push's change of speed | the Navier–Stokes force term: what is added          |

The six outputs (`BridgeOutput`, `Injection`), each in a unit of the body:

| Output                    | Unit                                                                                                     | Default       |
|---------------------------|----------------------------------------------------------------------------------------------------------|---------------|
| `boundary_velocity`       | widths/s, the body's velocity                                                                            | the trail 1   |
| `boundary_density`        | the dye, RGBA: the color times its presence, the presence as alpha, 0 to 1                               | the trail 1   |
| `boundary_temperature`    | the warmth, −1 cold to 1 hot, its presence folded in                                                      | the trail 1   |
| `body_force_acceleration` | widths/s², a Δv added once over a change                                                                 | all 0: `None` |
| `body_force_density`      | a push's dye, its presence from the change of speed                                                       | all 0: `None` |
| `body_force_temperature`  | a push's warmth, the same way                                                                             | all 0: `None` |

The presence is the mix's speed in widths/s, the sum of its readings' sizes (the length of each, at
its share), capped at 1: a part moving at 1 width/s or faster is fully there, a slower one partly,
and a still body leaves no dye. Widths/s is the velocity's own unit, so no reference speed is chosen.
The density's color is the picture's (`DensityColor`: the image, or white, times the tint, gated by
the mask) through `saturation` and `brightness`; the warmth is (warm − cold) × (1 − neutral × 0.5) ×
the presence, from the same color's red, blue and green. The readings of one unit share the mix, so
the mix has no gain: the motion's gains balance the axes and the fluid's gains set how the water
follows; a weight is a share. A body force output is `None` while its shares are all 0. The mixes'
ports are `boundary.velocity`, `boundary.density_amount`, `boundary.temperature_amount`,
`body_force.acceleration`, `body_force.density_amount` and `body_force.temperature_amount`; the
colored ones are the density and temperature bridges' own (`boundary_density.output`, ...).

A wake may drag (its share in `boundary.velocity`: the water behind a leaving limb is pulled to the
speed it left with, and let go when the pulse dies), kick (its share in `body_force.acceleration`),
leave dye where it is or push it in, or any of these at once; the same reading in two outputs is a
choice, not an error. The change beside its parts lets a preset take the physics (the true
acceleration, braking at stops) or the look (a stop that pushes).

## Measured

`scripts/bridge_compare.py` on the standmoves recording (`20261103-215324`: four whole-arm moves
from rest, with starts and stops, a player at 0.65–1.5 m), its frames as captured on 2026-11-05
(`temp/depth_track_1105/`, 1449 frames, 25 fps, the stamps the replay's decode times), the flow run
again on them with deep_flow's preset flow of 2026-11-06 (levels 4, same_surface and xy_smooth on,
the depth velocity's `smooth` 50 ms unless named), each bridge at its preset values (trail: smooth
1000 ms, attack 1 ms, advect 1, blur 4 px × 4; change: 25 / 150 ms, both region weights 1,
regions 128 px), stepped once per flow frame. Per frame, the mean over the body's pixels that move
in the loop; noise is the still part's p99 per pixel; slow and fast are the steady halves of the
moves; stop ÷ fast is the stop window's peak over steady fast motion, start ÷ fast the start
window's.

The depth velocity's `smooth`, with both bridges treating z like x/y:

| `smooth` | Flow z noise | Flow z, fast | Trail z noise | Trail z, fast | Start noise | Stop ÷ fast | Start share at starts / steady |
|----------|--------------|--------------|---------------|---------------|-------------|-------------|--------------------------------|
| 50 ms    | 0.160        | 0.093        | 0.243         | 0.387         | 0.079       | 4.07        | 0.34 / 0.37                    |
| 100 ms   | 0.012        | 0.079        | 0.066         | 0.329         | 0.000       | 4.24        | 0.38 / 0.36                    |
| 200 ms   | 0.000        | 0.047        | 0.005         | 0.171         | 0.000       | 10.41       | 0.42 / 0.30                    |
| 400 ms   | 0.000        | 0.020        | 0.000         | 0.057         | 0.000       | 10.88       | 0.48 / 0.24                    |

The trail follows a rising speed at once, so it keeps z's upward noise for about `smooth`: it needs
a calm z. At 100 ms the flow's z keeps 85% of its motion and loses 93% of its noise; at 200 ms its
noise is gone and half its motion is left; at 400 ms little of either. The start's region only
tells a start from a limb arriving once z is calm: its share at real starts against steady motion
grows from equal at 50 ms to 0.42 against 0.30 at 200 ms.

The stop's region, `smooth` 50 ms:

| Stop reading                                   | Noise | Stop ÷ fast | Falls in steady motion read as stop | At real stops |
|------------------------------------------------|-------|-------------|-------------------------------------|---------------|
| region weight 0                                | 0.040 | 2.46        | 1.00                                | 1.00          |
| region 64 px                                   | 0.019 | 3.57        | 0.44                                | 0.63          |
| region 128 px (the preset)                     | 0.010 | 4.07        | 0.38                                | 0.62          |
| region 256 px                                  | 0.008 | 4.31        | 0.36                                | 0.61          |
| region 128 px over the flow's moving pixels    | 0.011 | 3.48        | 0.45                                | 0.63          |

A region over the flow's moving pixels, as the start's, reads more of a leaving limb as a stop and
the real stops the same, so the stop's region reads the trails' mean speed. Without the region the
stop's noise is four times higher.

| GPU, each filter alone | Mean, ms | p95, ms |
|------------------------|----------|---------|
| the lowpass            | 1.55     | 2.13    |
| the bandpass           | 1.88     | 2.38    |

Advice, from these numbers on this clip:

| Setting                       | Value  | Why |
|-------------------------------|--------|-----|
| the depth velocity's `smooth` | 200 ms | z calm for the trail's attack and the start's region, at +150 ms of z's lag; 100 ms for a livelier z at some noise |
| `stop_region_size`            | 128–256 px | 256 a little cleaner (noise 0.008, 4.31×), 64 worse |
| `stop_region_weight`          | 1      | without it the stop's noise is four times higher and its stops half as distinct |

The clip moves whole arms from rest, so a limb arriving and a part speeding up come together: the
start and bow split, `start_region_*`, `fast_smooth` and z's changes toward the camera are not
tested on it, and the lowpass's and the bandpass's own settings are left as the preset.
