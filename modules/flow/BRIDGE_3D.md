# The 3D bridge

Every quantity and setting here is in a meaningful unit ([FLOW.md](FLOW.md#the-2d-and-3d-flow-systems)).

The bridge turns the motion's velocity into what the fluid takes, in units of the body: how fast it
moves, the dye it brings, its warmth. How the stages fit together is in [FLOW.md](FLOW.md), what
the fluid does with each output in [FLUID_3D.md](FLUID_3D.md#the-inputs), what is next in
[FLOW_PLAN.md](FLOW_PLAN.md). "Code" is what the files show; **(deduction)** marks what is
inferred from them, **(site fact)** what the operator saw. The 2D bridge (`bridge_2d/`) gives the
boundary only and stays as on `main` until it is rebuilt on the Lucas–Kanade flow.

## The goal

The motion translated into what the fluid takes, and held between the camera's frames
([FLOW.md](FLOW.md#the-goals)).

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
a gradual change; the velocity's height reads the Δv only for a sudden one
(`modules/flow/tests/test_bridge_accelerations.py`). The trail's rate, (new − moved) / dt, adds up to the
trail's change the same way. A filter
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
average follows a moving limb instead of a fixed pixel, and a fast move is not cut off by a smoothing time
longer than it takes to cross. `blur` spreads it in space; the ring it leaves
around the body is part of the look **(site fact)**. The blur runs on the trail each step feeds back, so
blurring every step is diffusion: `blur` is how far the trail spreads in one second, in widths at half its
height, and each step blurs by σ = `blur` / 2.355 × √dt widths (σ² = 2 D dt), the same spread at any frame
rate and resolution; what is seen also grows with the trail's memory. It is the textbook separable Gaussian, both directions in one compute dispatch
(`LowpassBlur`), the trail and its rate together.

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

What the fluid takes is two systems, the two a solver tells apart, and the expansion, a source of volume,
each output mixing the six readings, each reading at a share, 0 to 1, as the light renderer mixes its layers
(`Bridge3DLayer`, `WeightedSum`). The boundary is the body in the water, what the water is dragged toward: its
velocity, and its dye and warmth, which have no force and are held at the body. The body force is the
Navier–Stokes force term, what is added: an acceleration, which the water keeps after the body is gone, so a
stop can throw the water on. The panel shows `readings` (`lowpass`, `bandpass`), then one subpanel per output,
named as the output, in the order the bridge makes them. Each output's group holds the six shares as knobs on
the first row (`MixSettings`: `trail`, `change`, `start`, `bow`, `stop`, `wake`), an amount's (the density's,
the temperature's, the expansion's) a seventh, `mask` (`AmountMixSettings`), and the density's own settings on
the second (`saturation`, `brightness`).

| Subpanel       | Readings enter as | Goes in as     | Output, its unit                                                                                     | Default       |
|----------------|-------------------|----------------|------------------------------------------------------------------------------------------------------|---------------|
| `readings`     |                   |                | the filters' settings, `lowpass` and `bandpass`                                                      |               |
| `velocity`     | velocities        | boundary       | widths/s, the body's velocity; with its `peak` ([The peak](#the-peak))                               | the trail 1   |
| `acceleration` | accelerations     | body force     | widths/s², a Δv added once over a change                                                             | all 0: `None` |
| `density`      | velocities        | boundary value | the dye, RGBA: the color times its presence, the presence as alpha, 0 to 1                           | the trail 1   |
| `temperature`  | velocities        | boundary value | the warmth, −1 cold to 1 hot, its presence folded in                                                 | the trail 1   |
| `expansion`    | velocities        | source         | where the water swells, a presence 0 to 1, capped there; the newest capture's in a tick of several   | all 0: `None` |

The five outputs (`BridgeOutput`, `Injection`) are the subpanels' names, each named by what the fluid gets,
in a unit of the body; how it goes in, as the boundary or the body force, is in the column beside it.

The presence is the mix's speed in widths/s, the sum of its readings' sizes (the length of each, at
its share), capped at 1: a part moving at 1 width/s or faster is fully there, a slower one partly,
and a still body leaves no dye. The `mask` share adds the view's mask (`BridgeInput.mask`, 0 to 1 where the
player is) at the share, a presence 0 to 1 by being there: at share 1 a still
body is fully there. Widths/s is the velocity's own unit, so no reference speed is chosen.
The density's color is the picture's (`DensityColor`: the image, or white, times the tint, gated by
the mask) through `saturation` and `brightness`; the warmth is (warm − cold) × (1 − neutral × 0.5) ×
the presence, from the same color's red, blue and green. The readings of one unit share the mix, so
the mix has no gain: the motion's gains balance the axes and the fluid's gains set how the water
follows; a weight is a share. The acceleration and the expansion are `None` while their shares are
all 0. The ports follow the outputs, a group each, named as the output, in the order the bridge makes
them: `velocity.velocity`, `acceleration.acceleration`, `density.amount`, `temperature.amount` and
`expansion.expansion`, each with its `z_position`; the density's and the temperature's groups then hold
their bridges' own (`density.input`, `density.output`, ...).

A wake may drag (its share in `velocity`: the water behind a leaving limb is pulled to the
speed it left with, and let go when the pulse dies), kick (its share in `acceleration`),
leave dye where it is, or any of these at once; the same reading in two outputs is a
choice, not an error. The change beside its parts lets a preset take the physics (the true
acceleration, braking at stops) or the look (a stop that pushes).

The bridge has no finite guard, unlike the motion and the fluid ([MOTION.md](MOTION.md#the-guard),
[FLUID_3D.md](FLUID_3D.md#the-guard)): it makes no non-finite value itself, every division in its
shaders held above a small floor, and its input is the guarded motion.

The bridge steps once per capture, after the motion, by the capture's interval: its filters see each sample
once, at its own time. Without a capture (the camera stalls, -sim is paused) it does not step, and the fluid keeps
taking its last outputs; a stalled camera is the camera's to fix, not the flow's.

## The z positions

The camera sees the body's front surface, a plane of x, y and depth, so the fluid takes each pixel as a
thin skin at its own depth ([FLUID_3D.md](FLUID_3D.md#the-inputs)). Where that is, a z position, the
motion makes from the depth track: (depth − near) / (far − near) over the player's depth range, 0 front
to 1 back ([MOTION.md](MOTION.md#depth_velocity-z)). Every z position travels as itself times its
weight in red and the weight in green, so a clear, a reset or the image's border (0) is no z position,
not the front, and a linear read, the advect and the blur mix z positions only where they are known.

A trail holds the history of the motion: its velocity and its z position. Each trail's z position is
the motion's through the trail's own Ema step, in the same pass (a pixel's attack or release weight is
picked by its velocity), weighted by the speed each step brings, so it is where the motion the trail
holds was: an arm leaving the torso leaves its trail and its wake at the arm's depth, as the still
torso brings no speed, and a z position whose motion has died counts for nothing. The lowpass's moves
and blurs with its trail; the bandpass's fast and slow trail each hold one.

| Output                              | Its z position                                                                                                       |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `velocity_z_position`               | its readings' z positions weighted by their sizes at their shares (`ZPositionMix`)                                  |
| `acceleration_z_position`           | the same, by the accelerations' sizes                                                                                |
| the density's and the temperature's | the same, by its readings' sizes at its own shares: its amount and its place from the same trails                    |
| their `mask` share                  | the motion's z position now, weighted by the mask at the share                                                       |

The density's amount is mixed from the trails, which the lowpass advects and blurs past where the body has a
depth now (the depth velocity gives none in a band a depth sample wide along every outline); its z position
mixed from the same trails has one wherever its amount has, so the fluid never puts dye where there is no
place for it.

A reading reads its trail's z position: the trail the lowpass's, a start and a bow the fast trail's, a
stop and a wake the slow trail's, the change the fast trail's where it is the faster, else the slow's.
Each counts by its size at its share times the share of its trail's speed that came with a z position,
so motion without depth does not pull an output's.

## The peak

How fast the body gets, to set the gains by: `velocity.peak` (`VelocityPeak`, `VelocityMixSettings`),
`enabled`, off by default, and `speed`, read only: the 99th percentile of the velocity's speed over the
moving pixels of the last second, in widths/s. The percentile, not the maximum, so a few wrong pixels do not set
it; over the moving pixels, so a still body and the empty image do not pull it down, and a leaving limb's wake
counts as the water is dragged toward it.

Each velocity the bridge makes is counted by the monitor's histogram counter (`HistogramCount`, its speed mode)
on the same grid of samples, at most 512 a side: each sample's speed into 256 log bins from 0.01 to 10 widths/s,
each 2.7% wider than the one before, so a slow and a fast body read alike closely. Below 0.01 widths/s is still
and not counted; above 10 counts in the last bin, and `speed` reads 10, at least 10: the fastest body measured is
about 3 widths/s, so 10 says something is wrong (z noise, a bad frame). The bins add up over a second; then they
are copied into one of two pixel buffers behind a fence and cleared, and the percentile is read from a copy the
GPU has finished, log-interpolated within its bin; nothing waits. Off, nothing runs and `speed` is 0, and it stays
0 until the first read after it is turned on.

Measured with `scripts/velocity_peak_check.py` on an RTX 4090: on 448 × 512 speeds every sample is binned as
a count in numpy binned it but 2 in 183,500 at a bin's edge (the GPU's single-precision log), and `speed` reads the
bins' percentile, 3.989 widths/s against numpy's 3.988 on the speeds. On deep_flow's 448 × 800 velocity
(448 × 512 samples) the count takes 0.027 ms a velocity, the copy 0.018 ms once a second.

## Measured

`scripts/bridge_compare.py` on the standmoves ([gone](FLOW.md#recordings-gone)) recording (`20261103-215324`: four whole-arm moves
from rest, with starts and stops, a player at 0.65–1.5 m), its frames as captured on 2026-11-05
(`temp/depth_track_1105/`, 1449 frames, 25 fps, the stamps the replay's decode times), the flow run
again on them with deep_flow's preset flow of 2026-11-06 (levels 4, same_surface and z_smooth on,
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
