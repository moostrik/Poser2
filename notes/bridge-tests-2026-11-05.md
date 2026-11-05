# Flow bridge tests, 2026-11-05

The velocity trail and the kinetic bridge (all of its modules) on today's flow: Lucas–Kanade per level, capture-stamped intervals, the preset's flow settings (levels 5, threshold 0.5, same_surface off, depth_track as the preset).

## Summary

**Measured** means a number from the tables below. **Inferred** means reasoning from the code or from the numbers, not measured on its own.

| Module | Promise | Holds? | The numbers (standmoves unless named) |
|---|---|---|---|
| Velocity trail, as the preset (smooth 1 s, attack 1 ms, advect 1, blur 4.43) | smooth velocity, flicks at full strength, a stop coasts | yes | stops coast 0.87 s (flow 0.01 s); fast ÷ slow 2.01 (flow 1.93); lag 40 ms; smear 0.19 |
| trail `smooth` | longer coast | yes | 0.16 s: coast 0.07 s; 1 s: 0.87 s |
| trail `attack` | quicker flicks | yes | attack 0: fast 0.066 vs 0.301, start ÷ fast 0.59 vs 1.16, lag 160 vs 40 ms |
| trail `advect` | follows the limb, no fixed-pixel wake | yes, strongly | advect 0: still-body noise 0.27 vs 0.018, smear 0.40 vs 0.19, coast 1.84 vs 0.87 s |
| trail `blur_radius` | spatial smoothing | barely visible | blur 0 vs 4.43: noise 0.017 vs 0.018, smear 0.16 vs 0.19 |
| Kinetic: velocity (`smooth_velocity`) | fast trail, dropped to the slow one on a release | partly | almost the flow itself: coast 0.06 s, lag 0 ms, smear 0.10. drop_velocity 1 vs 0 only makes the coast 0.06 vs 0.01 s |
| Kinetic: emission (density amount) | ink on a release, not on steady motion | partly | stop ÷ fast 1.16 (trail 0.70) and start ÷ fast 0.40, so it does lean to stops. But noise on a still body is 0.24, as high as steady fast motion (0.26) |
| … with `scale_z` 0 | – | – | emission noise 0.000; stop ÷ fast 1.54. **All of its still noise is z** |
| `trails.fast_smooth` | how fast the fast trail follows | yes | 100 ms: emission noise 0.05, stop ÷ fast 1.62, but a third less velocity at starts (0.198 vs 0.294) |
| `trails.slow_smooth` | the reference the change is read against | yes | 500 ms: stop ÷ fast 3.18, but slow steady motion also emits (fast ÷ slow 1.25); 50 ms: stop ÷ fast 0.48 |
| `trails.advect` | a moving limb leaves no wake | a little | advect 1: emission start ÷ fast 0.21 vs 0.40, stop ÷ fast 1.31 vs 1.16, smear 0.07 vs 0.10 |
| `trails.blur_position` / `blur_radius` | spatial smoothing | no visible effect | INSIDE, BEFORE and radius 0 are within 0.02 on every measure |
| `release.stop_gate` | only a real slowdown releases | yes | gate 1: steady fast emission 0.106 vs 0.256, still noise 0.087 vs 0.243, stop ÷ fast 1.60 vs 1.16 |
| `release.hold` | holds the release | yes, but it lifts everything | hold 500: stop 0.81, but steady fast 0.56 too (stop ÷ fast 1.46); hold 0: 1.42 |
| `release.emission_gain` | ink per release | yes, linear | gain 2: every emission number ×2.0 |
| `release.drop_velocity` | velocity follows the slow trail on a release | yes, but small | coast 0.06 vs 0.01 s, stop velocity 0.033 vs 0.018 |
| `onset.force` | push at a start | yes | onset start ÷ fast 1.84, stop ÷ fast 0.11; force 2 is ×4.0 of 0.5. Noise 0.043 on a still body ≈ steady fast 0.046 (z); 0 with scale_z 0 |
| `onset.spin` / curl | spin from the motion's curl | not tested | no turning clip on this machine; the curl is there (start ÷ fast 1.58, still noise 0) |
| `reservoir` (enabled, leak) | motion stores ink, a release spends it | no | leak 4 s: stop ÷ fast 0.81, leak 1 s: 0.74, vs 1.16 off. The ink moves to steady fast motion, away from the stops |

**Kinetic vs trail on what the fluid gets** (measured):
- **Velocity:** the trail smooths and coasts (0.87 s); the kinetic bridge passes the flow on almost unchanged (0.06 s).
- **Density amount:** the trail's follows speed (stop ÷ fast 0.70); the kinetic's leans to stops (1.16, 1.60 with the stop gate).
- **Noise on a still body:**
  - kinetic emission: 0.24 on standmoves, 0.19 on stillslowfast
  - trail amount: 0.03 and 0.04
  - kinetic onset: 0.04, which is the same z noise
- **Fast ÷ slow:** about the same on standmoves (trail 2.0, kinetic 1.9). On stillslowfast, the kinetic amount is 10.5 and the trail amount 5.5.
- **GPU** (alone, stillslowfast): trail 0.9 ms, trail without advect/blur 0.5 ms, kinetic 2.1 ms. In runs with many sides the same bridges read 0.3–0.7 ms, so take GPU numbers as rough only.

**Inferred:**
- The kinetic bridge's main problem today is z noise. The stop gate and `scale_z` 0 both remove most of it, and a z-only gate would be the precise fix.
- The trail's `advect` matters more than its `blur`. Without advect, 1 s of smoothing with an instant attack keeps the arm's peaks at fixed pixels: noise 0.27, smear 0.40.

## How it was measured

- **Harness:** `scripts/bridge_compare.py` (new).
  - One `LucasKanadeFlow` pass over the saved frames, then any number of bridge variants side by side.
  - Each bridge steps once per flow frame by the frame's saved interval, as the kinetic bridge does in the app.
  - Every variant changes only the named setting; the rest is as in deep_flow's preset.
  - The kinetic velocity is divided by its `scale_x` (10), so it is in image widths per second like the trail. Emission and onset are as the bridge gives them.
  - `onset.force` is 0.5, so the onset is computed in every kinetic run.
- **Frames:** recaptured today with `scripts/depth_track_capture.py` into `temp/depth_track_1105/`. The Nov 3 captures in `temp/depth_track/` are kept.
  - **stillslowfast:** 275 frames, 15.1 s.
    - Intervals: median 0.040 s.
    - 20% of the frames are 0.080 s, because saving the frames drops frames; the capture-stamped interval covers the gap.
    - Its one move ends at 16.0 s, after the loop's end (15.1 s), so it has no stop. It is used only for noise and slow vs fast.
  - **standmoves:** 1449 frames, about 58 s.
    - Intervals: median 0.040 s, 4% at 0.06 s or more.
    - Four moves with starts and stops.
  - **Segments:** `segments.json` is copied from the Nov 3 captures (same recordings, same loop start). Check: the flow's own velocity is 0.018 in the stop windows vs 0.177 in fast steady motion, so the ends line up.
- **Stamps:**
  - The recordings replay through `modules/camera/replay`, which stamps a frame with `perf_counter()` when it decodes it.
  - So the intervals are decode times paced by the player, not the recorded capture times.
  - With the live camera they are the device's capture times.
- **Clips:** the 20261031 clips (Flick, Slow_Wave, Sudden_Stop, Circle_Hand, Buildup_Stop, Body_Move) are not on this machine, so every module was tested on standmoves.
- **Columns:** all values are per frame, the mean over the body's pixels that move in the loop (the mask, more than 12 px inside).

| Column | What it is |
|---|---|
| Noise | p99 per pixel on the body in the still part, its first 1.5 s left out. A p99 against means, so read it across rows, not against Slow and Fast |
| Slow, Fast | mean in each move's slow and fast half, the first 0.3 s of each left out |
| Start | the highest value in a move's first 0.4 s, mean over the moves |
| Stop | the highest value from 0.1 s before a move's end to 0.6 s after, mean over the moves |
| Lag | the delay that best lines the velocity up with the flow's own |
| Coast | seconds after a move's end until the velocity is below 20% of its fast mean |
| Smear | fast-half velocity 4–24 px outside the mask over the one on the body |

## Tables

### Velocity trail, standmoves

| Side | Measure | Noise | Slow | Fast | Fast / slow | Fast / noise | Start | Stop | Stop / fast | Start / fast | Lag ms | Coast s | Smear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow | velocity | 0.000 | 0.092 | 0.177 | 1.93 | no noise | 0.321 | 0.018 | 0.10 | 1.81 | 0 | 0.01 | 0.00 |
| flow | z | 0.160 | 0.038 | 0.090 | 2.38 | 0.6 | 0.130 | 0.020 | 0.22 | 1.44 | | | |
| preset | velocity | 0.018 | 0.150 | 0.301 | 2.01 | 16.4 | 0.349 | 0.207 | 0.69 | 1.16 | 40 | 0.87 | 0.19 |
| preset | z | 0.018 | 0.017 | 0.030 | 1.76 | 1.7 | 0.021 | 0.028 | 0.93 | 0.71 | | | |
| preset | amount | 0.026 | 0.154 | 0.308 | 1.99 | 11.7 | 0.353 | 0.216 | 0.70 | 1.15 | | | |
| smooth 0.16 | velocity | 0.002 | 0.108 | 0.227 | 2.10 | 120.7 | 0.331 | 0.057 | 0.25 | 1.46 | 0 | 0.07 | 0.09 |
| smooth 0.16 | amount | 0.059 | 0.123 | 0.254 | 2.06 | 4.3 | 0.346 | 0.063 | 0.25 | 1.36 | | | |
| attack 0 | velocity | 0.006 | 0.050 | 0.066 | 1.33 | 10.2 | 0.039 | 0.089 | 1.36 | 0.59 | 160 | 1.42 | 0.28 |
| attack 0 | amount | 0.019 | 0.058 | 0.083 | 1.44 | 4.4 | 0.051 | 0.106 | 1.28 | 0.61 | | | |
| advect 0 | velocity | 0.270 | 0.222 | 0.523 | 2.36 | 1.9 | 0.439 | 0.560 | 1.07 | 0.84 | 80 | 1.84 | 0.40 |
| advect 0 | amount | 0.270 | 0.225 | 0.526 | 2.34 | 1.9 | 0.442 | 0.564 | 1.07 | 0.84 | | | |
| blur 0 | velocity | 0.017 | 0.150 | 0.322 | 2.15 | 18.7 | 0.350 | 0.219 | 0.68 | 1.09 | 40 | 0.84 | 0.16 |
| blur 0 | amount | 0.032 | 0.155 | 0.328 | 2.12 | 10.2 | 0.355 | 0.226 | 0.69 | 1.08 | | | |
| defaults (advect 0, attack 0, smooth 0.16, blur 0) | velocity | 0.000 | 0.079 | 0.135 | 1.70 | 306.1 | 0.160 | 0.051 | 0.38 | 1.18 | 40 | 0.09 | 0.16 |
| defaults | z | 0.060 | 0.030 | 0.065 | 2.14 | 1.1 | 0.064 | 0.017 | 0.27 | 0.98 | | | |
| defaults | amount | 0.060 | 0.096 | 0.173 | 1.81 | 2.9 | 0.180 | 0.060 | 0.34 | 1.04 | | | |

### Kinetic bridge, standmoves: trails

| Side | Measure | Noise | Slow | Fast | Fast / slow | Fast / noise | Start | Stop | Stop / fast | Start / fast | Lag ms | Coast s | Smear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| preset | velocity | 0.001 | 0.095 | 0.176 | 1.85 | 339.0 | 0.294 | 0.033 | 0.19 | 1.67 | 0 | 0.06 | 0.10 |
| preset | z | 0.133 | 0.040 | 0.094 | 2.37 | 0.7 | 0.125 | 0.022 | 0.23 | 1.33 | | | |
| preset | amount | 0.243 | 0.126 | 0.256 | 2.03 | 1.1 | 0.102 | 0.296 | 1.16 | 0.40 | | | |
| preset | onset | 0.043 | 0.016 | 0.046 | 2.93 | 1.1 | 0.085 | 0.005 | 0.11 | 1.84 | | | |
| preset | curl | 0.000 | 0.014 | 0.025 | 1.80 | 61.0 | 0.039 | 0.004 | 0.15 | 1.58 | | | |
| fast_smooth 0 | amount | 0.339 | 0.143 | 0.267 | 1.87 | 0.8 | 0.113 | 0.320 | 1.20 | 0.42 | | | |
| fast_smooth 100 | velocity | 0.000 | 0.085 | 0.147 | 1.74 | 325.1 | 0.198 | 0.039 | 0.26 | 1.35 | 40 | 0.07 | 0.13 |
| fast_smooth 100 | amount | 0.047 | 0.043 | 0.096 | 2.22 | 2.0 | 0.015 | 0.155 | 1.62 | 0.15 | | | |
| fast_smooth 100 | onset | 0.009 | 0.005 | 0.012 | 2.43 | 1.4 | 0.021 | 0.001 | 0.10 | 1.78 | | | |
| slow_smooth 50 | velocity | 0.000 | 0.091 | 0.171 | 1.87 | 529.8 | 0.293 | 0.019 | 0.11 | 1.72 | 0 | 0.02 | 0.04 |
| slow_smooth 50 | amount | 0.169 | 0.054 | 0.142 | 2.61 | 0.8 | 0.072 | 0.069 | 0.48 | 0.50 | | | |
| slow_smooth 500 | velocity | 0.002 | 0.100 | 0.177 | 1.78 | 89.4 | 0.295 | 0.060 | 0.34 | 1.66 | 0 | 0.17 | 0.14 |
| slow_smooth 500 | amount | 0.172 | 0.166 | 0.208 | 1.25 | 1.2 | 0.099 | 0.661 | 3.18 | 0.48 | | | |
| slow_smooth 500 | onset | 0.057 | 0.033 | 0.080 | 2.44 | 1.4 | 0.136 | 0.013 | 0.16 | 1.70 | | | |
| advect 1 | velocity | 0.001 | 0.093 | 0.175 | 1.88 | 338.3 | 0.293 | 0.036 | 0.21 | 1.67 | 0 | 0.07 | 0.07 |
| advect 1 | amount | 0.243 | 0.115 | 0.258 | 2.24 | 1.1 | 0.054 | 0.338 | 1.31 | 0.21 | | | |
| blur INSIDE | amount | 0.236 | 0.123 | 0.252 | 2.04 | 1.1 | 0.101 | 0.291 | 1.16 | 0.40 | | | |
| blur 0 | amount | 0.254 | 0.128 | 0.261 | 2.04 | 1.0 | 0.104 | 0.301 | 1.15 | 0.40 | | | |
| scale_z 0 | velocity | 0.001 | 0.096 | 0.178 | 1.84 | 324.5 | 0.297 | 0.036 | 0.20 | 1.67 | 0 | 0.07 | 0.10 |
| scale_z 0 | amount | 0.000 | 0.112 | 0.187 | 1.67 | 762.2 | 0.131 | 0.288 | 1.54 | 0.70 | | | |
| scale_z 0 | onset | 0.000 | 0.011 | 0.031 | 2.68 | no noise | 0.073 | 0.002 | 0.05 | 2.38 | | | |

### Kinetic bridge, standmoves: release, onset, reservoir

A setting left out of a row leaves that measure as in the preset rows above.

| Side | Measure | Noise | Slow | Fast | Fast / slow | Fast / noise | Start | Stop | Stop / fast | Start / fast | Lag ms | Coast s | Smear |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stop_gate 1 | amount | 0.087 | 0.039 | 0.106 | 2.71 | 1.2 | 0.051 | 0.169 | 1.60 | 0.49 | | | |
| hold 0 | amount | 0.218 | 0.106 | 0.187 | 1.76 | 0.9 | 0.096 | 0.264 | 1.42 | 0.51 | | | |
| hold 500 | amount | 0.353 | 0.257 | 0.559 | 2.17 | 1.6 | 0.217 | 0.814 | 1.46 | 0.39 | | | |
| emission_gain 2 | amount | 0.486 | 0.252 | 0.512 | 2.03 | 1.1 | 0.205 | 0.592 | 1.16 | 0.40 | | | |
| drop_velocity 0 | velocity | 0.000 | 0.091 | 0.169 | 1.87 | 914.6 | 0.293 | 0.018 | 0.11 | 1.73 | 0 | 0.01 | 0.02 |
| onset.force 2 | onset | 0.171 | 0.063 | 0.185 | 2.93 | 1.1 | 0.339 | 0.021 | 0.11 | 1.84 | | | |
| reservoir, leak 4 | amount | 0.168 | 0.333 | 0.851 | 2.56 | 5.1 | 0.281 | 0.691 | 0.81 | 0.33 | | | |
| reservoir, leak 4 | reservoir | 0.970 | 1.386 | 2.192 | 1.58 | 2.3 | 1.613 | 2.119 | 0.97 | 0.74 | | | |
| reservoir, leak 1 | amount | 0.085 | 0.202 | 0.551 | 2.72 | 6.5 | 0.178 | 0.407 | 0.74 | 0.32 | | | |
| reservoir, leak 1 | reservoir | 0.467 | 0.591 | 1.112 | 1.88 | 2.4 | 0.691 | 0.948 | 0.85 | 0.62 | | | |

### Trail vs kinetic, stillslowfast (noise and slow vs fast; no stop in this loop)

| Side | Measure | Noise | Slow | Fast | Fast / slow | Fast / noise | Lag ms | Smear |
|---|---|---|---|---|---|---|---|---|
| flow | velocity | 0.011 | 0.020 | 0.124 | 6.16 | 11.1 | 0 | 0.00 |
| flow | z | 0.000 | 0.007 | 0.256 | 34.58 | no noise | | |
| trail preset | velocity | 0.037 | 0.050 | 0.258 | 5.17 | 6.9 | 40 | 0.18 |
| trail preset | amount | 0.037 | 0.051 | 0.280 | 5.48 | 7.5 | | |
| trail defaults | velocity | 0.004 | 0.017 | 0.082 | 4.80 | 22.7 | 40 | 0.10 |
| trail defaults | amount | 0.039 | 0.021 | 0.233 | 11.00 | 6.0 | | |
| kinetic preset | velocity | 0.010 | 0.021 | 0.122 | 5.90 | 12.3 | 0 | 0.05 |
| kinetic preset | amount | 0.185 | 0.046 | 0.485 | 10.47 | 2.6 | | |
| kinetic preset | onset | 0.011 | 0.004 | 0.080 | 18.85 | 7.1 | | |
| kinetic scale_z 0 | amount | 0.005 | 0.042 | 0.163 | 3.91 | 31.1 | | |
| kinetic scale_z 0 | onset | 0.003 | 0.003 | 0.032 | 10.70 | 11.0 | | |

### Stop amount from the trail, standmoves

`stop` = max(0, |trail x/y| − |flow x/y|) per pixel, from the read-back trail and flow velocities (numpy, `scripts/bridge_compare.py`). Trail as the preset (advect 1, attack 1 ms, smooth 1.0 s, blur 4.43) unless named; kinetic with `release.stop_gate` 1 and `scale_z` 0, its `amount` being its emission. One run, standmoves, once per flow frame. Burst: s after a move's end until under 20% of that stop's peak. Ring: the 4–24 px outside the mask over the body, in the stop windows.

| Side | Measure | Noise | Slow | Fast | Stop | Stop / fast | Burst s | Ring | GPU ms |
|---|---|---|---|---|---|---|---|---|---|
| trail preset | amount (speed) | 0.036 | 0.084 | 0.150 | 0.131 | 0.88 | 2.38 | 0.46 | 1.43 |
| trail preset | stop | 0.032 | 0.043 | 0.068 | 0.110 | 1.63 | 1.42 | 0.41 | |
| trail smooth 0.5 | stop | 0.006 | 0.029 | 0.053 | 0.073 | 1.38 | 0.67 | 0.30 | 1.48 |
| trail blur 0 | stop | 0.021 | 0.043 | 0.076 | 0.105 | 1.40 | 1.42 | 0.18 | 0.71 |
| trail smooth 0.5, blur 0 | stop | 0.005 | 0.030 | 0.060 | 0.071 | 1.19 | 0.69 | 0.16 | 0.71 |
| kinetic stop_gate 1, scale_z 0 | amount (emission) | 0.000 | 0.002 | 0.004 | 0.012 | 2.92 | 0.24 | 0.18 | 1.98 |

- The stop picks out stops better than the speed: 1.63 against 0.88 on the preset trail. The kinetic bridge with its stop gate stays higher at 2.92, with a far shorter burst (0.24 s).
- The stop is not 0 while moving: during fast moves it is about half the trail's speed (0.068 vs 0.139). Inferred: the coast after each slowdown inside a move, and advect and blur carrying the trail where the flow has none.
- The burst follows `smooth`: 1.42 s at 1.0, 0.67 s at 0.5.
- The ring follows the blur: 0.41 with blur 4.43, 0.18 without, as low as the kinetic bridge's.
- Noise at smooth 1.0 is likely still the trail letting go of the loop's start, which 1.5 s of settling doesn't cover at a 1 s smooth (inferred); at 0.5 it is 0.006.
- Blur 4.43 doubles the trail's GPU time (0.71 → 1.43 ms).

## Open questions

- **Curl and spin:** untested. They need a clip with a turning hand; the 20261031 Circle_Hand clip isn't on this machine.
- **Reservoir:** ink moves from the stops to steady motion. Is that the intended look, or should a release be the only way out?
- **z noise in the kinetic bridge:** the measures point at a z-only answer, e.g. the stop gate or a lower `scale_z`, before any tuning of hold or gain (inferred).
- **Kinetic blur:** `blur_radius` and `blur_position` show no effect on these measures at 448×800. Keep them, or drop them?
- **Trail timing in the app:** the app steps the trail every render tick, re-reading a held flow frame (Flow3DLayer). This harness steps it once per flow frame. At 50 fps into 60 Hz the app counts one frame in five twice, so its trail numbers would be somewhat higher than these (inferred, not measured).
- **Coast:** the trail coasts 0.87 s and the kinetic bridge 0.06 s. Which one the fluid needs is a look question for you.
