# Lucas–Kanade flow settings

The settings of `LucasKanadeFlow` (`lucas_kanade_flow.py`), in panel order, with their defaults and
the reason for each. How the flow works is in the module docstrings of `lucas_kanade_flow.py` and
`depth_velocity.py`.

The defaults are the best measured for deep_flow, in this order of importance:

- A: least z noise
- B: a clear difference between slow and fast motion, in x, y and z
- C: least smear

Every number is measured with `scripts/depth_track_compare.py` on the GPU, unless marked
**(deduction)**. The configuration: the flow at 448×800, deep_flow's preset, Neural Depth on an
OAK4, and two recordings at 0.65–1.5 m: `stillslowfast` (still, then an arm slow and fast) and
`standmoves` (arm sweeps, pushes toward the camera). The whole flow takes 2.0 ms per frame.

## optical_flow

| Setting  | Default | Range  | For | Why                                    |
|----------|---------|--------|-----|----------------------------------------|
| use_mask | on      | on/off | C   | no x/y spill onto the background       |
| x, y, z  | on      | on/off | –   | switch an axis off to look at the rest |

How hard the flow pushes the fluid is the bridges' `scale_x/y/z`; the flow's velocity is in image
widths per second.

## lucas_kanade (x/y)

| Setting      | Default | Range (step)        | For | Why                                    |
|--------------|---------|---------------------|-----|----------------------------------------|
| levels       | 3       | 1–5 (1)             | B   | follows fast motion                    |
| window       | 7×7     | 3×3–9×9             | B   | fast ÷ noise 4.4, against 3.7 at 5×5   |
| damping      | 0.03    | 0–0.1 (0.001)       | B   | takes the x/y noise, keeps slow motion |
| same_surface | on      | on/off              | C   | less smear, a hand far in front        |
| threshold    | 0.5     | 0–5 px/frame (0.05) | B   | the solve's jitter off                 |
| median       | 1×1     | 1×1–5×5, hidden     | –   | off: little gain for its GPU time      |

- levels: 1 and 2 barely see motion at 448×800. 4 levels cost +0.14 ms and spill more.
- window: 7×7 smears more than 5×5, but B ranks above C.
- same_surface works only where a no-depth band separates the hand from what is behind it, as with a
  hand held far out. Where a limb is about 20 cm before the body, Neural Depth ramps over about 30 px
  instead of stepping, and the patch's samples stay within the tolerance; see [Open](#open). It costs
  +0.05 ms.
- median: 3×3 costs +0.5 ms and 5×5 +1.5 ms. It is kept for a 2D flow at a lower resolution
  **(deduction)**.

## depth_track (z)

| Setting   | Default      | Range (step)    | For | Why                                    |
|-----------|--------------|-----------------|-----|----------------------------------------|
| edge      | 0.1 m        | 0.05–0.3 (0.01) | A   | halves the edge spikes                 |
| guide     | off          | on/off          | –   | smooths more than it sharpens          |
| window    | 7×7          | 1×1–9×9         | A   | the most samples averaged              |
| tolerance | 0.2 m        | 0.05–0.5 (0.01) | B   | keeps a push's z                       |
| smooth    | 50 ms        | 0–200 (1)       | A   | at 0 the still z noise is 15× higher   |
| xy_smooth | off          | on/off          | B   | x/y without extra lag                  |
| catch_up  | 8 ms         | 0–50 (1)        | B   | fast z 75% → 96%, same still noise     |
| dead_zone | 0.02 m       | 0–0.05 (0.001)  | A   | takes the depth's wander               |
| threshold | 1.0 px/frame | 0–5 (0.05)      | A   | with the dead zone, still z noise is 0 |

- guide, on `standmoves`: push and sweep z both drop 7%, the still z noise stays about the same, and
  it costs +0.14 ms. It does not separate a limb's edge from its motion.
- tolerance at 0.05: a push loses 77% of its z, and the track restarts 6× as often while moving.
- smooth: lower lags less; keep it as low as looks right.

## Constraints

- The false z at an edge cannot be told from a push by time. Restarting the track where the depth
  changed past the tolerance over 2 frames cuts the edge z, and a push's z by 26%.
- It cannot be told by the depth's slope either. A weight on the slope cuts edges and pushes alike,
  by about 10%.
- A forward–backward check on x/y drops half the fast motion.
- Iterations per level do less than one more level, at more GPU time.

## Open

- A limb moving sideways before the body reads false z at its edges: the leading edge carries 43% of a
  sweep's z, the trailing edge 12%. The cause is the depth's ramp at the limb's edge. The camera's
  temporal filter is off and its confidence does not sharpen the ramp. Accepted for now; the cue not
  yet tried is a forward-splat count of the x/y motion (0 = uncovered, above 1 = covered).
- Range: edge, tolerance and dead_zone are in metres and tuned at 0.65–1.5 m. Stereo noise grows with
  the square of the distance, so at 2–3.5 m they may need to be in disparity **(deduction)**. This
  needs a recording at that distance.
