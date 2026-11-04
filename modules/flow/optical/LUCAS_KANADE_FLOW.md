# Lucas–Kanade flow settings

The settings of `LucasKanadeFlow`, in panel order, with their defaults. The defaults are the measured
best for deep_flow's installation, in this order of importance: A least z noise, B a clear difference
between slow and fast motion in x, y and z, C least smear. (m) is measured with
`scripts/depth_track_compare.py` on recorded frames, on the GPU; (i) is inferred from the code. The
whole flow takes about 2.0 ms per frame at 448×800 (m).

## optical_flow

| Setting | Default | Range (step) | Panel text | Why |
|---|---|---|---|---|
| use_mask | on | on/off | Only the player moves the fluid (off: whole image) | C: no spill onto the background (m) |
| x / y / z | on | on/off | x / y / z in the velocity (off: it is 0) | To look at one axis; how hard the flow pushes the fluid is the bridge's |

## lucas_kanade (x/y)

| Setting | Default | Range (step) | Panel text | Why |
|---|---|---|---|---|
| levels | 3 | 1–5 (1) | Pyramid levels of the x/y solve, each doubles the reach (more: faster motion, more spill) | B: follows fast motion; 1–2 barely see motion at 448×800, 4 costs +0.14 ms and spills more (m) |
| window | 7×7 | 3×3–9×9 | x/y patch size (larger: cleaner, more spill) | B: best fast ÷ noise, 4.4 vs 3.7 at 5×5 (m); smears more than 5×5, but B ranks above C |
| damping | 0.03 | 0–0.1 (0.001) | Damps x/y where the patch has little texture (higher: less noise, smaller motion) | B: with the threshold, takes the x/y noise and keeps slow motion (m) |
| same_surface | on | on/off | Patch counts only the center's surface (on: less smear, wide fast parts slower) | C: less smear where a hand is far in front, with a no-depth band between; +0.05 ms (m). No effect up close: the stereo depth ramps over ~30 px where a limb is ~20 cm before the body (m) |
| threshold | 0.5 | 0–5 px/frame (0.05) | x/y below this is noise, in px/frame (higher: less noise, less slow motion) | B: the solve's jitter off, slow motion stays |
| median | 1×1 (hidden) | 1×1–5×5 | Median size per level, 1×1 = off (larger: cleaner, more GPU) | Off: 3×3 +0.5 ms, 5×5 +1.5 ms for little gain (m). Kept for a 2D flow at a lower resolution |

## depth_track (z)

| Setting | Default | Range (step) | Panel text | Why |
|---|---|---|---|---|
| edge | 0.1 m | 0.05–0.3 (0.01) | Edge depth step, m (lower: cleaner edges, fewer fingers) | A: halves the edge spikes (m) |
| guide | off | on/off | Depth sharpened at the image's edges (on: crisper limbs, more GPU) | Tried: −7% z on pushes and sweeps alike, about the same noise, +0.14 ms (m); smooths more than it sharpens |
| window | 7×7 | 1×1–9×9 | Depth averaging size (larger: less noise, small parts blur) | A: most samples averaged, least z noise |
| tolerance | 0.2 m | 0.05–0.5 (0.01) | New-surface depth jump, m (higher: fuller pushes, fewer splits) | B: at 0.05 a push loses 77% and the track restarts 6× as often (m) |
| smooth | 50 ms | 0–200 (1) | Smoothing time of z in ms, 0 = two-frame change (higher: less noise, more lag) | A: at 0 the still z noise is 15× higher (m); as low as looks right |
| xy_smooth | off | on/off | x/y smoothed over smooth, so both lag alike (on: less noise, more lag) | B: x/y without extra lag |
| catch_up | 8 ms | 0–50 (1) | Shorter smoothing during a fast push, in ms, 0 = off (lower: quicker pushes) | B: fast z 75% → 96%, still noise unchanged (m) |
| dead_zone | 0.02 m | 0–0.05 (0.001) | Depth drift ignored before z counts as motion, in m (higher: less drift, slower start) | A: takes the depth's wander (m) |
| threshold | 1.0 px/frame | 0–5 (0.05) | z below this is noise, in px/frame (higher: less noise, less slow z) | A: with the dead zone, the still z noise measures 0 (m) |

## Tried and removed

| What | Why it went |
|---|---|
| xy iterations | 4 levels do the same better and cheaper (m) |
| xy round trip (forward–backward check) | Dropped half the fast motion, gained nothing (m) |
| Gaussian-weighted patch | No gain over the plain patch (m) |
| An x/y gate on z, a One-Euro filter on z | Both worse than catch_up (m) |
| Restart over several frames | Cut the false z at a sweeping arm's edges, but also 26% of a push at 2 frames (m) |
| Slope weight (PD-Flow style) | Cut edges and pushes alike, ~10% each (m) |
| Gains on the flow | Moved to the bridges (scale_x/y/z); the flow is in true units |

## Known limits

- A limb moving sideways in front of the body reads some false z at its edges: the leading edge
  carries 43% of a sweep's z, the trailing edge 12% (m). The cause is the depth itself: Neural Depth
  draws a ramp of ~30 px, not a step, where a limb is ~20 cm before the body (m), and the camera's
  filters and confidence do not sharpen it. Accepted.
- Range: the depth settings in metres (edge, tolerance, dead_zone) are tuned at 0.65–1.5 m. Stereo
  noise grows with the square of the distance, so at 2–3.5 m they may need to be in disparity; this
  needs a recording at that distance.
