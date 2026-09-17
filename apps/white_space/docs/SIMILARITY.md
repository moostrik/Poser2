# White Space — Similarity

How alike two players' arm postures are, as one number in [0, 1], and what the show does with it. Two
consumers read it in two different ways: the **hit sync** (`STATES.md` *Vocabulary*: in sync) compares the
poses at the moments the playhead hits them and decides INTRO → INTRO_PLAY; the **projection window**
(`LAYERS.md` *pose_instrument*) reads the live, smoothed `Similarity` feature and opens two players'
patterns toward each other. Both run the same kernel on the same settings; they differ in what they feed it
and what they do with the result. This document owns the settings, their meaning in degrees, and how they
depend on each other. All numbers assume `studio.json`.

## Tuning

In this order; each step's number is read off the data view (`render.layers.data`) or the panel.

1. **Calibrate the angles** (`CALIBRATION.md`, the pose reference poses). Everything below reads calibrated
   angles: 0 with the arm hanging and the elbow straight, π with the arm raised. A wrong neutral reference
   moves the neutral floor (step 2) and nothing else; the similarity compares differences.
2. **Set what neutral is**: `pose.arm_deviation_extractor.min_degrees` is the angle from neutral of the arm
   joint furthest from it below which a pose is neutral (`ArmDeviation` 0); `max_degrees` is where it is
   fully out (1). Stand relaxed, read `ArmDeviation`: it must be 0 with the arms hanging and the elbows
   relaxed, and 1 at a T. With `n_top` 1 the single furthest joint decides, so a relaxed elbow is usually
   the one that sets the neutral edge.
3. **Set how alike**: `pose.similarity.window_similarity.angle_tolerance`, in degrees: in sync is the arm
   joints within this. Two players hold the same arm pose; the streak (`states.sync.hits`) should count on
   their hits. Too eager, lower it by 3°; too strict, raise it. Do not touch the remap for this: the
   tolerance is the one knob for likeness (*Interdependence*), and the sync is the test of it — too strict
   and the show never spins up.
4. **Set where the windows open**: `PI.window.sync_threshold`, on the raw scale, in PLAY. Windows too rarely
   open toward each other, lower it; open on unlike poses, raise it.

Restart the app before saving a preset: the running app rewrites `studio.json`.

## The kernel

`modules/pose/analytics/posture_similarity.py`, used by `WindowSimilarity` (`window_similarity.py`, the
live feature) and by `posture_similarity` (the hit sync). For two poses:

1. Per joint, the wrapped angle difference Δ gives `exp(-(Δ / angle_tolerance)²)`: 1 at identical, e⁻¹
   (0.37) at one tolerance, 0.02 at two.
2. Only the joints `joints` selects count (`studio.json`: the four arm joints); a joint missing on either
   side is skipped.
3. The joints left are aggregated with `method`: the harmonic mean is strict, one poor joint pulls the
   whole down.
4. The remap `[remap_low, remap_high] → [0, 1]`. White Space sets 0 and 1, the identity: the value is the
   raw similarity and its meaning is the tolerance's. hd_trio keeps the module defaults.
5. Times the coverage, the fraction of the selected joints both poses have, so a mostly occluded pair
   cannot read as fully alike.
6. Times the **neutral weight** while `pose.similarity.neutral_weight.enabled`: the smaller of the two
   players' `ArmDeviation`. The weight is 0 up to `min_degrees` and 1 from `max_degrees` of the arm joint
   furthest from neutral, linear between. It multiplies, so the closer either player is to neutral, the
   more alike the arms must be; a pair with a person at neutral reads 0.

## Two paths, one kernel

| | Hit sync | Projection window |
|-----------------------|-------------------------------------------------|-----------------------------------------------------|
| Component             | `HitSync` (`pose/hit_sync.py`)                  | `PoseInstrument._set_windows` (`light/layers/projection/pose_instrument.py`) |
| Input                 | the hit player's LERP pose at the hit tick      | the `Similarity` row on the LERP frames             |
| Kernel call           | `posture_similarity` on two hit poses           | `WindowSimilarity` on the SMOOTH windows, `window_length` 1 |
| Smoothing             | none: the angles as they were at the hit        | `SimilarityStickyFiller`, `SimilarityEuroSmoother` (SMOOTH), `SimilarityChaseInterpolator` (LERP) |
| Neutral weight        | `NeutralWeight.weigh` on the two poses          | `NeutralWeight.process` on the analytics result — one rule, one switch |
| Decision              | each hit's harmonic mean toward the others in the run ≥ e⁻¹, the kernel's value at one `angle_tolerance` (`hit_sync.py` `_IN_SYNC`); `min_players` alike hits in a row | pair mean of both directions ≥ `PI.window.sync_threshold`; opening eased from there to 1 |
| Output                | `HitStreak` on the board; `sync.hits`, `sync.similarity` | the window widths; also `/pose/N/similarity/pose` to Max |

The live feature's chain: the `SMOOTH` stage's `Angles` windows feed the `WindowSimilarity` thread; its
result passes the sticky filler (a present pair's gap is held, a departed player's slot is NaN), the
neutral weight, and is stamped on the next SMOOTH frames by `SimilarityApplicator`, then smoothed by the
Euro smoother, then chased at LERP. The hit sync reads none of that: it takes the LERP frame's angles at the
hit tick and scores them directly, so a change to the similarity smoothers changes the windows and the
sound, never the sync.

## Settings

| Setting                                              | Reads it            | `studio.json` | Meaning |
|------------------------------------------------------|---------------------|---------------|---------|
| `pose.angle_calibrator.*`                            | both                | see `CALIBRATION.md` | 0 and π of every joint |
| `pose.angle.smoother` (`min_cutoff`, `beta`)         | both                | 0.1, 0.7      | the angles both paths compare are these smoothed ones |
| `pose.arm_deviation_extractor.min_degrees`           | both                | 20°           | neutral up to here (weight 0) |
| `pose.arm_deviation_extractor.max_degrees`           | both                | 45°           | fully out from here (weight 1) |
| `pose.arm_deviation_extractor.n_top`                 | both                | 1             | the N joints furthest from neutral, averaged |
| `pose.similarity.window_similarity.angle_tolerance`  | both                | 20°           | in sync when the arm joints are within this; the window reads e⁻¹ there. **The knob for how alike** |
| `pose.similarity.window_similarity.joints`           | both                | the arms      | which joints count and are covered |
| `pose.similarity.window_similarity.method`           | both                | HARMONIC_MEAN | how the joints aggregate |
| `pose.similarity.window_similarity.remap_low/high`   | both                | 0, 1          | identity: raw similarity |
| `pose.similarity.window_similarity.window_length`    | live feature        | 1             | posture, not movement; must stay 1 |
| `…use_velocity_similarity`, `use_motion_weighting`, `use_time_penalty` | live feature | off  | movement terms, off for posture |
| `pose.similarity.neutral_weight.enabled`             | both                | on            | the neutral weight, one switch for both paths |
| `pose.similarity.sticky.enabled`, `hold_scores`      | live feature        | on, off       | a present pair's gap held |
| `pose.similarity.smoother` (`min_cutoff`, `beta`)    | live feature        | 0.01, 0.4     | the Euro smoother on the stamped rows |
| `pose.similarity.interpolator` (`responsiveness`, `friction`) | live feature | 0.33, 0.05  | the chase at LERP |
| `states.min_players`                                 | hit sync            | 2             | alike hits in a row to spin up |
| `PI.window.sync_threshold`                           | window              | 0.6           | where windows start to open |
| `PI.window.width`                                    | window              | 33°           | the window before it opens |

## What the numbers mean

A cut `t` on raw similarity means the joints are within `angle_tolerance × √(−ln t)` of each other when all
four are equally off. For the hit sync `t` is e⁻¹ by construction, so the factor is 1 and in sync is the
arms within the tolerance; only the window has a cut of its own. With the harmonic mean, one joint alone may
be further: solving `4 / (3 + 1/s) = t` for the one joint's `s`. With the neutral weight `w`, the raw
likeness needed is `t / w`, and no likeness passes once `w < t`, so identical arms count only from
`min + t × (max − min)` of the joint furthest from neutral.

### Hit sync (`angle_tolerance` 20°, remap 0–1, ramp 20°–45°)

| Meaning                                                                | Value                    |
|------------------------------------------------------------------------|--------------------------|
| in sync, arms 45° or more out: all four arm joints within              | 20°                      |
| in sync: one joint off, the other three exact                          | 28.7°                    |
| in sync, arms 37° out (weight 0.7): all four within                    | 16°                      |
| in sync floor: identical arms count from (joint furthest from neutral) | 29°                      |
| `sync.similarity` reads 0.9 at (all four within)                       | 6.5°                     |

Each degree of tolerance moves the all-four limit by 1° and the one-joint limit by 1.43°.

### Projection window (`angle_tolerance` 20°, remap 0–1, `sync_threshold` 0.6, ramp 20°–45°)

| Meaning                                                                | Value                    |
|------------------------------------------------------------------------|--------------------------|
| windows start to open, arms 45° or more out: all four within           | 14.3°                    |
| windows start to open: one joint off, the other three exact            | 22.8°                    |
| windows half open (eased, similarity 0.8): all four within             | 9.4°                     |
| windows fully open                                                     | identical (0.9 at 6.5°)  |
| windows start to open, arms 37° out (weight 0.7): all four within      | 7.9°                     |
| window floor: identical arms open from (joint furthest from neutral)   | 35°                      |

The window reads the smoothed feature, so these are where it settles, not when: the Euro smoother at
`min_cutoff` 0.01 Hz follows a slow drift with a time constant of about 16 s and a fast change within about
a second (`beta` 0.4) **(deduction from the 1€ filter's formula)**.

## Interdependence

- **The sync's cut is the tolerance point.** The per-joint score is a bell, 1 at identical, e⁻¹ at one
  tolerance, never 0, so "within tolerance" on it is the cut at e⁻¹ and a cut at > 0 would pass any pair out
  of neutral (the remap that once clipped weak likeness to 0 is off). Any other cut `t` only rescales the
  tolerance by `√(−ln t)` and hides the degrees: 20° at e⁻¹ is the same rule as 24° at 0.5. So the cut is a
  constant (`hit_sync.py` `_IN_SYNC`), not a setting, and the tolerance is the sync's only knob. A hard
  gate instead of the bell would give the sync the same answer while taking the level away from the window
  and the sound, and with it the trading between joints and the graded neutral range.
- **The remap and every threshold.** A remap moves what 0 and 1 mean for all consumers at once; both
  thresholds would shift with it. White Space keeps it at the identity so the thresholds read raw.
- **Neutral weight and likeness.** The weight divides the cut: the required likeness is `t / w`. Moving
  `min_degrees` or `max_degrees` moves both floors above (29° for the sync, 35° for the window) and the
  likeness required in between. That is the intended graded range; it is not a second likeness knob.
- **Joints and coverage.** Deselecting a joint removes it from the comparison and from the coverage, so
  unseen legs no longer lower an arm match. The deviation extractors keep their own joint lists
  (`ArmDeviationExtractor`, `LegDeviationExtractor`); the similarity's `joints` is independent of them.
- **Angle smoothing feeds both.** `pose.angle.smoother` shapes the angles the window compares (SMOOTH) and
  the angles the hit reads (LERP, after prediction and the chase). The similarity smoothers shape only the
  live feature.
- **Calibration feeds the floor, not the likeness.** The similarity is a difference of two players'
  angles, so a shared calibration offset cancels; the arm deviation is an absolute angle from neutral, so
  the neutral reference sets where the floor is.
- **`window_length` 1.** Longer windows compare movement over time (hd_trio); the velocity, motion and time
  terms belong to that use and stay off here.
- **`max_players`** (root) is the `Similarity` row's width; a player id beyond it has no slot.

## Open

- `/pose/N/similarity/motion` (`modules/inout/osc_sound.py`) sends the plain `Similarity` row; its comment
  says motion-gated. `MotionGate` goes out separately on `/pose/N/similarity/gate`. Whether Max uses either.
- Whether the window should read the raw hit-tick likeness rather than the smoothed feature, so the two
  paths open and count on the same moment.
