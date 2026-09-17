# White Space — Similarity

How alike two players' arm postures are, and what the show does with it. The posture module
(`modules/pose/analytics/posture_similarity.py`) answers two questions: how far apart two postures are, in
degrees, and how alike that makes them, 0 to 1. Two consumers use the answers in two ways: the **hit sync**
(`STATES.md` *Vocabulary*: in sync) compares the poses at the moments the playhead hits them and decides
INTRO → INTRO_PLAY; the **`Similarity` feature** on the pose frames carries the live 0-to-1 value to the
projection window (`LAYERS.md` *pose_instrument*), the sound and the data view. This document owns the
definition, its settings, their meaning in degrees and how they depend on each other. All numbers assume
`studio.json`.

## Tuning

In this order; each step's number is read off the data view (`render.layers.data`) or the panel.

1. **Calibrate the angles** (`CALIBRATION.md`, the pose reference poses). Everything below reads calibrated
   angles: 0 with the arm hanging and the elbow straight, π with the arm raised. A wrong neutral reference
   moves the neutral edge (step 2) and nothing else; the distance compares two players' angles, so a shared
   offset cancels.
2. **Set what neutral is**: `pose.arm_deviation_extractor.min_degrees` is how far the arm joint furthest
   from neutral may be while the pose still counts as neutral (`ArmDeviation` 0); `max_degrees` is where the
   pose is fully out (1). Stand relaxed, read `ArmDeviation`: 0 with the arms hanging and the elbows relaxed,
   1 at a T. With `n_top` 1 the single furthest joint decides, so a relaxed elbow usually sets the neutral
   edge. `max_degrees` is also the hit sync's gate: below it a hit does not count.
3. **Set how alike**: `pose.similarity.posture.angle_tolerance`, in degrees. Two players fully out of
   neutral hold the same arm pose; `states.sync.distance` reads the degrees between their hits and
   `states.sync.hits` counts. Too eager, lower the tolerance by 3°; too strict, raise it. The sync is the test
   of it: too strict and the show never spins up. Two people at different positions are seen from different
   directions, so the same pose reads some degrees apart for that reason alone; the tolerance absorbs it.
4. **Set the forgiveness** only if one joint, usually an elbow, keeps breaking matches that look right:
   `pose.similarity.posture.forgiveness`. 1 is none.
5. **Set where the windows open**: `PI.window.sync_threshold`, in PLAY. Windows too rarely open toward each
   other, lower it; open on unlike poses, raise it.

Restart the app before saving a preset: the running app rewrites `studio.json`.

## The definition

**Distance**, in degrees, 0 identical to 180 opposite:

1. Per joint, the wrapped difference of the two players' calibrated angles. Left compares to left.
2. Only the joints `joints` selects count (`studio.json`: the four arm joints). A joint missing on either
   side is not compared; the fraction of the selected joints that were is the coverage, and becomes the
   `Similarity` score.
3. Over the joints compared, a soft maximum: the worst joint decides, but one joint is forgiven when the
   rest match. `forgiveness` says by how much: a lone joint at `forgiveness × D` with the others exact reads
   `D`. It is a power mean with exponent `ln n / ln forgiveness` for the `n` joints compared, so the sentence
   holds at any coverage. Below 1.1 the extra room is under what a pose estimate resolves, and the distance is
   the plain maximum.

Degrees, not tolerance units: the same pair reads the same number whatever the tolerance is, so the
tolerance can be tuned while watching the distance.

**Similarity**, 0 to 1, from the distance, with `angle_tolerance` as the slack at both ends: 1 within the
tolerance of identical, 0 within the tolerance of opposite, a straight line between. It is exactly 1 on the
plateau, so "fully alike" is the similarity reading 1, and noise between two matching players does not show.

The arms are alike, then, when every arm joint is within the tolerance, allowing one joint to stray to
`forgiveness` times the tolerance when the others are spot on.

## Two consumers

| | Hit sync | `Similarity` feature |
|-----------------------|----------------------------------------------------------|---------------------------------------------------------------|
| Component             | `HitSync` (`pose/hit_sync.py`)                           | `PostureSimilarity` (the module), wired in `main.py`          |
| Input                 | the hit player's LERP pose at the hit tick               | the current SMOOTH poses, every frame                         |
| Reads                 | the distance, and the similarity's plateau               | the similarity, 0 to 1                                        |
| Neutral               | a gate: every hit in the run has `ArmDeviation` 1        | a ramp: the pair times the smaller `ArmDeviation` (`pose/neutral_weight.py`) |
| In time               | none: the poses as they were at the hits                 | `SimilarityStickyFiller`, `SimilarityEuroSmoother` (SMOOTH), `SimilarityChaseInterpolator` (LERP) |
| Decision              | a run of hits is in sync when every pair's similarity is 1 and the gate holds; `min_players` in a row | none; each reader shapes it: the window opens from `PI.window.sync_threshold` to 1, eased |
| Output                | `HitStreak` on the board; `states.sync.hits`, `states.sync.distance` (°) | the rows on the LERP frames; `/pose/N/similarity/pose` to Max |

Both read one settings group, `pose.similarity.posture`, and one switch for neutral,
`pose.similarity.neutral_weight.enabled`. The feature's chain: `PostureSimilarity` runs on the SMOOTH
broadcast; its rows pass the sticky filler (a present pair's gap is held, a departed player's slot is NaN)
and the neutral weight, and are stamped on the next SMOOTH frames by `SimilarityApplicator`, smoothed, then
chased at LERP. The hit sync reads none of that, so a change to the similarity smoothers changes the windows
and the sound, never the sync. White Space produces no leader scores; `/pose/N/similarity/leader` carries
zeros.

## Settings

| Setting                                              | Reads it            | `studio.json` | Meaning |
|------------------------------------------------------|---------------------|---------------|---------|
| `pose.angle_calibrator.*`                            | both                | see `CALIBRATION.md` | 0 and π of every joint |
| `pose.angle.smoother` (`min_cutoff`, `cutoff_rise`)  | both                | 0.1, 0.7      | the angles both compare are these smoothed ones |
| `pose.similarity.posture.angle_tolerance`            | both                | 30°           | the slack at both ends; **the knob for how alike** |
| `pose.similarity.posture.forgiveness`                | both                | 1.4           | how far one joint may exceed the tolerance when the rest match |
| `pose.similarity.posture.joints`                     | both                | the arms      | which joints are compared and covered |
| `pose.arm_deviation_extractor.min_degrees`           | both                | 20°           | neutral up to here (weight 0) |
| `pose.arm_deviation_extractor.max_degrees`           | both                | 45°           | fully out from here: weight 1, and the sync's gate |
| `pose.arm_deviation_extractor.n_top`                 | both                | 1             | the N joints furthest from neutral, averaged |
| `pose.similarity.neutral_weight.enabled`             | both                | on            | the ramp on the feature and the gate on the sync |
| `pose.similarity.sticky.enabled`, `hold_scores`      | feature             | on, off       | a present pair's gap held |
| `pose.similarity.smoother` (`min_cutoff`, `cutoff_rise`) | feature          | 0.3, 1.0      | the Euro smoother on the stamped rows |
| `pose.similarity.interpolator` (`responsiveness`, `friction`) | feature    | 0.33, 0.05    | the chase at LERP |
| `states.min_players`                                 | hit sync            | 2             | alike hits in a row to spin up |
| `PI.window.sync_threshold`                           | window              | 0.9           | the feature value from which windows open, fully at 1 |
| `PI.window.width`                                    | window              | 33°           | the window before it opens |

## What the numbers mean

The similarity of a distance `d` at tolerance `T` is `(180 − T − d) / (180 − 2T)`, clipped to 0..1: at 30°,
one hundredth per 1.2°. A feature value `v` on a pair both fully out of neutral is therefore a distance of
`180 − T − v × (180 − 2T)`. The neutral weight `w` multiplies the feature, so a reader with a threshold `t`
needs `w > t` before anything passes, which is the joint furthest from neutral beyond
`min + t × (max − min)`.

### Hit sync (tolerance 30°, forgiveness 1.4, ramp 20°–45°)

| Meaning                                                                | Value |
|------------------------------------------------------------------------|-------|
| in sync: every arm joint within                                        | 30°   |
| in sync: one joint off, the other three exact                          | 42°   |
| the gate: every hit's joint furthest from neutral at least             | 45°   |

Each degree of tolerance moves the first by 1° and the second by 1.4°. The instrument's distinct poses are
90° apart on a joint (neutral, a T, raised, a folded elbow) and diagonal arms against a T are 45° apart, so
poses that sound different stay unlike at this tolerance.

### `Similarity` feature and the projection window (tolerance 30°, threshold 0.9, ramp 20°–45°)

| Meaning                                                                | Value |
|------------------------------------------------------------------------|-------|
| similarity 1: every arm joint within                                   | 30°   |
| similarity 0.5 (a T against arms hanging)                              | 90°   |
| similarity 0: from                                                     | 150°  |
| windows start to open, both fully out of neutral: within               | 42°   |
| windows half open (eased, feature 0.95): within                        | 36°   |
| windows fully open: within                                             | 30°   |
| window floor: nothing opens below (joint furthest from neutral)        | 42.5° |

The window reads the smoothed feature, so these are where it settles, not when. The similarity moves gently,
under a hundredth per degree, so its speed rarely lifts the Euro smoother off its floor: `min_cutoff` 0.3 Hz
is a time constant of about half a second at rest, and with `cutoff_rise` 1.0 about a fifth of a second while two
players' arms converge at 60°/s **(deduction from the 1€ filter's formula)**. The inputs are already
smoothed angles; this stage only takes the edge off.

## Interdependence

- **The tolerance is the definition.** It is the slack at both ends of the similarity and the sync's only
  knob. The similarity has no other shape parameter: a reader that wants a threshold or a curve applies its
  own, as the window does.
- **The pure similarity is high for most pairs.** With 0 pinned at opposite, two players in different poses
  commonly read 0.5 to 0.8. A reader that should respond only to alike pairs needs a threshold near 1; the
  window's 0.9 is that, and unthresholded it would stand half open between nearly everyone.
- **Neutral is a gate for the sync and a ramp for the feature.** `max_degrees` is both where the ramp
  reaches 1 and where a hit starts to count; `min_degrees` is only the ramp's foot. Moving `max_degrees`
  moves the window floor with it.
- **Forgiveness and tolerance.** Forgiveness widens the room of one joint only; the tolerance widens every
  joint's. One stubborn elbow is a forgiveness problem, a generally strict sync a tolerance problem.
- **Joints and coverage.** Deselecting a joint removes it from the comparison and from the coverage, so
  unseen legs do not lower an arm match. A pair is judged on the joints both have; the coverage is the score,
  not a factor. The deviation extractors keep their own joint lists (`ArmDeviationExtractor`,
  `LegDeviationExtractor`), independent of `joints`.
- **Angle smoothing feeds both.** `pose.angle.smoother` shapes the angles the feature compares (SMOOTH) and
  the angles the hit reads (LERP, after prediction and the chase). The similarity smoothers shape only the
  feature.
- **`max_players`** (root) is the `Similarity` row's width; a player id beyond it has no slot.
- **hd_trio** compares movement over time with `WindowSimilarity` (`window_similarity.py`), a bell kernel
  with its own remap; White Space does not use it. The joint selection (`joint_select.py`) is shared.

## Open

- Left compares to left, so two players mirroring each other read as unlike, in line with the instrument,
  where left and right shoulders drive different drawbars. If mirrored poses should count, the distance is
  the smaller of the two with the sides swapped.
- One tolerance for every joint. If elbows prove noisier than shoulders beyond what the forgiveness absorbs,
  a per-joint tolerance in the `joints` group.
- A pair hovering at the tolerance can break a streak on one flicker. With `min_players` 2 that is cheap;
  with three or more, hysteresis (staying in a run at slightly more than the tolerance).
- The arm deviation is itself a posture distance, from the neutral pose with forgiveness 1, and could share
  the distance function and the joint selection. Not done: little code gain, `ArmDeviation` would need a new
  name once its joints are selectable, `n_top` has no equivalent, and a degrees-from-neutral readout needs a
  feature of its own.
- A pair sharing very few joints is judged on those alone; a minimum coverage for the sync.
- `/pose/N/similarity/motion` (`modules/inout/osc_sound.py`) sends the plain `Similarity` row; its comment
  says motion-gated. `MotionGate` goes out separately on `/pose/N/similarity/gate`. Whether Max uses either.
