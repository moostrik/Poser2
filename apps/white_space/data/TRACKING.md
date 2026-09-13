# White Space — Tracking

How the cameras' detections become people with an azimuth, and why the tracker decides what it
decides. Companion to `CALIBRATION.md` (tuning the cameras and reading the panorama), `STATES.md`
and `LAYERS.md`. The code-side rules are in `.claude/rules/tracking.md`.

This document covers the panoramic tracker (`modules/tracker/panoramic/`), the one White Space runs.
The one-per-camera tracker (`modules/tracker/onepercam/`) is not described here.

- Everything here is code-verified unless marked **(site fact)** — told by the operator — or
  **(deduction)** — follows from the facts, not checked on hardware.
- Numbers assume the studio preset: 4 cameras on a ring of R 0.36 m at 0.50 m, `fov` 127, P720,
  tilt 15, the derived 960-row frame, the tracked zone R 1.5 – R 3.5. A table on another
  configuration names it.
- Every length is a radius from the fixture axis, as in `CALIBRATION.md`.

---

## Two tiers

1. **On each camera**, YOLO feeds depthai's `ObjectTracker` (`modules/oak/camera/pipeline.py`,
   `TRACKER_TYPE` in `definitions.py`). It detects people, associates them frame to frame, bridges
   short misses as LOST under the same id, de-duplicates, and hands out device ids.
2. **On the host**, the panoramic tracker (`modules/tracker/panoramic/`, `Tracker`) joins those
   per-camera tracks into people across the seams, chooses which camera's view speaks for each,
   and gives each an azimuth.

The host adds no motion model and no per-frame assignment. A problem inside one camera — an id
switch, a duplicate box — is a device-tracker setting first.

## Identity

### Observations and worlds

Each device track becomes an **observation** with a host-owned `obs_id` that is never reused. An
observation is immutable and always carries its own camera's box and angles. Observations are
grouped into **worlds**, one per person. A world id comes from a pool of `num_players` ids (6) and
is the id everything downstream uses (`/pose/N`); a freed id goes to the back of the queue, so it
is reused as late as possible.

A device id means something only while its device track lives. After REMOVED the device may give
the number to someone else, so the observation leaves the live index and stays behind as a LOST
**anchor** until `lost_timeout`. Anchors are what let a far camera link a person after the near
camera gave up on them. `ObservationStore` (`modules/tracker/panoramic/observations.py`) owns this.

### Intake

Each detection is handled by the first row that applies (`Tracker._add_tracklet`):

| # | the detection                                              | result                                  |
|---|------------------------------------------------------------|-----------------------------------------|
| 1 | REMOVED                                                    | the observation becomes an anchor       |
| 2 | LOST                                                       | the observation goes LOST, still live   |
| 3 | a filter rejects it (`young`, `small`, `past R3.5`)        | tracked: LOST; new: not counted         |
| 4 | already tracked                                            | refreshed, anywhere in frame            |
| 5 | same camera, within `reacquire_angle` of a not-active view | joins that view's world                 |
| 6 | within `seam.dead_zone` of the field edge                  | not counted (`dead zone`)               |
| 7 | in the overlap, matching another camera's view             | joins that view's world                 |
| 8 | no world id free                                           | not counted (`no id`)                   |
| 9 | anything else                                              | a new world                             |

- The filters come before every identity branch, so nobody is born, re-acquired or linked on a box
  a filter rejects. A tracked person a filter stops counting is handled as a missed detection:
  their box keeps following them, but `last_active` does not advance, so both timeouts run.
- Re-acquisition is matched on position alone. The device tracker has no appearance model, so a
  person it drops returns under a new id, and whatever made it drop them usually changed their box.
- The dead zone applies to births only. An existing person is refreshed and re-acquired inside it;
  starving them would freeze their angles and expire them while the camera still sees them.
- One camera never has two active views in one world: a link or re-acquisition skips a world the
  camera already sees, since the device de-duplicates and a second id is a second person.

### Each tick

`Tracker._update_and_notify`, after the intake:

1. Observations older than `lost_timeout` are retired.
2. **Split** (`Seams.split_worlds`). A world's active view that is more than 1.5 × `link_angle`
   from every other active view, or a second active view from the same camera, moves to a world of
   its own (the youngest first).
3. **Collapse** (`Seams.collapse_worlds`). Two worlds whose views match across cameras as mutual
   nearest neighbours merge; the older world keeps its id.
4. **Primary** (`Seams.pick_primary`) and **azimuth** (`Seams.world_azimuth`) per world, then one
   tracklet per world is emitted.

**Why split.** A seam link can join the wrong pair: two people arrive together and one camera counts
the second before the other camera does. Nothing about the link is wrong when it is made, so it is
repaired when the views disagree. The 1.5 margin above the link rule keeps a pair hovering at
`link_angle` from being split and collapsed on alternate ticks. Two active views from one camera
arise when a re-acquisition or link anchored on a LOST view whose device track then returns — a
person standing past the far edge while someone passes close by, for example. Collapse then
rejoins the detached view with its own person.

**The primary** is the view a world is emitted as. It supplies the box (the crop) and the camera:

| situation                           | primary                                                        |
|-------------------------------------|----------------------------------------------------------------|
| no view active                      | the most recently seen view                                    |
| the primary loses the person        | the active view furthest from its field edge, at once          |
| within `seam.hold` of that handover | unchanged                                                      |
| otherwise                           | a view whose edge distance ≥ the primary's ÷ `seam.hysteresis` |

A LOST primary's box stops updating, so the pose would drop a person another camera still sees;
that is why it is replaced at once. That forced handover skips the ratio and often picks the
worse-placed view, which the ratio would hand straight back when the first camera returns; `hold`
blocks that, so a one-frame miss costs one camera switch, not two. The ratio keeps an
active-to-active handover from bouncing.

**The azimuth** is the circular mean of the world's active views, each weighted by its distance
from its field edge less `seam.dead_zone`. A view can only link once it is that far in, so it
arrives weighing nothing, and a view walking out weighs nothing before it is lost: the azimuth moves
continuously through appearance, handover and disappearance. The two views at a seam disagree by
the parallax residual in opposite directions (*Why the azimuth does not use the measured distance*),
so the blend lies between them.

## Downstream

| consumer                   | what it takes                                                   |
|----------------------------|-----------------------------------------------------------------|
| `PosesFromTracklets`       | per world: `BBox` from the primary, `Azimuth` of the world      |
| `EyeAzimuthExtractor`      | shifts `Azimuth` to the eyes, in the primary's camera           |
| show layers, state machine | pose frames only (`LAYERS.md`, *Inputs*)                        |
| panorama                   | every observation and rejected detection; primaries by `obs_id` |

`PosesFromTracklets` poses a world while its primary's last detection is younger than
`pose.tracklets.detection_timeout`.

Two clocks follow a person the cameras lose:

| after                              | studio | what ends                                          |
|------------------------------------|--------|----------------------------------------------------|
| `pose.tracklets.detection_timeout` | 1.0 s  | the pose, so the person leaves the show            |
| `lost_timeout`                     | 2.0 s  | the identity: no more seam links or re-acquisition |

The tracker itself filters nothing on freshness; how stale is too stale is each consumer's call.

---

## The azimuth

### Why the azimuth does not use the measured distance

The cameras sit on a ring, 0.36 m out, aimed radially outward. The same person is seen at different
bearings by two neighbours, and turning a camera bearing into a rig-centre azimuth takes a triangle
that needs a distance (`camera_local_to_azimuth`, `modules/tracker/panoramic/projection.py`).

The tracker can measure a person's distance, off their feet on the floor plane, and does not use it
here. The detector's box bottom varies per person, so the reading is noisy; and the two cameras at a
seam err in opposite directions, each pulling its bearing toward its own axis, so the disagreement
doubles. Seam disagreement for one person on the cam0/cam1 seam:

| distance the correction uses           | R 1.5 | R 2.25 | R 3.5 | worst |
|----------------------------------------|-------|--------|-------|-------|
| none — no correction at all            | 23.1° | 14.5°  | 9.0°  | 23.1° |
| the measured one, reading ~50% short   | 16.5° | 11.6°  | 7.8°  | 16.5° |
| a fixed assumed depth, R 2.1           | 6.7°  | 1.0°   | 6.0°  | 6.7°  |
| a perfect per-person distance          | 0°    | 0°     | 0°    | 0°    |

A real body's own seam disagreement — one camera on the chest, the other on a shoulder — is 5.9° /
2.3° / 0.85° at those radii. A fixed depth therefore sits at the irreducible floor, while the
measured distance costs about 10° more. A bias could be calibrated away; the per-person variance of
a box bottom could not, because it is not a physical landmark.

So the tracker assumes one depth, `track.rig.parallax_radius`, derived and never set: the tracked
zone's harmonic mean, `2·min·max/(min+max)` = R 2.1 for R 1.5 – R 3.5. The correction is linear in
`1/d`, and the minimax of that over an interval sits at the midpoint of `1/d`. It is exact at R 2.1
and bounded by 6.7° everywhere in the zone (`Rig._update_parallax_depth`).

What follows from it:

- **Nothing behavioural is in metres.** The chain is `fov`, `tilt`, the lens, the frame → a local
  angle → one assumed depth → an azimuth, and every gate is in degrees, a fraction or seconds. The
  metres that remain (`camera_radius`, `camera_height`, the zone) are tape measurements and
  declarations, never derived from a detector.
- **The box bottom cannot move a bearing.** Drag it 30% of the frame height and the azimuth is
  unchanged to nine decimals; only the reported metres move.
- **The bound is the zone's.** Outside it the residual keeps growing: 11.7° at R 1.25, 7.1° at R 4,
  8.5° at R 5, 10.1° at R 7. Past the far edge `zone_filter` stops counting people; nearer than R 1.5
  nothing filters, so a near person crossing a seam can split. The zone is load-bearing and has to be
  taped as it is.
- **`seam.link_angle` has to cover the residual plus the body** — see *Linking on a seam*.

## Linking on a seam

A person on a seam is seen twice, and the tracker decides that the two sightings are one person
before anything else in the app sees them. The settings are in degrees of world azimuth, a fraction
of a measured height, or seconds — never in metres, and never as a fraction of the overlap:

| setting            | unit                          | studio | what it decides                                                 |
|--------------------|-------------------------------|--------|-----------------------------------------------------------------|
| `seam.dead_zone`   | ° from a camera's field edge  | 6.5    | where that camera refuses to start a new person                 |
| `seam.link_angle`  | ° of world azimuth            | 15     | how far apart two cameras' views may be and still be one person |
| `seam.link_height` | fraction of the larger `H`    | 0.2    | a veto on that link when the two heights disagree               |
| `seam.hysteresis`  | ratio                         | 0.6    | how sticky the primary is (*Each tick*)                         |
| `seam.hold`        | s                             | 0.5    | how long a forced handover blocks the way back (*Each tick*)    |
| `reacquire_angle`  | ° of local angle, same camera | 7      | how far a returning person may be from the one just lost        |

The link and re-acquire values are provisional (*Open*). How to set them from the panorama is in
`CALIBRATION.md`, *Reading the panorama*.

**Why azimuth and not distance.** Azimuth is the quantity the calibration verifies; the distance
estimate is not, and a person mid-jump sends it far past the zone. A gate in metres would inherit
both problems, so the link is decided on the one number the panorama proves.

**How far apart two honest sightings can be.** The two cameras see different parts of a body, and
the disagreement is the body's own width re-projected, shrinking with distance:

| what the two cameras disagree about                          | R 1.5 | R 2.25 | R 3.5 |
|--------------------------------------------------------------|-------|--------|-------|
| a torso (one camera on the chest, the other on a shoulder)   | 5.9°  | 2.3°   | 0.85° |
| one arm held out sideways                                    | 15.1° | 6.1°   | 2.4°  |

`link_angle` also has to cover the parallax residual. Both are worst cases, so summing them is
conservative:

|                            | R 1.5     | R 2.25   | R 3.5    |
|----------------------------|-----------|----------|----------|
| body, arms down            | 5.9°      | 2.3°     | 0.85°    |
| parallax residual at R 2.1 | 6.7°      | 1.0°     | 6.0°     |
| **budget**                 | **12.6°** | **3.3°** | **6.9°** |

At R 1.5 the budget exceeds the 12.2° the two pictures share there (`CALIBRATION.md`, *What the
horizontal field allows*), so near the fixture the link is tightest.

**Why the height gate is a fraction.** `H` is measured in metres and needs no re-projection, because
one camera sees both feet and head. Two cameras at genuinely different distances therefore agree on
it while their pixel box heights differ by tens of percent. `|H_a − H_b| / max(H_a, H_b)` is
scale-free — the lens height cancels out of it too — so one value of `link_height` means the same at
every distance and after any change of frame. It is a veto only, and it is skipped whenever either
reading is not a measurement (`height_is_measured`): `H` reads 0 when the feet are at or above the
horizon and saturates at 3 m when they barely clear it, and a jumper reads one of those. Nobody is
harder to re-find than mid-air, so the height never refuses a link the azimuth supports.

**Why the overlap band is taken at the far edge.** Every fusion decision reads the raw local angle or
a world azimuth at one assumed depth; the measured distance reaches none of them. The overlap band —
the local angle inside which a link is attempted — is derived once, at the zone's far edge, rather
than per person. Someone truly at R 3.5, 27° in from their camera's field edge, is inside the R 3.5
band (28.3° of local angle) and gets a link attempt. A band computed from a distance reading R 2.25
would be 22.8°, the flag would come out false, and they would become two people at the seam. Too wide
costs nothing — the link finds no partner within `link_angle`; too narrow splits a person. A
threshold on an image column cannot be a fixed azimuth at every depth, so a mark's field widens
5.8° before the two pictures overlap at R 2.1, and exactly 0 at R 3.5 where the band is derived.

**What the dead zone costs.** A person arriving on a seam inside about R 1.5 sits within `dead_zone`
of both field edges and is not picked up until they move (the red bands overlap there, `CALIBRATION.md`,
*Two axes, and the depth that varies*). Half a person tracked from one frame edge is worse than no
person. The play zone starts at R 1.35 for a separate reason: the sectors stop meeting at R 1.0.

---

## Distance and height

### The tracker's distance

Floor plane, on tangent rows: `camera_height · focal / (foot_px − horizon_px)`. The rows below the
horizon are the tangent of the depression, so nothing is converted. The row model is the frame's
own (`frame_window`, derived by `RigSync._set_frame` from the camera fields the warp uses) and is
published under `track.rig` as the frame's two edge angles, `angle_bottom` and `angle_top`, from
which `projection.row_model` rebuilds the row form exactly. Two limits:

- **It cannot see nearer than the picture reaches.** At the recommended tilt — 16° at P800 or 12° at
  P720 — the lowest row with picture is 23.7° below the horizon: 1.14 m from the lens, which is the
  R 1.5 feet rule by construction. Anyone closer has their feet below the frame, and the reading is
  the detector's extrapolated box bottom, a guess. For that reason nothing filters on the near side.
- **It is weak at range.** Each degree of horizon error moves the reading by ≈0.15 m at 2 m,
  ≈0.9 m at 5 m, ≈1.7 m at 7 m. A tripod within ±1° still leaves roughly ±1 m at the far wall.

The reading is not clamped: the far-edge filter, the panorama's `R` and the foot tick need a reading
past the zone to be past the zone. Feet at or above the horizon read as infinitely far — not standing
on this floor.

### The far edge

Past `rig.zone_max_radius` the tracker does not count a person while `track.zone_filter` is on (the
switch sits with the other filters, `age_filter` and `height_filter`; off means nothing is filtered
at the far edge and nothing mentions it). It is handled as a missed detection:

| situation                                         | what happens                                           |
|---------------------------------------------------|--------------------------------------------------------|
| someone first seen past the edge                  | never started: not born, re-acquired or linked         |
| a tracked person steps past briefly (jump, feet)  | nothing visible: posed from their last box             |
| a tracked person walks out                        | leaves the show, then is forgotten (*Downstream*)      |
| they step back inside before `lost_timeout`       | the same person, same colour, same id                  |

No new setting: the timeouts are the ones a missed detection already uses, and `zone_max_radius` is
the edge. Raising it tracks further out, and also moves the overlap band and the parallax depth.

**What it measures.** The corrected foot row turned into a distance from the camera by the floor
plane, then into a radius from the fixture. A camera sits `camera_radius` toward the person, so it
always reads them nearer than their radius: on its own axis at R 3.6 it reads 3.24 m, which a
camera-distance test against 3.5 would accept. The radius is also what two cameras at a seam agree
on (`Rig.beyond_zone`).

**Far edge only, and no margin.** Near the fixture the feet are often below the frame, so a near test
would judge the detector's guesses. A margin is not `foot_offset`: that corrects a bias, and raising
it makes everyone read further, so the filter would reject more. Brief errors are the timeouts' job;
steady ones are calibration.

**How precise it is**, on a camera's axis:

|       | per pixel of foot row | per degree of horizon / tilt / roll |
|-------|-----------------------|-------------------------------------|
| R 1.5 | 0.5 cm                | 5 cm                                |
| R 3.5 | 3.4 cm                | 35 cm                               |

The far edge is the weak side and angle errors dominate; roll doubles at the seams, so the cameras
are levelled (`CALIBRATION.md`, *The horizon check*, *The mount readout*) before this is trusted.

**Where it costs.** A jump or feet hidden behind someone make a person read further. Briefly, that is
invisible. A person near the far edge whose feet stay hidden for longer than `detection_timeout`
drops out of the show while still in view, and at a seam a second camera's view of someone jumping
is not linked until they land; the first camera still carries them.

**`foot_offset` sets which way it fails.** Too little makes everyone read nearer, so the filter
barely acts and nobody is wrongly dropped. Too much makes people inside the zone read past the edge.

### The box bottom is not the feet

The detector puts its box bottom below the feet by a fixed pad in pixels. `track.foot_offset` is the
correction, in frame heights: **0.03** on the studio frame, ≈29 px of 960 rows, walked out on the
rig **(site fact)**. Uncorrected, both read-outs read short. The bias is the box's, not the model's:
nothing touches the ROI (`Tracklet.from_depthcam` is a field-for-field copy) and the floor-plane
model is exact. It is corrected in one place, `Rig._foot_px`, which derives the foot row both
`estimate_distance` and `estimate_height` read. The ROI itself is never rewritten, so
`height_filter` and the crop extractor still see the detector's box.

The azimuth does not use it (*Why the azimuth does not use the measured distance*). What consumes the
metres is the panorama's `R` label, the foot tick, the far-edge filter and `seam.link_height`; that
last gate compares two readings of the same person and so survives a shared bias (1–2% across a
seam).

`R` and `H` share the denominator, so they move together in a way that names the cause:

| cause                  | `R`                       | `H`                       | signature        |
|------------------------|---------------------------|---------------------------|------------------|
| wrong horizon row      | short                     | short by the same factor  | `H/R` constant   |
| wrong `camera_height`  | scales                    | scales                    | `H/R` constant   |
| box a fixed % too tall | short by a constant %     | wrong, constant           | `H` flat         |
| box a fixed px too low | error grows with distance | falls with distance       | `H/R` falls      |

The last row is what `foot_offset` corrects.

`foot_offset` is a property of the detector, so it travels with the detector and not the room; it is
a fraction of frame height and scales with `resolution` like `height_filter` does
(`CameraResolution` names the list). The procedure is in `CALIBRATION.md`, *Calibrating the metres*.

### The tracker's height

`Rig.estimate_height`, on the same rows, is a pure pixel ratio:

    height = camera_height · box height / (rows from the horizon down to the feet)

The focal length, the field, the tilt and the distance all cancel, because the person and the camera
stand on one floor — the single-view horizon ratio, which takes this form because the rows are
tangents. Three properties follow:

- **Scale-free.** The same person at 2 m and at 6 m reads the same metres from box heights that
  differ by more than a factor of two.
- **Parallax-free.** One camera sees the feet and the head, so nothing is re-projected to the rig
  centre. Two cameras at a seam therefore must agree, which is the check the label carries.
- **It reads reach, not stature.** The box top is the highest pixel, so arms up read ≈2.2 m where the
  same person reads 1.8 m with arms down — the number the tilt table is built around.

Accuracy is the distance's, in relative terms, since it is the same denominator: a pixel of box noise
is a centimetre, a degree of horizon error is 9 cm at 1.5 m and 35 cm at 7 m. It reads 0 when the
feet sit at or above the horizon, and is capped at 3 m, above anything a person can reach, so the cap
only catches a mangled box. Those two readings are not measurements, and anything comparing two
heights skips them (*Linking on a seam*).

It rides on the annotation, prints on the label, and is the one thing `seam.link_height` gates on;
nothing in the show consumes it.

---

## Open

- **`seam.link_angle` is provisional at 15°**, and `link_height` 0.2 and `reacquire_angle` 7 with
  it. Shrink `link_angle` until one person's two fields still overlap with their arms down at R 3.5 —
  the body table in *Linking on a seam* says 8° should be enough — and two people a metre apart do
  not overlap at all.
- **`track.height_filter` is a fraction of the frame**, so its meaning changes with every frame height
  or tilt; in metres it would say what it means, "at least a 1 m person". It is a tuned value, so
  changing its unit takes a rig session.
- **`flip_v` is not applied to the row model.** The warp mirrors the delivered rows
  (`definitions.py`, `warp_mesh_points`), and `FrameWindow`'s docstring says a caller must then read
  the horizon at `out_h − 1 − horizon_px`. `RigSync._set_frame` never does, and never receives
  `flip_v`, so with it on the `Rig` would compare an un-flipped horizon against delivered box rows:
  every distance and height would go wrong at once, and `link_height` would silently stop vetoing.
  Dormant: all four cameras are `flip_v: false`. `flip_h` has no analogue — the column model is
  symmetric and the azimuth numbering absorbs that flip; the row model is not, the horizon sitting at
  0.78 of the frame and a mirror moving it to 0.22. The fix: `frame_window` takes `flip_v` and returns
  the delivered window, with `horizon' = out_h − 1 − horizon` and `focal' = −focal` (a negative focal
  is the mirror, and `horizon − row = focal · tan(e)` still holds); `Rig.set_window`'s
  `max(1e-6, focal)` clamp goes, and `flip_v` is shared into the tracker like `tilt`.
- **`square = true` would break the distance silently.** That branch sets `setKeepAspectRatio(True)`
  on the detector's `ImageManip`, so the NN input is letterboxed and a normalised ROI row stops mapping
  linearly onto a delivered row — the one assumption `estimate_distance` makes about the ROI. The
  studio preset is `square: false`; the flag is there for other apps.
- **A stalled camera (deduction).** A camera that stops sending leaves its views TRACKED with a frozen
  `last_active` until `lost_timeout`. If one is a primary, the pose ends at `detection_timeout` while
  another camera still sees the person, for up to `lost_timeout − detection_timeout` (1 s).
- **A dropped REMOVED message (deduction).** The tracklet queue is `maxSize=1, blocking=False`
  (`Camera`). If a REMOVED message were ever lost, the next holder of that device id (`SMALLEST_ID`)
  would inherit the world. Not observed; measure on the rig before guarding against it.
- **A plan view** (maybe): `R` and the azimuth drawn on a floor plan would be a natural way to read
  the installation, now that the metres are calibrated.
