# White Space — Calibration

How the cameras, the fixture in beam mode, the fixture in projection mode and the sound agree on
**where** something is. Companion to `STATES.md` (the choreography) and `LAYERS.md` (the layers).

- **Read the Steps first.** Everything after them is reference: the *why*, for when a step fails.
- Everything here is code-verified unless marked **(site fact)** — told by the operator — or
  **(deduction)** — follows from the facts, not checked on hardware.
- Operator-facing angles are **degrees, 0–360, counter-clockwise seen from above**. The code keeps
  radians internally and on the wire.
- **The fixture firmware is never changed.** Everything is modelled and corrected app-side.
- Setting paths are written as they appear in the preset (`camera.mount.status`); `cam_N` means
  each of `cam_0`…`cam_3`.

---

## Steps

Calibrate in this order. Cameras first: everything else is tuned against the frame they define.

0. **Place by the layout.** Camera 0's sector starts at the connection side; speaker 0 stands on
   it; all numbered counter-clockwise. Placement *is* the room-side calibration — there is no
   camera offset to turn instead. See *Layout*.
1. **Cameras.**
   - Set `fov`, `resolution`, `tilt` and the three lens numbers (`lens_fov`, `lens_centre_x`,
     `lens_centre_y`) in the preset — `tilt` from the table under *Camera*, the lens from *The
     lens* — and leave `frame_height` at 0; then relaunch. All of them are baked in when the
     devices open, and the frame height is derived from the tilt at startup.
   - Read the open log: the `frame_height` line (the derived height), then per camera one
     `lens:` line (its field, centre offset and `lens_error`, which must match the table under
     *The lens*) and one `frame:` line (the elevation window, the horizon row, the rows covered).
   - Read the pinned `camera.mount.status`. It must say *mount OK*.
   - Level each camera against a spirit level, read its roll, type it into
     `camera.cam_N.readings.roll_offset`.
   - Look at the panorama row (always on; `render.panorama.parts` says which pieces draw). With
     someone standing near the middle of the room, the overlaps must coincide at head *and* knee
     height. If not, see *Reading the panorama*.
   - Tape at 50 cm on the wall in front of each camera: it must sit on the green horizon line. See
     *The horizon check*.
   - **The metres**, last of the camera step and the only part needing a person to walk: turn
     `track.foot_offset` until the label's `H` stops drifting as they walk out, then read
     `H`'s value to check `rig.camera_height`. Tape confirms it: standing on R 1.5 or R 3.5, the mark's
     foot tick must land on that edge of the zone field. See *Calibrating the metres*. Nothing in
     the show depends on this — only the readouts — so it may be deferred.
2. **Playhead offset** — `light.playhead.pulse_offset`. Beam mode (IDLE is fine), one person stands
   still; adjust until the flash fires as the beam crosses them.
3. **Projection offset, then interlace** — `inout.osc_light_sender.projection_offset`, then
   `.interlace`. Projection mode; a *static* line at the person (`pose_instrument`) must land on
   them — never the moving playhead line. Then adjust the interlace until a thin line is single on
   the wall, not doubled.
4. **Speakers** — nothing to tune when placed by the layout. In IDLE, Max voicing `/global/playhead`
   must follow the beam around the room.

Steps 2 and 3 may be done in either order.

**Re-check without re-tuning:** a spin-up (the projected playhead line continues where the beam
was); the flash on the first person at IDLE → INTRO; the sound on the beam.

| step | settings | readout | passes when |
|---|---|---|---|
| 1 cameras | `fov`, `resolution`, `frame_height`, `tilt`, `lens_fov`, `lens_centre_x`, `lens_centre_y`, `camera.cam_N.readings.roll_offset`, `track.rig.*`, `track.seam.*`, `track.reacquire_angle`, `track.foot_offset` | the open log's `lens:` and `frame:` lines; `camera.mount.status`; the panorama row | lens errors as tabled; mount OK; overlaps coincide at head and knee height; tape on the horizon line; `H` flat as a person walks out, then their real height; standing on taped R 1.5 and R 3.5 the foot tick lands on that zone edge; one person crossing a seam keeps one id and two overlapping fields |
| 2 playhead | `light.playhead.pulse_offset` | beam mode, `beam_flash`; `/pose/N/playhead/offset` | flash on the person; offset reads 0 at the crossing |
| 3 projection | `inout.osc_light_sender.projection_offset`, `.interlace` | projection mode, `pose_instrument` | static line on the person; single line on the wall |
| 4 speakers | `inout.osc_sound_sender.speaker_offset` (0) | IDLE, Max voicing `/global/playhead` | sound follows the beam |

**No gathered calibration panel.** Each setting stays with the code that applies it: the pulse
offset with the playhead under `light`, the projection offset and interlace with the sender under
`inout`, `speaker_offset` with the sound sender, the camera constants under `camera`.

**Units.** Degrees for every angle an operator reads or turns, in 0.1° steps (one ring pixel);
`speaker_offset` in steps of 1, since a speaker stand is not placed to a tenth of a degree. The
interlace is **not** an angle: the firmware shifts an integer pixel index, so it stays in pixels
(±10 px; 1 px = 0.1° is a remark, not a conversion).

---

## Theory

### One frame: azimuth

Every position in the system is an **azimuth**, an angle around the room. There is one azimuth
frame, and **azimuth 0 is the centre of the connection side** of the cube (see *Layout*).

The cameras produce it:

    azimuth = target_fov · cam_id + local_angle − fov_overlap        (modules/tracker/panoramic/geometry.py)

so azimuth 0 is the start of camera 0's sector. There is deliberately **no camera offset**: the
cameras are *placed* so that sector starts at the connection side. Placing them carefully has to be
done anyway, and a knob would only invite skipping it. If the flash is off in a room, move a
camera, don't turn an offset.

A person's bearing leaves the tracker as `Azimuth`; the playhead is an azimuth; every layer draws
at an azimuth's strip position (`angle_to_strip_position`: azimuth / 360 × 3600 pixels); every
angle Max receives is an azimuth.

That formula is the whole of it at `camera_radius = 0`. The cameras are **not** at the centre,
though, so one term is added on top — and where that term's distance comes from is the next
section.

### Why the azimuth does not use the measured distance

The cameras sit on a ring, 0.36 m out, aimed radially outward. So the same person is seen at
different bearings by two neighbours, and turning a camera bearing into a **rig-centre** azimuth
takes a triangle that needs a distance (`camera_local_to_azimuth`).

We can measure a person's distance — floor plane, off their feet — and we deliberately **do not use
it here**. The device tracker's box bottom sits below the feet, so the reading is short; and the two
cameras at a seam err in *opposite* directions, each pulling its bearing toward its own axis, so the
disagreement doubles. Measured, for one person at the cam0/cam1 seam:

| distance the correction uses | R 1.5 | R 2.25 | R 3.5 | worst |
|---|---|---|---|---|
| none — no correction at all | 23.1° | 14.5° | 9.0° | 23.1° |
| the measured one, reading ~50% short | 16.5° | 11.6° | 7.8° | 16.5° |
| **a fixed assumed depth, R 2.1** | **6.7°** | **~1°** | **6.0°** | **6.7°** |
| a perfect per-person distance | 0° | 0° | 0° | 0° |

A real body's *own* seam disagreement — one camera on the chest, the other on a shoulder — is
5.9° / 2.3° / 0.85° at those radii. So a fixed depth already sits at the irreducible floor,
while the number we can actually measure costs about 10° more than assuming one. The bias could be
calibrated away; the per-person **variance** could not, because a detector's box bottom is not a
physical landmark.

So the tracker assumes one depth, **`track.rig.parallax_radius`** — derived, never set:
the tracked zone's *harmonic* mean (`2·min·max/(min+max)` = R 2.1 for R 1.5 – R 3.5), because the
correction's term is linear in `1/d` and the minimax of that over an interval sits at the midpoint
of `1/d`. It is exact at R 2.1 and bounded by 6.7° everywhere in the zone.

**Consequences worth holding onto:**
- **Nothing behavioural is in metres.** The chain is `fov`, `tilt`, the lens, the frame → a local
  angle → one assumed depth → an azimuth, and every fusion gate is in degrees or a fraction. The
  metres that remain (`camera_radius`, `camera_height`, the zone) are tape measurements and
  declarations, never derived from a detector.
- **The box bottom cannot move a bearing.** Drag it 30% of the frame height and the azimuth is
  unchanged to nine decimals; only the reported metres move.
- **The guarantee is the zone's.** Outside R 1.5 – R 3.5 the residual keeps growing (7.1° at R 4, 8.5°
  at R 5, 10.1° at R 7, 11.7° at R 1.25) and nothing filters on distance, so a far person crossing
  a seam can still split. Not a regression — today they are worse everywhere — but the zone is now
  load-bearing and should be taped honestly.
- **`seam.link_angle` has to cover this plus the body**: 12.6° at R 1.5, ~3° at R 2.25, 6.9° at R 3.5.
  Those are two worst cases summed, and at R 1.5 the overlap is only 9.4° wide, so it bites within
  about 5° of a seam. See *Linking on a seam*.

### Direction

The bar turns **counter-clockwise seen from above** (site fact). The firmware's ring counter
advances with the bar and a strip index *is* that counter, so azimuth increases with the bar.

On the camera side azimuth increases with the image column, and every camera has `flip_h` set
(`INIT`) — so the columns must run counter-clockwise too, and the cameras are numbered
counter-clockwise (deduction: the installation works, and an offset can shift a mirrored frame but
never un-mirror it). A mirror is invisible with one person — one crossing per turn can always be
phased in. Checking it needs two people, or one walking along the sweep.

### Two modes, two offsets

The fixture puts light at an azimuth by two mechanisms. Both restart at the same reference: the
**sensor pulse**, once per revolution, when a reflective line on the head passes the sensor.

- **Beam mode** — commanded below `FIXTURE_PROJECTION_RPM` = 200 rpm (the firmware's `SLOW` flag).
  The four lamps are seen as beams. The pulse says where the bar is; the **playhead** tracks it;
  the **playhead offset** turns "angle since the pulse" into the front lamp's azimuth. It is tuned
  where it is visible — the flash landing on a person — so it includes the loop's delay at that
  speed, which is correct because it is tuned at the speed it runs at.
- **Projection mode** — commanded at or above 200 rpm. The bar is a blur and the image is painted
  from the firmware's own counter, restarted at the pulse, with a fixed quarter turn built in
  (`TEST = 900` px, `firmware.cpp:18`, applied at `:277-280`). The **projection offset** rotates
  the authored ring into that counter's frame. It is tuned on *static* content.

The fixture switches mechanism on the **commanded** rpm the moment it receives it
(`firmware.cpp:475`), regardless of the bar's actual speed.

**Why the projection offset is never tuned on the moving playhead line.** Tuning the pulse offset
makes the flash *land* on the person, so the internal playhead leads the visible beam by exactly
the loop's output delay. The playhead is never reset at spin-up, so it keeps that lead, and the
projected playhead line reaches the wall one output delay later — exactly where the beam would
have been. That is why the playhead needs no offset of its own in projection mode, and why tuning
the projection offset onto the moving line would rotate the whole ring by that delay. (The firmware
applies a frame on the next fast revolution, so the line may lag up to 30 ms more — a degree or
two, inside the flash window; deduction.) The hit and the sound fire on the internal playhead in
both modes, so their timing against the light matches too.

**Why they sit at opposite ends of the pipeline.** The playhead offset corrects a measurement
**coming in** — the flash, the hit, the state machine and the sound all consume it, so it is
applied at the source. The projection offset corrects an image **going out** — nothing reads the
rotated value back, so it is applied in the light sender. The two are independent; neither has to
be tuned first.

### The relation between the offsets

A readout, not a step. With θ the front lamp's azimuth at the pulse and *d* the loop's output
delay:

    projection offset = 90° − θ              (the 90° is the firmware's TEST = 900 px of 3600)
    playhead offset   = θ + d·rpm·6          (rpm·6 = degrees per second)
    sum               = 90° + d·rpm·6        θ cancels — the sum checks both at once

This build: 262.8° + 198.0° = 100.8°, so **θ = 252°** and the 10.8° residual is the delay — 50 ms
at 36 rpm, or 42–58 ms given the 3.6° slider both were tuned on. That is about what a fall message,
a 30 Hz tick and a frame on the wire cost.

### Interlacing

In projection mode each channel is painted twice per revolution: by the lamp on one end of the
bar and, half a turn later, by the lamp on the other end reading the pixel 1800 further on
(`firmware.cpp:277-280`). The two arms' LEDs are mounted out of phase, so one arm's LEDs fill the
other's gaps (site fact) — the two halves **interlace** into one image. That only works if the
lamps are exactly 180° apart and the blue pair exactly a quarter turn from the white; a degree off
shows as a doubled line. The four **interlace** values (`inout.osc_light_sender.interlace`, sent as
`/WS/o/0..3`, firmware `cor0..3`) shift each lamp's readout by a few pixels so the whites
interlace, the blues interlace, and the blue image sits on the white. Projection mode only: the
firmware ignores them below 200 rpm (`firmware.cpp:297-305`).

### Sound inherits the frame

Every position Max is sent is already an azimuth, so Max cannot disagree with the cameras through
anything in this repository; the only question is where speaker 0 stands, which the layout
settles. Max inherits the show's *timing* from the playhead offset: the hit that starts INTRO is
the beam crossing a person.

---

## Layout

The base is a cube holding the motor and electronics, with every connection on one face — the
**connection side** (site fact). It is the one physical reference the fixture carries everywhere,
so the layout is defined from it. Drawing: `White Space Layout Sheet.pdf` in this folder (A the
room, B the machine).

- **Azimuth 0 is the centre of the connection side**, increasing counter-clockwise.
- **Cameras on the corners**, pointing diagonally outward, 15 cm beyond the cube's corner (lens
  0.36 m from the axis, 0.50 m up). **Camera 0 at the corner counter-clockwise of the connection
  side**, then counter-clockwise. Camera *i* points at 90·*i* + 45, so every seam is a face centre.
  The 15 cm keeps the speakers out of frame: the nearest speaker corner is ≈76° off a camera's
  axis, outside its 63.5° half-field. The left lens sits 37.5 mm off the tripod thread — put the
  *lens* on the corner line.
- **Speakers parallel to the faces, 10 cm off them**, pointing out, not touching the fixture
  (site fact). **Speaker 0 on the connection side**, then counter-clockwise, so Max needs no
  constant.
- **Everything numbered from 0**, counter-clockwise from the connection side: `cam_0`…, `/pose/0`…,
  `white_0` / `blue_0`. (The firmware's comments count lamps from 1; internal only.)

        face:   spk 0 (az 0) — connection side · seam cam 3 | cam 0    corner: cam 0 (→ 45)
        face:   spk 1 (az 90)                  · seam cam 0 | cam 1    corner: cam 1 (→ 135)
        face:   spk 2 (az 180)                 · seam cam 1 | cam 2    corner: cam 2 (→ 225)
        face:   spk 3 (az 270)                 · seam cam 2 | cam 3    corner: cam 3 (→ 315)

**What the layout buys:** the two offsets stop being per-venue tunings. The playhead offset encodes
where the reflective line sits on the head relative to the front lamp, plus the loop delay; the
projection offset encodes the same plus the firmware's quarter turn. Both are properties of the
*build*. Re-tune them only after the head, a strip or a lens has been remounted; in a new room,
correct the placement, not the offsets.

---

## Camera

**Role:** defines the azimuth frame. Nothing aligns the cameras; everything aligns to them.

### Hardware

Luxonis OAK-D Pro W; the app runs `color = false` and uses **the left mono camera only**
(`SetupMono`, `modules/oak/camera/pipeline.py`): OV9282 W, global shutter, 1280 × 800, lens
**127° × 79.5°** by the spec, a little more by the units' own calibrations (see *The lens*). The
lens maps angle linearly to radius (equidistant): the factory calibrations confirm it to 0.5 %
out to 75° off axis, and the warp relies on it. The sensor table for every OAK variant in use,
with Luxonis links, lives beside the resolution tables in `modules/oak/camera/definitions.py`;
opening a device logs the sensor behind each socket and its lens.

The mono sensors carry an IR filter and do not see the light show (site fact); they use the 940 nm
flood (`camera.ir_flood_light`). Keep the dot projector off. `camera.mono_auto_exposure` applies
to all four.

### The camera frame

The warp (`warp_mesh_points`, `modules/oak/camera/definitions.py`) delivers a **levelled,
cylindrical** frame: a column is one azimuth at every height, a row is one elevation at every
column, and the rows are spaced by the **tangent** of the elevation. Column = azimuth is what the
tracker assumes. Tangent rows are for the pose: a narrow band of columns is then exactly a level
pinhole camera panned to that bearing, which is the kind of picture the detector and the pose model
were trained on, and a body keeps its proportions at any height in the frame. (Rows linear in
elevation, as before, shrink a metre at 40° up by 41 % against eye level, 67 % at 55°.)

**A straight line is curved unless it is vertical or at the horizon.** A ceiling edge `H` above the
lens on a wall `D` away sits at elevation `atan(H · cos b / D)` at bearing `b`: highest straight
ahead, falling toward the sides by tens of degrees. That is the projection, not a fault. The checks
are a vertical edge (one column, anywhere) and tape at lens height (one row).

Seven constants at the preset root define the frame, all `INIT` — baked in when the devices open,
so changed by editing the preset and relaunching:

- **`fov`** — 127°, **the azimuth span of the delivered frame**, quoted for the full sensor width.
  A contract, not a lens fact: the tracker's `cam_fov`, the overlaps and the azimuth of a column
  all follow from it. Each camera owns `target_fov = 360 / num_cameras`; the excess is overlap
  shared with its neighbours. It must not exceed the lens's field (`lens_fov`) or the edge columns
  read past the sensor and go black; the warp warns.
- **`lens_fov`, `lens_centre_x`, `lens_centre_y`** — the lens itself; see *The lens*.
- **`resolution`** — `P720` or `P800`. P720 is a pure vertical crop of the 800-row sensor, so the
  horizontal field and the whole column-to-azimuth mapping are identical; only the bottom of the
  window moves (4° higher, so the feet run out of frame sooner).
- **`frame_height`** — the delivered frame's rows, a multiple of 16. **0 means derived**: at
  startup `main.py` sets it to the sensor's full reach for the preset's mode, tilt and lens
  (`full_frame_height`), and logs it — 848 at P720 and tilt 0, **960** at P720 and tilt 15,
  **1152** at P800 and tilt 16. The rows are tangents, so the full reach needs more of them than
  the sensor has. A number in the preset is an explicit override, for when the tilt is final and
  a detector blob matching the frame is worth making. Getting it wrong costs, per direction: too
  many rows give a flat black bar of dead pixels across the top (113 rows at tilt 0 with 960);
  too few cut the **centre** of the top, the raised-arm zone on each camera's axis (4.5°, 45
  sensor rows, at tilt 15 with 848), while the sides, which never reached that high, lose nothing.
- **`tilt`** — up-tilt in degrees, positive = aimed up, shared by all four. The warp re-aims the
  camera by it.

**The window is pinned at the bottom.** The last row is the sensor's lowest reach on the centre
column — `tilt − 35.1°` at P720, `tilt − 39.2°` at P800, with the lens centre 10.5 px low — and the
rows run upward from there for as many as `frame_height` gives. Pinned there because the feet are
what the tilt rule pins — the floor-plane distance reads off the feet — and because the tangent
spends its rows at the top. At the full-reach height the top row is the sensor's top on the centre
column: **−20.1° to +52.3°** at P720 and tilt 15 on 960 rows, **−23.2° to +57.3°** at P800 and
tilt 16 on 1152. The horizon is therefore **not the centre row**: row 747 of 960 and row 894 of
1152 respectively (`horizon_row` ≈ 0.78), and `frame_window` (`definitions.py`) is the one place
that turns a row into an elevation or back. The open log prints the window per camera.

**Black is where the sensor did not look.** The sensor is a rectangle in the lens's own projection,
and that is not a rectangle in azimuth/elevation, so no rectangular window fills without cutting.
Tilting up lifts the centre column by the full tilt but the edge columns by less, so the sensor's
reach falls toward the sides (the table under *Tilt*), and the tangent rows magnify the difference
at the top: at P720, tilt 15, 960 rows the centre column is covered top to bottom, the seam columns
(45° off axis) from row **139**, the edge columns from row **256** — a black arch across the top,
deepest at the sides. `frame_coverage` says exactly which rows each column carries, and the open
log summarises it (centre, seams, edges). Downstream must ask it rather than assume the frame's top
row: a raised arm near a seam is judged against what the camera could see there.

`keystone` is the other installations' full-frame correction; it stays 0 here, and is exclusive
with `tilt`.

### The lens

Read off the four units' factory calibrations (mono CAM_B, 1280 × 800; the factory model is a
pinhole with a rational distortion polynomial, and on these lenses that polynomial keeps radius
linear in angle to 0.5 % out to 75°, so its focal and centre are the equidistant lens):

| unit | focal px/rad | centre x | centre y | field across 1280 px | `lens_error` |
|---|---|---|---|---|---|
| cam …F124D9D600 | 575.4 | 615.2 | 409.9 | 127.5° | **1.4°** |
| cam …110AD3D200 | 567.2 | 634.3 | 407.6 | 129.3° | 0.5° |
| cam …31DDD2D200 | 566.7 | 632.6 | 411.7 | 129.4° | 0.5° |
| cam …1136D1D200 | 565.0 | 634.5 | 410.3 | 129.8° | 0.6° |
| the `fov = 127` model | 577.5 | 639.5 | 399.5 | 127.0° | — |

Modelling the lens as `fov` read bearings ~1° short toward the seams — the "constant sideways
offset in the overlap" of the table under *Reading the panorama* — and put the horizon 10 px ≈ 1°
too high, which is ≈ 1 m of distance error at 5 m (see *The tracker's distance*).

**One lens for all four**, the mean, in the preset: `lens_fov` **128.9** (568.6 px/rad across
1280 px), `lens_centre_x` **−10.5**, `lens_centre_y` **+10.5** (px from the frame centre, at the
full sensor mode; the 720-row crop keeps the offset). Per unit rather than shared would have
removed the residuals in the last column; shared was the deliberate trade, and the residual stays
visible: at open each camera logs its own lens and `camera.cam_N.readings.lens_error`, the largest
bearing error it has under the shared lens. Unit F124's optical centre sits 24 px left of the frame
centre, so it keeps 1.4° at its azimuth zero; the other three are within 0.6°. Re-read the numbers
only after a lens or a unit is replaced.

The tracker's other constants, under `track.rig`: `camera_radius` (**R 0.36 m** — how far
each lens sits from the fixture axis, like every figure here), `camera_height`
(0.50 m), and the tracked zone `zone_min_radius` / `zone_max_radius` (R 1.5 – R 3.5). The first two
are **measured with a tape**, the zone is **decided and then taped**; none is tuned. Plus `seam.*`
(handover in the overlap). There is no distortion correction and no person-height assumption: the
projection is fixed in the warp, and distance comes off the floor plane.

### Tilt — derived from the build

`tilt` is derived, not set by eye, from the lens height, the ring radius and the **R 1.35 m** inner
circle (site decision).

The reference person is **1.8 m**, with an overhead hand reach of **2.2 m** — raised arms are
content the pose reads. On the R 1.35 m circle, on a camera's axis, they stand 0.99 m from the lens,
which puts their hands at +59.8°, head at +52.7° and feet at −26.8°. A camera aimed up by
`tilt` sees from `tilt − 39.2°` to `tilt + 41.3°` on its centre column (P800 with the shared lens;
35.1° and 37.3° at P720 — the sensor's own vertical field, which the frame delivers in full at the
derived `frame_height`); fitting hands *and* feet at R 1.35 would need 120° of it, against
80°. So the tilt is a trade, and **it is resolved in favour of the top**: losing the feet degrades
the distance estimate (it extrapolates the box bottom), losing the arms loses a gesture outright.

**The rule:** the feet are in frame from **R 1.5 m** — the inner edge of the calibrated play zone
(R 1.5 – R 3.5) — and all remaining room goes to headroom.

| `resolution` | sensor field | tilt | hands from | head from | feet from |
|---|---|---|---|---|---|
| **P800** | 79.4° | 13 | R 1.655 | R 1.35 | R 1.355 |
| | | **16** ← use | R 1.52 | R 1.245 | **R 1.5** |
| | | 18 | R 1.435 | R 1.18 | R 1.615 |
| | | 20 | R 1.355 | R 1.12 | R 1.755 |
| **P720** | 71.4° | **12** ← use | R 1.905 | R 1.54 | **R 1.5** |
| | | 14 | R 1.8 | R 1.46 | R 1.615 |
| | | 16 | R 1.7 | R 1.385 | R 1.755 |

P720 needs less tilt because its field is 8° narrower, so the feet run out of frame sooner — and
the feet are what the rule pins. P720 cannot reach the R 1.35 inner circle at any tilt: head
clipping inside about R 1.55 is expected there, not a fault. Beyond 16° at P800 each extra degree
costs more feet than it gains reach.

**The table is on-axis, and the reach falls toward the seams.** The sensor's top edge lifts by
less than the tilt off axis, so what a camera sees at tilt 16 (P800, the shared lens) by bearing:

| bearing off the camera axis | top | bottom | hands from | head from | feet from |
|---|---|---|---|---|---|
| 0° (axis) | 56.3° | −24.3° | R 1.5 | R 1.25 | R 1.45 |
| 30° | 53.8° | −24.6° | R 1.6 | R 1.3 | R 1.45 |
| 45° (seam) | 50.5° | −24.9° | R 1.75 | R 1.45 | R 1.45 |
| 63.5° (frame edge) | 44.1° | −25.3° | R 2.1 | R 1.7 | R 1.4 |

The feet rule holds all round; the overhead reach is a four-leaf pattern, R 1.5 on the axes and
R 1.75 at the seams. That is the mount, the same in any projection or frame height: a shorter frame
can only equalise it by cutting the middle. The frame delivers all of it at the full-reach
`frame_height` (1152 rows at P800 and tilt 16); at the sensor's own 800 rows the tangent rows would
cap the top at 43.7° everywhere — hands from R 2.1, head from R 1.7 — which is why the frame
is taller than the sensor.

**The panel shows this live, for the configuration actually running.** The tables above are the
design reference; `track.rig` is the check. After `hfov`, `vfov`, `tilt` and the frame's
two edge angles it publishes, as radii from the fixture:

| read-out | what it is |
|---|---|
| `feet_from` | nearest radius with the feet in frame — compare with `zone_min_radius` |
| `hands_from` | nearest radius with 2.2 m raised hands in frame, on a camera's axis |
| `hands_seam` | the same along a seam line, the worse of its two sides |

All three are measured against what the **sensor** fills per column (`frame_coverage`), not the
frame's rows, so the black arch counts. The gap between the two hands read-outs says which limit
the frame is running into. **Wide** — the rows reach past the sensor, so the top is the sensor's
own edge, falling toward the seams. **Near zero** — the rows run out first and cap the top at one
angle on every column, leaving only the ring's parallax. On the studio preset (P800, tilt 15, 960
rows of the 1136 the sensor could fill) it is the second: feet from R 1.47, hands from R 1.76 on
the axis and R 1.78 on the seam. The seam line is searched, not read off the 45° column: a person
on it near the fixture is seen by the camera ≈9° wider than the seam's own bearing, because the
camera sits 0.36 m out.

Three facts about what `tilt` moves, since they are easy to get wrong. **`angle_bottom` follows it
1:1**: the frame is pinned at the sensor's lowest reach, a lens constant below the tilt. **`angle_top`
does not** — +1° to +1.5° per +3° of tilt at a fixed `frame_height` (the taller the frame, the less)
— because a fixed number of tangent rows spend themselves at the top. And **the row scale does not depend on it at all** at a fixed
`frame_height`: that is `fov` and the frame's shape. Tilt reaches it only when `frame_height` is 0 and
the height is derived from the tilt.

### What the horizontal field allows

The vertical field decides whether a person *fits*; the horizontal field — `fov`, identical at
P720 and P800 — decides whether they are inside any camera's sector at all. It binds at the
**seams**, where a person is 45° off both neighbouring axes. Because each camera sits 0.36 m out
from the centre, its 127° covers less of the room as measured from the centre, worst up close:

| R | one camera's azimuth span | overlap at each seam |
|---|---|---|
| 1.0 m | 89.4° | **−0.6°** — the sectors do not meet |
| 1.35 m | 99.4° | +9.4° |
| 1.5 m | 102.2° | +12.2° |
| 2.25 m | 110.5° | +20.5° — the panorama's focus depth |
| 3.5 m | 116.4° | +26.4° |

- **R 1.015 m** — a seam person's centre enters one camera. Below it they are in the gap and
  invisible, which is why **R 1 m is the hard floor**: a consequence of the ring radius, not a
  choice.
- **R 1.765 m** — a whole body (50 cm shoulders) fits inside one camera. Between the two, a seam
  person is cut on one side in each camera and the box centre leans toward the visible side, ≈3°
  at R 1.35 m — a wobble at the handover, not a failure.

On a camera's own axis the horizontal field never binds (a whole body fits from R 0.485 m), so these
are purely seam properties.

### Reading the panorama

The second row is the whole ring as one 360° strip: azimuth 0 at the left edge, linear in degrees,
elevation up the side, both measured at the rig centre. The rows are the **tangent** of elevation,
as the camera frames' are, so a person or a ceiling edge has the same shape in the strip as in the
frames above it, and the only difference between the two is the azimuth re-projection to the rig
centre. A degree at the horizon is the same size either way; the grid's elevation lines spread
toward the top. (The strip is stitched from the frames through the tracker's published row model;
`panorama_map.strip_y` owns the strip's own rows.) The
four images are drawn on top of each other at the azimuth each camera claims, and **the tracker's
own view of the same people is drawn over them on the same vertical scale** — which is the point:
image right and marks wrong means the distance model, not the camera.

`render.panorama.parts` is a checklist of the pieces, each independent:

| part | what it draws |
|---|---|
| `image` | the four camera frames, stitched |
| `seams` | the seam rule defined on a camera's own frame: the dead zone (red bands) |
| `grid` | **every reference mark in the strip's own two axes**: the degree lattice, the sector boundaries (orange), the camera axes (blue), the overlap (yellow verticals), the green horizon, the yellow zone field, the labels, the footer |
| `observations` | a line per observation with a **foot tick** at the reported distance, inside a field as wide as the rule that governs it — and a **grey box** for every detection a filter dropped |
| `labels` | `#id cam az R distance H height` per observation; the filter's name above each grey box |

**A mark is the tracker's belief, not the picture, and its two axes use two depths on purpose.** Its
**x** is the fused `world_angle` — the number the light, the sound and the hit detector all receive
— which the tracker derives at `rig.parallax_radius` (R 2.1), never from a person's measured
distance; the tolerance field follows x onto that cylinder. Its **rows** go through the person's own
distance instead, which is what makes the foot tick exact against the zone field (*Two axes, and the
depth that varies*). The image under it is stitched at `focus_radius` (R 2.25), so a line sits a
small **constant** distance from its own pixels — a chosen consequence of deriving the parallax
depth from the zone rather than from a render slider, not a fault.

The strip asks four questions, and reading them in order says which number to reach for:

| what you read | what it tests |
|---|---|
| the two pictures coincide in an overlap | the lens and the mount — `lens_fov`, `fov`, `tilt`, roll |
| standing on taped R 1.5 and R 3.5, a mark's **foot tick** lands on that edge of the yellow field | the distance chain end to end: `rig.camera_height`, the zone's own radii, and `foot_offset`. The only metres on the strip, and exact — see *Calibrating the metres* |
| the **gap** between two lines of one colour at a seam | how far that person is from R 2.1 — a **depth indicator**, not an error. Zero on the cylinder, up to 6.7° at the zone's edges. A gap *larger* than that is the azimuth chain: `fov`, `tilt`, `camera_radius` |
| whether two fields of one colour overlap | the linking rules — the tracker will join exactly the pairs whose fields touch |

A **line, not a box**: the box's width said nothing its azimuth does not. Its bottom end carries the
**foot tick**, which is the picture of the `R` beside it and the one thing on the strip that can be
checked against a tape to the pixel. `H` is that person's height in metres
(see *The tracker's height*), and it is **the one number on the strip that checks itself**: a
seam's two observations are at different distances and so have different box heights in pixels, but
their `H` must agree — it is also exactly what `seam.link_height` compares. Two labels of one
colour showing different `H` means the distance model, the levelling or a camera's roll, before any
of it reaches the azimuth. The primary is opaque and 2 px; another camera's view of the same person
is half-lit and 1 px. A label's *height* is its id — the same index its colour comes from — so
labels never collide and never move as people do, and its x always sits on its own line.

**Nobody leaves the strip without a reason.**

- **A mark fading from its colour to grey is an identity the tracker holds but no longer counts** —
  a LOST observation: the device missed them, or they walked past the far edge (the label then ends
  in `past R3.5`). How grey it is says how close it is to being forgotten: their pose leaves the show
  after `emit_timeout`, the identity at `lost_timeout`, the moment it is fully grey.
- **A grey box is a detection the tracker dropped**, outlined at the detector's own size, with the
  filter named above it:

| tag | dropped because | the setting |
|---|---|---|
| `young` | the device has not held the track long enough | `track.age_filter` |
| `small` | the box is shorter than the minimum | `track.height_filter` |
| `dead zone` | a new person arriving right at a camera's field edge | `track.seam.dead_zone` |
| `past R3.5` | standing past the far edge | `track.zone_filter`, at `track.rig.zone_max_radius` |

So a person walking out keeps their own mark, fading, and when it has gone grey a grey box tagged
`past R3.5` takes its place for as long as the camera still sees them.

| what you see | what is wrong |
|---|---|
| the overlap coincides | nothing |
| aligns at head height but not at knee height | the mount — `tilt` or roll; read `camera.mount.status` to tell which |
| a constant sideways offset across the whole overlap | `lens_fov` (the lens, not `fov`, which is the frame's span) |
| a residual on one camera's seams only | that unit's `lens_error` — the shared lens's residual; F124 is expected to show ≈ 1.4° |
| a residual growing toward the frame edges on every camera | the lens is not equidistant after all — re-read the calibrations |
| two lines in one colour, side by side on a seam, opening and closing as they walk | nothing — that is the depth indicator; it closes at R 2.1 and opens to 6.7° at the zone's edges |
| two lines in one colour further apart than 6.7°, anywhere | the azimuth chain — `fov`, `tilt` or `camera_radius` |
| every line sits the same small distance beside its own pixels | nothing — the image is stitched at R 2.25 and the marks at R 2.1 (see above) |
| the lines coincide but the primary still jumps at the seam | `track.seam.hysteresis` |

The image is stitched for one assumed depth, `render.panorama.focus_radius` — **R 2.25 m**, the
middle of the play zone. It is exact there and ghosts by a bounded amount elsewhere (+3.9° at R 1.5,
−2.5° at R 3.5), so judge alignment with someone near the middle of the room. Nothing about a person
feeds the image, so nothing can fool it — and since the change that took the measured distance
out of the azimuth, nothing about a person feeds the *marks* either. Both now assume a depth; they
just assume slightly different ones, R 2.25 and R 2.1, for reasons each section gives.

### Two axes, and the depth that varies

**There is one space, with two axes, and everything on the strip uses them**: x is azimuth at the
rig centre, y is elevation at the rig centre (`strip_y`). There is no "image space" — the stitch
starts from a strip column, turns it into (azimuth, elevation), and works *backwards* into each
camera's frame, which is exactly what makes drawing data over pixels meaningful. Two things opt out
of one axis each and neither is a measurement: the dead zone uses x only and runs the full height,
and a label's y is a lane picked by `world_id` so labels never collide.

**What varies is the depth assumed to get into that space.** Both axes need an assumed distance to
turn a camera's view into the centre's, and each thing drawn names its own, for its own reason:

| drawn thing | x | y | depth |
|---|---|---|---|
| image (`image`) | centre azimuth | centre elevation | `focus_radius` |
| dead zone (`seams`) | centre azimuth | — full height | `focus_radius` |
| lattice, seam + axis lines (`grid`) | centre azimuth | — | **exact** |
| overlap verticals (`grid`) | centre azimuth | — | `parallax_radius` |
| horizon, zone field (`grid`) | — | centre elevation | **exact** |
| mark + foot tick (`observations`) | centre azimuth | centre elevation | x: `parallax_radius` · y: **the person's own distance** |
| label (`labels`) | centre azimuth | pixel lane by id | `parallax_radius` |

**The rule: two things on the strip are comparable only if they share an axis *and* a depth.** Every
misalignment this display has had was a pair that shared the axis and not the depth — the overlap
line 2.3° from where a mark's field actually switched, the marks beside their own pixels, the foot
tick 20 px off its own zone line. None was an axis confusion. In the code the depth parameter of
the shared functions is therefore called `depth_radius`, never after any one
caller's depth; `elevation_window` is the exception, because it is the strip's *single* y scale and
moves everything on it together.

*Exact* means no depth enters at all — a seam is a bearing, and a floor circle's depression is
`atan(camera_height / R)` whatever anything assumes — which is why those are the things a tape can
check. The two lines that cross the boundary are worth naming: the **dead zone** agrees with the
picture at every depth because the rule reads the raw image column and the stitch places the
picture through the same map, and the **foot tick** is exact against the zone field because going
through the person's own distance makes the lens height cancel algebraically
(`atan(tan(−atan(h/d))·d/R) = atan(−h/R)`, the zone line's own formula).

- **The overlap**, `rig.overlap` wide (**31.0°** on this rig), symmetric about each seam.
  Nothing tunable, and read it as one thing only: **exactly where a mark's tolerance field changes
  width**, from `seam.link_angle` inside to `reacquire_angle` outside. It is `angle_in_overlap`'s
  own threshold — a *local* angle of 28.3°, derived at the zone's far edge so the flag never
  under-reports — projected at `rig.parallax_radius`, the depth the marks are drawn at. That
  projection is the whole point: a local angle has no single position on the ring, and the same
  threshold drawn at the far edge instead would land 2.3° away, where nothing happens.

  Two things it is **not**, both easy to misread:
  - *not* the azimuth two cameras geometrically share at that depth — that is 19.4°, and the line
    is wider because the flag is deliberately generous (too wide costs nothing, too narrow splits
    a person);
  - *not* where the two **pictures** meet, which is 20.5° at R 2.25 and which the frames show for
    themselves. So the yellow lines sit outside the visible overlap, on purpose.

  It goes to zero below about R 1, where the sectors stop meeting at all (the table under *What
  the horizontal field allows*), and the lines then vanish — no threshold, nothing to mark.
- **The tracked zone**, a translucent field between `rig.zone_min_radius` and
  `rig.zone_max_radius`, at the depressions they subtend at the rig centre,
  `atan(camera_height / R)` — R 1.5 is −18.4° and R 3.5 is −8.1°. A floor circle of constant radius is
  a constant depression, so the zone is a band of rows, the same at every azimuth. **Tape the two
  circles on the floor and they must land on the field's two edges.** It is also what a person's
  mark is read against: a mark's line ends at the foot row, so someone inside the zone has that end
  inside the field. The field is clipped to the strip's window, and the clipping means something —
  at the studio preset it runs off the bottom, because the window bottom is −17.1° against R 1.5's
  −18.4° (the strip shows less than the frames do: `elevation_window` takes the band at its
  tightest column so no column fades to black). A fill that runs off the edge says *continues past
  here*, which a line pinned to the boundary row could not.
- **The dead zone**, `seam.dead_zone` in from each camera's own field edges, where *it* refuses to
  start a new person. It is a band in image space because the rule reads the raw local angle —
  literally the image column — and the stitch places the picture through the same map, so band and
  pixels agree **by construction at every depth**. That is what makes it checkable against the
  image anywhere rather than only at the focus radius. A person is born as long as **one** camera
  accepts them, so the region where nobody can be born is where two red bands **overlap**: at
  R 1.35 they do, on the seam; from about R 1.5 they no longer do.

The link tolerance is deliberately *not* a zone here — it is a property of a pair of observations,
not of a place — so it is drawn as a field around each mark instead. See *Linking on a seam*.

**`render.panorama.blend` — how the overlap combines.** `MAX` is the default and what the rest of
this procedure assumes; the others are second opinions on the same seam:

| mode | read it for |
|---|---|
| `MAX` / `MIN` | the plain picture; a ghost as a doubled bright edge, or a doubled dark one on bright content |
| `AVERAGE` | a ghost as a soft double image |
| `DIFFERENCE` | tune for **black** — the most sensitive. Each camera runs its own auto-exposure, so a brightness mismatch lifts the whole band off black: read the edges, not the level |
| `SPLIT` | **which way** it is wrong — one camera to red, the other to green; the leading fringe is the left camera |
| `STRIPE` | alternating columns, so a straight edge zigzags. Blind to exposure differences — use it when the two cameras disagree on brightness |

**A free check of the whole azimuth chain:** at R 2.25 m each camera should span **110.5°** of the
strip, not 127°. Set `track.rig.camera_radius` to 0 and every image should snap to
exactly 127° with 37° overlaps. It is a live slider, so this exercises the entire geometry in two
drags.

### Linking on a seam

A person on a seam is seen twice, and the tracker has to decide that the two sightings are one
person before anything else in the app sees them. Four settings do it, all in **degrees of world
azimuth or a fraction of a measured height** — never in metres, and never as a fraction of the overlap:

| setting | unit | studio | what it decides |
|---|---|---|---|
| `seam.dead_zone` | ° from a camera's field edge | 6.5 | where that camera refuses to start a new person |
| `seam.link_angle` | ° of world azimuth | 18 | how far apart two cameras' views may be and still be one person |
| `seam.link_height` | fraction of the larger `H`, 0–1 | 0.25 | a veto on that link when the two heights disagree |
| `seam.hysteresis` | ratio | 0.6 | how sticky the chosen primary is at a handover |
| `reacquire_angle` | ° of local angle | 5 | how far a returning person may be from the one just lost — *same camera*, so it sits beside `lost_timeout`, not under `seam` |

**Why azimuth and not distance.** Azimuth is the quantity this whole procedure verifies; the
distance estimate is not, and a person mid-jump sends it far past the zone. A gate in metres
would inherit both problems. So the link is decided on the one number the panorama proves.

**How far apart two honest sightings can be.** The two cameras see different parts of a body, and
the disagreement is the body's own width re-projected — it shrinks with distance:

| what the two cameras disagree about | R 1.5 | R 2.25 | R 3.5 |
|---|---|---|---|
| a torso (one camera on the chest, the other on a shoulder) | 5.9° | 2.3° | 0.85° |
| one arm held out sideways | 15.1° | 6.1° | 2.4° |

**But the body is only half the budget.** `link_angle` also has to cover the parallax residual, the
price of correcting the azimuth at one assumed depth rather than per person (*Why the azimuth does
not use the measured distance*). Both are worst cases, so summing them is conservative:

| | R 1.5 | R 2.25 | R 3.5 |
|---|---|---|---|
| body, arms down | 5.9° | 2.3° | 0.85° |
| parallax residual at R 2.1 | 6.7° | ~1° | 6.0° |
| **budget** | **12.6°** | **~3°** | **6.9°** |

At R 1.5 the overlap is only 9.4° wide, so the inner figure bites within about 5° of a seam. **Set it
from the display**: one person crossing a seam should keep two overlapping fields of one colour with
their arms down, and two people a metre apart should not overlap at all. A value that once looked
loose may simply have been paying for the residual honestly.

**Why the height gate is a fraction.** `H` is measured in metres and needs no re-projection (one
camera sees both feet and head), so two cameras at genuinely different distances agree on it while
their pixel box heights differ by tens of percent. Comparing `|H_a − H_b| / max(H_a, H_b)` is
scale-free — the lens height cancels out of it too — so the same 0.15 means the same thing at every
distance and after any change of frame. It is a **veto only**, and it is skipped whenever either
reading is not a measurement: `H` reads 0 when the feet are at or above the horizon and saturates
at 3 m when they barely clear it, and a jumper reads one of those. Nobody is harder to re-find than
mid-air, so the height is never allowed to refuse a link the azimuth supports.

**Nothing here is distance-based — not the gates, and not the azimuth either.** Every fusion
decision reads the raw local angle or a world azimuth corrected at one assumed depth; the measured
distance reaches none of them. The reason is one asymmetry, and it applies twice over. The overlap
band is derived at the zone's **far** edge rather than per person: someone truly at R 3.5, standing
27° in from their camera's field edge, is inside the R 3.5 band (28.3° of local angle) and gets a
cross-camera link attempt, where a band computed from a distance reading R 2.25 would be 22.8°, the
flag would come out false, and they would become **two people** at the seam. Too wide costs nothing
— `_find_world_candidate` runs and finds no partner within `link_angle`. Too narrow splits a person.
The same asymmetry keeps the link in degrees rather than metres, and (see the azimuth section) kept
the distance out of the parallax correction, where a 50%-short reading was manufacturing 11.6° of
seam disagreement at R 2.25.

One residual is visible and worth knowing: because a threshold on an image column cannot be a fixed
azimuth at every depth, a mark's field still widens about **4.8° before** the two pictures overlap
at R 2.25 — down from 12.3° when the band was the infinite-distance one, and exactly 0 at R 3.5 where
it is derived.

**What the dead zone costs, stated rather than fixed.** A person arriving on a seam inside about
R 1.5 sits within `dead_zone` of *both* field edges and is not picked up until they move (the two red
bands overlap there — see *Three coordinate systems*). That is chosen: half a person tracked from
one frame edge is worse than no person, and the play zone starts at R 1.35 for the separate reason
that the sectors stop meeting at R 1.0.

**The rule is drawn around the mark it governs.** Each observation's line sits inside a translucent
field of its own colour, the same height as the line, as wide as the tolerance that decides what
that observation may be joined to: `seam.link_angle` where a second camera also sees it,
`reacquire_angle` where none does. So the width says which rule owns that part of the ring, and
walking one person from mid-field to a seam visibly widens their field as the second camera picks
them up.

**Read it as a pair test.** The field is the tolerance wide rather than that much *either side* of
the line, and that is the whole point: both gates have the form `|Δ| ≤ angle`, so two fields each
`angle` wide touch at exactly the difference the gate allows. **Two fields of one colour that
overlap are two observations the tracker will join**; two colours that overlap are two people it
might confuse. (Fields of ±`angle` would overlap out to twice the gate and claim links that never
happen — at `link_angle` 18 they would show a link for two views 30° apart.)
`modules/render/tests/test_marks.py` asserts that drawn overlap and `_observations_match` agree
case for case, so the display cannot quietly drift from the rule.

Two things the field does not say, worth knowing rather than fixing:

- Both rules do apply inside an overlap — a re-acquisition is tried first, everywhere — but the
  cross-camera one is the one being tuned, so it is the one drawn there.
- The re-acquire rule is in a camera's *local* angle, and the strip is in azimuth, so that field
  converts through the person's own distance and measures **less** than `reacquire_angle` against
  the degree grid: `d / (d + r)` of it on axis, so 5° reads as 4.5° at 3 m and 4.7° at 6 m. The
  same conversion is applied to the positions, which is what keeps the pair test valid once drawn.

The footer prints all of it — `dead 6.5°  link 18.0°/15%  reacquire 5.0°` — so a width on screen
can be checked against the number that produced it.

### The horizon check

The green line is elevation 0: the level plane at lens height, 0.50 m. Anything at that height
lands on it at *any* distance — the parallax correction scales elevations, and zero stays zero —
so it is the one row in the panorama that is exact everywhere, not only at the focus radius.
It is also the zero the tracker measures distance from (see *The tracker's distance*).

**Tape at 50 cm on the wall in front of each camera** and read it against the line:

| the tape | what is wrong |
|---|---|
| on the line, all the way round | nothing |
| below the line in every camera | the cameras aim higher than `tilt` — raise it |
| above the line in every camera | the cameras aim lower than `tilt` — lower it |
| slants across one camera's image | that camera is rolled |
| on the line in one camera, off in its neighbour | those two disagree; that seam ghosts vertically |

**Without tape:** a standing person's knees are at about lens height. Standing still at a few
distances along one camera's axis, a gap that stays **constant** is the levelling error; a gap that
**grows as they come closer** is only their knee not being at 50 cm (5 cm is ≈3° at 1 m, ≈0.6° at
5 m). Judge standing still — a stride moves the knee.

### The mount readout

The warp models `tilt` only — it assumes the camera is not rolled about its optical axis. A rolled
camera tilts the horizon, which looks exactly like a wrong `tilt` in the panorama; the image cannot
tell them apart. The cameras can:

- **`camera.cam_N.readings.tilt_measured` / `roll_measured`** — from each board's IMU.
- **`camera.mount.status`** — pinned, always on screen: the average deviation from the preset, and
  a warning naming the worst camera when any exceeds `camera.mount.tolerance` (2°). If a board has
  no IMU it says *not measured*; then check roll by eye against a vertical edge.
- **`camera.cam_N.readings.roll_offset`** — what that camera reads when level. Subtracted from the
  raw reading, so the roll shown is how far the camera has moved since it was levelled. Per camera
  and never shared: it absorbs each unit's own sensor error, and those differ (cam_2 reads −2.25°
  sitting level; see *Site facts*). Tilt has no offset — there is no reference to calibrate it
  against on site.
- **`camera.cam_N.readings.fov_factory`** — the field this unit's own calibration spans across the
  frame (127.5–129.8° on this rig). A check that the sensor variant is the wide one (not ~97°),
  never alarmed on.
- **`camera.cam_N.readings.lens_error`** — the largest bearing error this unit has under the shared
  lens (see *The lens*). A property of the build, so also never alarmed on; it is expected to read
  1.4° on F124 and under 0.6° on the others, and a different number means a unit was swapped.

**Roll matters more than it looks: it doubles at the seams.** Neighbouring cameras see a seam on
opposite sides of their own centres, so the same roll moves the shared content in opposite vertical
directions — ≈**2.2° of vertical mismatch per 1.2° of roll**. A roll common to all four does not
cancel. The offset corrects the *reading*, not the image.

### The tracker's distance

Floor plane, on tangent rows: `camera_height · focal / (bottom_px − horizon_px)` — the rows below
the horizon *are* the tangent of the depression, so nothing is converted. The row model is the
frame's own (`frame_window`, derived by the tracker from the same camera fields the warp used)
and published under `track.rig` as the frame's two edge angles, `angle_bottom` and
`angle_top` — from which `panorama_map.row_model` rebuilds the row form exactly, so the panel shows
degrees and the panorama still draws with the tracker's own rows. Two limits:

- **It cannot see nearer than the picture reaches.** At the recommended tilt — 16° at P800 or 12° at
  P720 — the lowest row with picture is 23.7° below the horizon: 1.14 m from the lens, which is
  the R 1.5 m feet rule by construction. Anyone closer has their feet in the empty band, and the
  reading is the detector's extrapolated box bottom — a guess, not a measurement. A limit, not a
  fault, and the reason nothing filters on the near side.
- **It is weak at range.** Each degree of horizon error moves the reading by ≈0.15 m at 2 m,
  ≈0.9 m at 5 m, ≈1.7 m at 7 m. A tripod within ±1° still leaves roughly ±1 m at the far wall.

**The reading is not clamped.** It used to be pinned to the zone (1.14 m to 3.86 m from a lens),
back when it fed the parallax correction and a mangled box could throw a person's azimuth off.
The azimuth now uses a fixed depth, so the clamp guarded nothing — and a pinned reading is never
outside the zone, which made the far edge impossible to act on and drew everyone past it as
standing on it. Feet at or above the horizon read as infinitely far: not standing on this floor.

### The far edge

**Past `rig.zone_max_radius` the tracker does not see a person** — while `track.zone_filter`
is on (the switch sits with the tracker's other filters, `age_filter` and `height_filter`; off means
off: nothing is filtered at the far edge and nothing mentions it). That is the whole rule, and it is
handled exactly like a missed detection. On the panorama they show as a grey box tagged `past R3.5`:

| situation | what happens |
|---|---|
| someone first seen past the edge | never started — not born, not re-acquired, not linked at a seam |
| a tracked person steps past it briefly (a jump, feet hidden for a moment) | nothing visible: still emitted for `emit_timeout` |
| a tracked person walks out | stops driving the show after `emit_timeout`, forgotten after `lost_timeout` |
| they step back inside before that | the same person, same colour, same id |

No new setting: the timeouts are the ones a missed detection already uses, and `zone_max_radius` is
the edge — raise it to track further out (it also moves the overlap band and the parallax depth,
as it should: people are then tracked that far).

**What it measures.** The corrected foot row (the box bottom less `foot_offset`) turned into a
distance from the camera by the floor plane, then into a **radius from the fixture** — not a camera
distance. A camera sits `camera_radius` toward the person, so it always reads them nearer than their
radius: on its own axis at R 3.6 it reads 3.24 m, which a camera-distance test against 3.5 would
wrongly accept. The radius is also what two cameras at a seam agree on.

**Far edge only, and no margin.** Near the fixture the feet are often below the frame, so a near
test would judge the detector's guesses. A margin is not `foot_offset`: that corrects a bias, and
raising it makes everyone read *further*, so the filter would reject *more*. Brief errors are the
timeouts' job; steady ones are calibration.

**How precise it is**, on a camera's axis:

| | per pixel of foot row | per degree of horizon / tilt / roll |
|---|---|---|
| R 1.5 | 0.5 cm | 5 cm |
| R 3.5 | 3.4 cm | **35 cm** |

The far edge is the weak side, and angle errors dominate — roll doubles at the seams, so level the
cameras (*The horizon check*, *The mount readout*) before trusting it.

**Where it costs.** A jump or feet hidden behind someone make a person read *further*. Brief, that
is invisible. But a person near the far edge whose feet stay hidden for longer than `emit_timeout`
drops out of the show while still in view, and at a seam a second camera's view of someone jumping
is not linked until they land (the first camera still carries them).

**It fails open until `foot_offset` is calibrated.** An uncorrected box bottom makes everyone read
nearer, so the filter barely acts — nobody is wrongly dropped. Tune `foot_offset` by `H` staying
flat (*Calibrating the metres*), never by overshooting: too much makes people inside the zone read
past the edge.

**The box bottom is not the feet, and `track.foot_offset` is the correction.** The device
tracker puts its box bottom *below* the feet, by what measures as a fixed pad in pixels (≈94 px on
the studio frame, 0.098 of frame height). Uncorrected, `below` is too large and both readouts read
short. The bias is the box's, not the model's: nothing in our code touches the ROI
(`Tracklet.from_depthcam` is a field-for-field copy) and the floor-plane model is exact — so it is
corrected in exactly one place, `Geometry._foot_px`, which derives one foot row that both
`estimate_distance` and `estimate_height` read. **The ROI itself is never rewritten**, so
`height_filter` and the crop extractor still see the detector's own box.

It no longer touches the azimuth either way — see *Why the azimuth does not use the measured
distance* — so this setting is about making the metres readable, not about fusion. What consumes
the metres is the panorama's `R` label, the **foot tick**, and `seam.link_height`; that last gate
compares two readings of the **same** person and so survives a shared bias (1–2% across a seam).

**Recognising the cause rather than chasing it.** `R` and `H` share the denominator, so they move
together in a way that names the culprit — which is what makes `H` a usable calibration signal:

| cause | `R` | `H` | signature |
|---|---|---|---|
| wrong horizon row | short | short **by the same factor** | `H/R` constant |
| wrong `camera_height` | scales | scales | `H/R` constant |
| box a fixed **%** too tall | short by a constant % | wrong but **constant** with distance | `H` flat |
| box a fixed **px** too low | error **grows** with distance | **falls** with distance | `H/R` falls — **what `foot_offset` corrects** |

(An earlier version of this document said *"a box that misses the feet moves `R` alone"*. That is
false — `estimate_height` shares the denominator, so it moves both, and the fourth row is the real
signature.)

### The tracker's height

`Geometry.estimate_height`, on the same rows, is a **pure pixel ratio**:

    height = camera_height · box height / (rows from the horizon down to the feet)

The focal length, the field, the tilt and the distance all cancel, because the person and the
camera stand on one floor — the single-view horizon ratio, which only takes this form because the
rows are tangents. Three properties follow, and they are why it is worth having:

- **Scale-free.** The same person at 2 m and at 6 m reads the same metres from box heights that
  differ by more than a factor of two.
- **Parallax-free.** One camera sees the feet and the head, so nothing is re-projected to the rig
  centre. Two cameras at a seam therefore *must* agree, which is the check the label carries.
- **It reads reach, not stature.** The box top is the highest pixel, so arms up read ≈2.2 m where
  the same person reads 1.8 m with arms down. That is the number the tilt table is built around.

Accuracy is the distance's, in relative terms, since it is the same denominator: a pixel of box
noise is a centimetre, a degree of horizon error is 9 cm at 1.5 m and 35 cm at 7 m. It reads 0 when
the feet sit at or above the horizon, and is capped at 3 m — above anything a person can measure,
so the cap only ever catches a mangled box. Those two readings are **not measurements**, and
anything comparing two heights has to skip them: `seam.link_height` does (*Linking on a seam*),
which is what keeps a jumper from being refused a link.

It rides on the annotation, prints on the label, and is the one thing `seam.link_height` gates on;
nothing in the show consumes it yet.

#### Calibrating the metres — two stages, and the first needs no tape

The two stages fix different things and do not interact: stage 1 removes the **drift**, stage 2 the
**scale**. Do them in order, with `image` + `grid` + `observations` + `labels` on the panorama.

**1. Walk one person out and turn `track.foot_offset` until `H` stops drifting.** `H` is
distance-invariant by construction, so any drift is the detector's pad and nothing else. With the
offset at 0 a 1.8 m person on the studio rig prints:

| camera distance | 1.5 m | 2.5 m | 3.8 m |
|---|---|---|---|
| `H`, offset 0 | 1.37 | 1.22 | 1.08 |
| `H`, offset 0.098 | **1.80** | **1.80** | **1.80** |

It is **self-signing**: falling as they walk away means *increase*, rising means *decrease*. And
the model check is built in — if `H` is already flat but simply wrong, the bias is proportional
rather than a fixed pad, and this setting is the wrong shape for it (row 3 of the table above).
Don't read the absolute value yet; only the flatness.

**2. Then `H`'s absolute value checks `rig.camera_height`.** Once flat, `H` is the person's real
reach in metres and nothing else can be moving it. If a known 1.80 m person reads 1.65, the lens
height is out by the same factor — remeasure it rather than tuning `H` back.

**Absolute confirmation, with tape.** Tape the R 1.5 and R 3.5 circles and stand on each: the mark's
**foot tick** must land on that edge of the yellow zone field. This is exact, not approximate —
going through the person's own distance makes the lens height cancel
(`atan(tan(−atan(h/d))·d/R) = atan(−h/R)`), which is literally the formula the zone edge is drawn
from, so the two are the same kind of number to the pixel. `R` on the label should read the taped
distance at the same time — **the same number** as `rig.zone_max_radius` and the footer's `zone`,
with nothing to convert — and the azimuth should not have moved through any of it.

`foot_offset` is a fraction of frame height and a property of the **detector**, so it travels with
the detector and not the room; it scales with `resolution` like `height_filter` does
(`CameraResolution` names the list).

---

## Motor and sensor

The sensor pulses once per revolution when the reflective line on the head passes it; the firmware
forwards it as `/WS/sensor/fall` (only while commanded below 200 rpm) and restarts its ring counter
on it. `MotorController` measures phase and rpm from consecutive pulses (`light/motor.py`); the
phase is raw, 0 = the pulse, offset-agnostic by design. Above 200 rpm the sensor is silent: the show
anchors the spin-up on that silence (`ring_formed`) and the spin-down on the re-lock (`synced`) —
see `STATES.md`. Where the sensor or the line sit is not a calibration input; the playhead offset
absorbs it.

---

## Beam mode

**Role:** the playhead is the content clock in both modes; in beam mode it is also the bar's
heading as an azimuth, because the beams are where the bar points.

**What sets it:** the **playhead offset**, `light.playhead.pulse_offset`, degrees. The playhead NCO
tracks the measured motor phase while locked and adds it (`light/playhead.py`). (The per-pose
feature `PlayheadOffset` is a different thing in a different namespace.)

**What depends on it** — the number with the widest reach:
- `PlayheadOffset = azimuth − playhead` per pose (`pose/playhead_offset.py`): the flash layers fire
  on it, and the sound receives it (`/pose/N/playhead/offset`).
- The **hit** that starts INTRO is `PlayheadOffset` changing sign (`statemachine/machine.py`,
  `_detect_hit`). A wrong offset fires the intro early or late.
- The bar simulation on screen draws the four lamps at this heading.
- Max receives it as `/global/playhead`.

**Calibrating it** aligns all three consumers at once. The flash window is 11.5° wide in the preset.
Re-tune after changing `light.motor.beam_rpm`: the loop delay inside the offset scales with speed.

**Across a spin-up and a spin-down** the playhead is never reset; only its rate source changes
(`light/playhead.py`):

- *Spin-up:* it stops tracking the bar and free-runs at `beam_rpm` from wherever it was. The beam at
  azimuth θ becomes the `projection_playhead` marker at θ, continuing at the same rate. (What the
  wall shows during the acceleration is covered in `STATES.md`.)
- *Spin-down:* the bar lands at an angle unrelated to the content clock. The playhead keeps
  free-running until the sensor settles near `beam_rpm` (a two-stage re-lock gate), then `tracking`
  eases it onto the bar over roughly 1 / `tracking` ticks. Up to half a turn of re-alignment is
  physics, not calibration; the show hides it by exiting S8/S9 only once the lock is in.

---

## Projection mode

**Role:** the ring — four strips painting one 3600-pixel image around the room.

**What sets it:**
- The **projection offset**, `inout.osc_light_sender.projection_offset`, degrees — rotates the
  whole ring as it goes on the wire (`inout/osc_light_sender.py`), so the frame on the board stays
  azimuth-true. It absorbs the reflective line's position and the firmware's quarter turn.
- The **interlace**, `inout.osc_light_sender.interlace` (see *Theory*), ±10 px. The firmware boots
  with its own values (`cor2` 3, `cor1` 1); ours replace them on connect and once a second, so the
  firmware side is never where to tune.
- `light.brightness` and the sender's `curve` / `lower_edge` / `upper_edge` — brightness, not
  position.

**The four lamps**, relative to the front white in the bar's direction: back white +180°,
`blue[0]` −90°, `blue[R/2]` +90° — what `BEAM_LIGHT_HEADINGS` (`light/frame.py`) encodes. Standing
at the fixture facing along the front beam, `blue[0]` is on the right; the code and the fixture's
labels call it "left" (site fact), reading from the wall looking *at* the fixture. Which physical
strip is wired to `blue[0]` is a wiring fact worth one look.

---

## Layers

**Beam layers** write the four beam lights by name — `front_white`, `back_white`, `left_blue`,
`right_blue` on `Frame.beam_lights` — and nothing else (`layers/_base_layer.py`, `BeamLayer`). No
calibration of their own: the lamp shines where the bar points, and where that is as an azimuth is
the playhead. The sender copies them into the pixels the firmware reads in beam mode (pixel 0 and
1800 of each channel, `FIRMWARE_LIGHT_SLOT_TURNS`), which the projection offset and interlace
provably cannot reach. A beam layer that reacts to people (`beam_flash`, `beam_haunted`) depends on
`PlayheadOffset`.

**Projection layers** draw the ring at azimuth strip positions — `pose_instrument` at each person's
`Azimuth`, `projection_playhead` at the playhead (`ProjectionLayer`). No calibration of their own:
they author in azimuth, and the sender applies the projection offset and interlace on the way out.

---

## Sound (Max)

Max spatialises over the four speakers (site fact). It receives, all azimuths produced here:
`/global/playhead`, `/pose/N/azimuth`, `/pose/N/distance`, `/pose/N/playhead/offset`, plus the
state (`inout/osc_sound_sender.py`, `modules/inout/osc_sound.py`). With the speakers placed by the
layout there is nothing to tune on the Max side.

Two settings of our own, in `inout.osc_sound_sender`, sent in every bundle:

- **`speaker_offset`** — where speaker 0 stands, as an azimuth (`/global/speaker/offset`, radians on
  the wire). The correction stays on *our* side so every azimuth Max receives stays true; Max adds
  this one constant in its panner. Placed by the layout it is **0**.
- **`volume`** — main volume, 0–1 (`/global/volume`).

Neither is show state, so neither is zeroed on a blackout: a fader must read true whenever it is
turned, and a calibration must not snap to 0 between shows.

**The return path:** `/WS/sound/level` (left, right) → `beam_blue_sound` → the left and right blue
lamps (named after the fixture's blue-left / blue-right; nothing to do with stereo). No alignment;
the lamps turn with the bar.

---

## Screen

Two simulations share the `ws_light` row, and the render draws whichever matches the fixture's mode
(`render/render.py`): the **beam view** in beam mode draws the four lamps at the playhead heading;
the **ring view** in projection mode shows the ring buffer as it leaves the compositor. Both are
azimuth-true, because the projection offset is applied in the sender and never touches the frame on
the board.

**The check, no hardware needed:** in projection mode the ring view's playhead line must sit under
the beam view's front lamp and the tracker row's person, and a spin-up must not move it. The screen
shows the room's angles; only the wall shows what the fixture makes of them.

---

## Simulation

Nothing here applies to a simulated session, and no separate preset is needed: the offsets describe
the physical build, and a recording carries its own frame. The show compares a person's `Azimuth`
from the recording with the playhead from the simulated motor, both in one frame, so the flash, the
hit, the sound and both screen views are self-consistent.

- Existing recordings are 1280 × 720 raw clips, shot at `tilt = 0`. Set `resolution` to **P720** in
  the playback preset and the whole chain derives correctly — the frame height, the window and the
  horizon row, the distance estimate and the panorama's geometry. Left at P800 the warp expects an
  800-row clip and every frame-relative number is off (the simulator warns once).
- `camera.simulator.apply_warp` applies `tilt` to a raw clip. It assumes the clip was shot at exactly
  that tilt; capture-time tilt is not stored with clips, so old footage can carry a horizon error.

---

## Open

- **The frame fractions were tuned on 720 rows.** `track.height_filter` and
  `pose.distance_extractor.near_y` / `far_y` are fractions of the frame, and the frame is now
  taller and its rows tangents. Re-tune on the rig. (The seam rules are no longer among them — see
  *Linking on a seam* — but `height_filter` still is: a frame fraction whose meaning changes with
  every frame or tilt change, where in metres it would say what it means, "at least a 1 m person".
  Not changed yet, because it is a tuned value and swapping its unit is a rig session.)
- **`seam.link_angle` is loose at 18°.** It has to be, until the fields have been read on the
  rig. Shrink it until one person's two fields still overlap with their arms down at R 3.5 — the body
  table in *Linking on a seam* says 8° should be enough — and two people a metre apart do not
  overlap at all.
- **The zone's own two radii are declared, not measured.** `rig.zone_min_radius` /
  `zone_max_radius` (R 1.5 – R 3.5) now decide the overlap band and the far edge, so they are
  load-bearing rather than documentation. Tape both circles on the floor and check them against the
  yellow field's edges; the near one is currently below what the strip can show at P720 / tilt 15,
  so the field runs off the bottom there and R 1.5 has to be judged in the camera frames instead. If the room's
  usable area turns out different, change these rather than anything derived from them.
- **The panorama counts the black arch as covered.** The stitch culls by the frame's window (the
  centre column's reach), not per column, so in the top corners of a camera's field the comparing
  blends (`AVERAGE`, `DIFFERENCE`, `SPLIT`, `STRIPE`) mix black into the count. `MAX`, the mode the
  procedure uses, is unaffected: black loses to the neighbour. Exact culling would take a
  per-column coverage texture per camera from `frame_coverage`, published by each `Camera`.
- **A virtual camera per person for the pose** (maybe): the cylindrical crop is a level pinhole
  panned to the person; re-projecting the crop as a pinhole *pitched at* the person would be the
  most typical photograph the pose model could get. A per-crop warp in the crop extractor, no
  change to the shared frame. Worth an A/B on keypoint confidence at raised arms and close range.
- **Roll is not modelled by the warp.** `warp_mesh_points` takes `tilt` only, so a camera that
  is genuinely rolled still ghosts at its seams (≈2.2° vertical per 1.2° of roll). The mount readout
  says whether that is happening; the fix, if it is, is the tripod or a second rotation in the mesh.
- **`track.foot_offset` is built but still 0 — it is a measurement, waiting on rig time.**
  The mechanism is in place (`Geometry._foot_px`, shared by both readouts) and the instrument to
  tune it is on the panorama; what has not happened is a person walking out on the real rig. Until
  it does, `R` and `H` read short — the studio table under *The tracker's height* has the numbers.
  The procedure is there too, and stage 1 needs no tape.

  **It is display-only**, which is why it can wait: the change that took the measured distance out
  of the parallax correction means no bearing, no link and no identity depends on it — only the
  label's `R`, the mark's foot tick, and `seam.link_height`, which compares two readings of one
  person and so survives a shared bias.

  The one thing the walk-out can still disprove: if `H` comes out **flat but wrong**, the bias is
  proportional rather than a fixed pad and `foot_offset` is the wrong shape — it would have to
  become a fraction of box height instead. The measurements so far (≈94 px, constant) say fixed.

  Levelling is a smaller, separate term: tape at lens height on the far wall sits a few degrees off
  the horizon line, and the lens read found the horizon ≈1° high under the old model.

  **Metres and a floor plan still wait on this.** A plan view drawn from `R` and the azimuth would
  be the natural way to read this installation, and it is exactly the view that would be
  confidently wrong while the distance is.
- **The cam row draws the CROP box, not the tracker's ROI.** `BBoxRenderer` reads *stage* frames,
  whose `BBox` the crop extractor has already overwritten with the crop ROI — zoomed 1.1× and
  aspect-filled to 3:4 — so its bottom sits **10–27 px below** the tracker's real ROI bottom, and by
  a distance-dependent amount. Judging "the box is below the feet" up there therefore partly
  measures the crop expansion, not the detector. Not touched: it is a render concern, the crop box
  is the honest thing to draw for a crop, and the pose skeleton already draws the ankles if pixel
  truth about the feet is wanted. Tune `foot_offset` from `H` on the panorama instead, where the
  number being corrected is the number being shown.
- **`flip_v` is not applied to the row model.** The warp mirrors the delivered rows
  (`definitions.py`, `warp_mesh_points`), and `FrameWindow`'s own docstring says a caller must then
  read the horizon at `out_h − 1 − horizon_px`. `Tracker._set_frame` never does, and never even
  receives `flip_v` — so with it on, `Geometry` would compare an **un-flipped** horizon against
  *delivered* box rows and every distance and height would go wrong at once, with `link_height`
  silently ceasing to veto. Dormant: all four cameras are `flip_v: false`. (`flip_h` has no
  analogue — the column model is symmetric and the azimuth numbering already absorbed that flip;
  the row model is not, the horizon sitting at 0.78 of the frame and a mirror moving it to 0.22.)
  The fix: `frame_window` takes `flip_v` and returns the delivered window, with
  `horizon' = out_h − 1 − horizon` and **`focal' = −focal`** — a negative focal *is* the mirror, and
  `horizon − row = focal · tan(e)` then still holds. `Geometry.set_window`'s `max(1e-6, focal)`
  clamp would have to go, and `flip_v` would need sharing into the tracker like `tilt`.
- **`square = true` would break the distance silently.** That branch sets
  `setKeepAspectRatio(True)` on the detector's `ImageManip`, so the NN input is letterboxed and a
  normalized ROI row stops mapping linearly onto a delivered row — which is the one assumption
  `estimate_distance` makes about the ROI. The current preset is `square: false`, and the flag is
  there for other apps.
- **`pose.distance_extractor` has its own, separate bias.** It reads the **crop** bbox, which the
  crop extractor has already zoomed 1.1× and aspect-filled to 3:4, so the `/pose/N/distance` sent to
  Max is biased downward from a different cause than the tracker's metres, and by a different
  amount. It is a unitless 0–1 screen ramp, not metres, and nothing else reads it.
- **A placement aid** (maybe): since placement *is* the room-side calibration, projection layers
  that put the sector boundaries and centres on the wall would make it easier. The IMU cannot help
  with azimuth — its magnetometer is useless next to the motor and the LED strips.

---

## Site facts

**`R` is a radius from the fixture axis**, because that is where everything here is built and taped
from, and it is what the settings, the panorama's footer and a mark's `R` all carry — one number, no
halving. The only `Ø` left in this document is the `fixture:` line's own hardware: a ring or a tube
is described by its cross-section, as its supplier does, and converting those to radii would make
the doc harder to check against the parts, not easier. Leave them.

    reference:        the connection side of the cube                            (site fact)
    direction:        counter-clockwise seen from above                          (site fact)
    azimuth 0:        the centre of the connection side                          (rule — follows from the cameras)
    cameras:          on the corners, 15 cm beyond the cube, pointing diagonally;
                      camera 0 at the corner counter-clockwise of the connection
                      side, then counter-clockwise; camera R 0.36, height 0.50 m  (rule; taped)
    speakers:         parallel to the faces, 10 cm off, pointing out; speaker 0
                      on the connection side, then counter-clockwise             (rule)
    play zone:        R 1.35 m to R 3.5 m; hard floor R 1 m                        (site decision)
    tracked zone:     R 1.5 – R 3.5 = rig.zone_*; drives the overlap band and the
                      far edge. THREE notions in this doc: the play zone
                      (R 1.35 – R 3.5), the hard floor (R 1) and this, the
                      calibrated span, which is the one the code acts on  (site decision)
    panorama focus:   R 2.25 m
    seam overlap:     the FLAG is 28.3° of local angle, taken at the zone's far
                      edge so it never under-reports; drawn as 31.0° of azimuth
                      at the parallax depth, which is where a mark's field
                      switches. The two cameras' pictures share 20.5° at R 2.25,
                      9.4° at R 1.35, none by R 1 — a different number, and the
                      one the frames show                                       (derived)
    parallax depth:   R 2.1 = the zone's harmonic mean 2·1.5·3.5/(1.5+3.5). The one depth
                      the world azimuth is corrected at; exact there, 6.7° of seam
                      disagreement worst-case over R 1.5 – R 3.5                    (derived)
    far edge:         past zone_max_radius a person is not seen; emit_timeout,
                      then lost_timeout, as for a missed detection                (rule)
    seam births:      none on a seam inside ≈ R 1.5 at dead_zone 6.5           (chosen consequence)
    room:             8 × 8 m, machine in the middle                              (site fact)
    fixture:          cube 25 × 25 × 28 cm; rings Ø 25 / Ø 20 × 3.5 cm; tube Ø 20 × 154 cm;
                      head 9 × 9 cm; light from ≈ 32 cm                          (site fact)
    speakers:         25 × 25 × 33 cm, centres 35 cm from the axis                (site fact)
    camera:           OAK-D Pro W left mono, 127° × 79.5°; 10 × 3.5 × 3.5 cm body on a
                      50 cm tripod; IR filter, does not see the light             (spec / site fact)
    lens, shared:     lens_fov 128.9 (568.6 px/rad), centre (-10.5, +10.5) px    (mean of the four calibrations)
    lens, per unit:   see the table under The lens; F124 is the outlier (1.4°)   (read 2026-09-12)
    frame_height:     0 = derived at startup: 960 at P720 / tilt 15, 1152 at P800 / tilt 16 (the open log names it)
    drawing:          "White Space Layout Sheet.pdf", two A3 pages, to scale
    blue[0]:          wired to the strip labelled "blue left" / "blue right"     (confirm)
    LED strips:       the two arms' LEDs are mounted out of phase and interlace   (site fact)

    playhead offset:    262.8°                                   (tuned — re-check with the flash)
    projection offset:  198.0°                                   (tuned — re-check with the projected line)
    front lamp at pulse: θ = 252°                                (derived from the two)
    loop delay, beam:   ≈ 50 ms (42–58) at 36 rpm                (derived — the 10.8° residual)
    interlace:          white_1 +5, blue_0 −10, blue_1 +9 px     (tuned)
    resolution / tilt:  target P800, tilt 16°                    (the table under Camera; next recording)
                        the preset's P720 / tilt 15 is for playing back the old 720-row footage,
                        whose capture tilt is unknown; frame_height follows either at startup
    tilt, measured:     reads as configured on all four          (IMU)
    roll offsets:       cam_0 −1.00, cam_1 −0.91,
                        cam_2 −2.25, cam_3 −0.70                 (against a level; cam_2 is a sensor error)
