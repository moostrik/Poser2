# White Space — Calibration

How the cameras, the fixture in beam mode, the fixture in projection mode and the sound agree on
where something is. Companion to `TRACKING.md` (how detections become people), `STATES.md` (the
choreography) and `LAYERS.md` (the layers).

- The Steps come first; everything after them is reference, for when a step fails.
- Everything here is code-verified unless marked **(site fact)** — told by the operator — or
  **(deduction)** — follows from the facts, not checked on hardware.
- Operator-facing angles are degrees, 0–360, counter-clockwise seen from above. The code keeps
  radians internally and on the wire.
- The fixture firmware is never changed. Everything is modelled and corrected app-side.
- Setting paths are written as they appear in the preset (`camera.mount.status`); `cam_N` means
  each of `cam_0`…`cam_3`.

---

## Steps

Calibrate in this order. Cameras first: everything else is tuned against the frame they define.

0. **Place by the layout.** Camera 0's sector starts at the connection side; speaker 0 stands on
   it; all numbered counter-clockwise. Placement is the room-side calibration — there is no camera
   offset to turn instead. See *Layout*.
1. **Cameras.**
   - Set `fov`, `resolution`, `tilt` and the three lens numbers (`lens_fov`, `lens_centre_x`,
     `lens_centre_y`) in the preset — `tilt` from the table under *Tilt — derived from the build*,
     the lens from *The lens* — leave `frame_height` at 0, and relaunch. All of them are baked in
     when the devices open, and the frame height is derived from the tilt at startup.
   - Read the open log: the `frame_height` line (the derived height), then per camera one `lens:`
     line (its field, centre offset and `lens_error`, which must match the table under *The lens*)
     and one `frame:` line (the elevation window, the horizon row, the rows covered).
   - Read the pinned `camera.mount.status`. It must say *mount OK*.
   - Level each camera against a spirit level, read its roll, type it into
     `camera.cam_N.readings.roll_offset`.
   - Look at the panorama row (always on; `render.panorama.parts` says which pieces draw). With
     someone standing near the middle of the room, the overlaps must coincide at head and knee
     height. If not, see *Reading the panorama*.
   - Tape at 50 cm on the wall in front of each camera: it must sit on the green horizon line. See
     *The horizon check*.
   - Set `render.panorama.focus_radius` to the `track.rig.parallax_radius` read-out.
   - The metres, last of the camera step and the only part needing a person to walk: turn
     `track.foot_offset` until the label's `H` stops drifting as they walk out, then read `H`'s value
     to check `rig.camera_height`. Tape confirms it: standing on R 1.5 or R 3.5, the mark's foot tick
     must land on that edge of the zone band. See *Calibrating the metres*. The metres decide the far
     edge and the read-outs; no bearing or link depends on them.
   - Seam settings: read the fields as one person crosses a seam. See *Reading the fields*.
2. **Playhead offset** — `light.playhead.pulse_offset`. Beam mode (IDLE is fine), one person stands
   still; adjust until the flash fires as the beam crosses them.
3. **Projection offset, then interlace** — `inout.osc_light_sender.projection_offset`, then
   `.interlace`. Projection mode; a static line at the person (`pose_instrument`) must land on them
   — never the moving playhead line. Then adjust the interlace until a thin line is single on the
   wall, not doubled.
4. **Speakers** — nothing to tune when placed by the layout. In IDLE, Max voicing `/global/playhead`
   must follow the beam around the room.

Steps 2 and 3 may be done in either order.

Re-check without re-tuning: a spin-up (the projected playhead line continues where the beam was);
the flash on the first person at IDLE → INTRO; the sound on the beam.

| step         | settings                                   | readout                                   | passes when                                   |
|--------------|--------------------------------------------|-------------------------------------------|-----------------------------------------------|
| 1 cameras    | frame constants, `track.*`, `focus_radius` | open log, `camera.mount.status`, panorama | every check in step 1                         |
| 2 playhead   | `light.playhead.pulse_offset`              | `beam_flash`, `/pose/N/playhead/offset`   | flash on the person; offset 0 at the crossing |
| 3 projection | `projection_offset`, `interlace`           | projection mode, `pose_instrument`        | static line on the person; single line        |
| 4 speakers   | `speaker_offset` (0)                       | IDLE, Max voicing `/global/playhead`      | sound follows the beam                        |

There is no gathered calibration panel. Each setting stays with the code that applies it: the pulse
offset with the playhead under `light`, the projection offset and interlace with the sender under
`inout`, `speaker_offset` with the sound sender, the camera constants under `camera`.

**Units.** Degrees for every angle an operator reads or turns, in 0.1° steps (one ring pixel);
`speaker_offset` in steps of 1, since a speaker stand is not placed to a tenth of a degree. The
interlace is not an angle: the firmware shifts an integer pixel index, so it stays in pixels (±10 px;
1 px is 0.1°).

---

## Theory

### One frame: azimuth

Every position in the system is an azimuth, an angle around the room. There is one azimuth frame,
and azimuth 0 is the centre of the connection side of the cube (see *Layout*).

The cameras produce it:

    azimuth = target_fov · cam_id + local_angle − fov_overlap

(`camera_local_to_azimuth`, `modules/tracker/panoramic/projection.py`), so azimuth 0 is the start of
camera 0's sector. There is no camera offset: the cameras are placed so that sector starts at the
connection side. Placing them carefully has to be done anyway, and a knob would invite skipping it.
If the flash is off in a room, move a camera, don't turn an offset.

That formula is the whole of it at `camera_radius = 0`. The cameras sit 0.36 m out, so the tracker
adds a parallax term at one assumed depth, `track.rig.parallax_radius` (R 2.1, derived from the
zone); why a fixed depth and not the measured distance is in `TRACKING.md`, *Why the azimuth does
not use the measured distance*.

A person's bearing leaves the tracker as their world azimuth (at a seam, a blend of both cameras'
views: `TRACKING.md`, *Each tick*), becomes the pose's `Azimuth`, and is moved to the eyes by
`EyeAzimuthExtractor`. The playhead is an azimuth; every layer draws at an azimuth's strip position
(`angle_to_strip_position`: azimuth / 360 × 3600 pixels); every angle Max receives is an azimuth.

### Direction

The bar turns counter-clockwise seen from above (site fact). The firmware's ring counter advances
with the bar and a strip index is that counter, so azimuth increases with the bar.

On the camera side azimuth increases with the image column, and every camera has `flip_h` set
(`INIT`) — so the columns run counter-clockwise too, and the cameras are numbered counter-clockwise
(deduction: the installation works, and an offset can shift a mirrored frame but never un-mirror it).
A mirror is invisible with one person — one crossing per turn can always be phased in. Checking it
needs two people, or one walking along the sweep.

### Two modes, two offsets

The fixture puts light at an azimuth by two mechanisms. Both restart at the same reference: the
sensor pulse, once per revolution, when a reflective line on the head passes the sensor.

- **Beam mode** — commanded below `FIXTURE_PROJECTION_RPM` = 200 rpm (the firmware's `SLOW` flag).
  The four lamps are seen as beams. The pulse says where the bar is; the playhead tracks it; the
  playhead offset turns "angle since the pulse" into the front lamp's azimuth. It is tuned where it
  is visible — the flash landing on a person — so it includes the loop's delay at that speed, which
  is correct because it is tuned at the speed it runs at.
- **Projection mode** — commanded at or above 200 rpm. The bar is a blur and the image is painted
  from the firmware's own counter, restarted at the pulse, with a fixed quarter turn built in
  (`TEST = 900` px in `firmware.cpp`, applied in `loop1`). The projection offset rotates the authored
  ring into that counter's frame. It is tuned on static content.

The fixture switches mechanism on the commanded rpm the moment it receives it (`loop` sets `SLOW`
from `RPM`), regardless of the bar's actual speed.

**Why the projection offset is never tuned on the moving playhead line.** Tuning the pulse offset
makes the flash land on the person, so the internal playhead leads the visible beam by exactly the
loop's output delay. The playhead is never reset at spin-up, so it keeps that lead, and the projected
playhead line reaches the wall one output delay later — exactly where the beam would have been. So
the playhead needs no offset of its own in projection mode, and tuning the projection offset onto the
moving line would rotate the whole ring by that delay. (The firmware applies a frame on the next fast
revolution, so the line may lag up to 30 ms more — a degree or two, inside the flash window;
deduction.) The hit and the sound fire on the internal playhead in both modes, so their timing
against the light matches too.

**Why they sit at opposite ends of the pipeline.** The playhead offset corrects a measurement coming
in — the flash, the hit, the state machine and the sound all consume it, so it is applied at the
source. The projection offset corrects an image going out — nothing reads the rotated value back, so
it is applied in the light sender. The two are independent; neither has to be tuned first.

### The relation between the offsets

A readout, not a step. With θ the front lamp's azimuth at the pulse and *d* the loop's output delay:

    projection offset = 90° − θ              (the 90° is the firmware's TEST = 900 px of 3600)
    playhead offset   = θ + d·rpm·6          (rpm·6 = degrees per second)
    sum               = 90° + d·rpm·6        θ cancels — the sum checks both at once

This build: 262.8° + 198.0° = 100.8°, so θ = 252° and the 10.8° residual is the delay — 50 ms at
36 rpm, or 42–58 ms given the 3.6° slider both were tuned on. That is about what a fall message, a
30 Hz tick and a frame on the wire cost.

### Interlacing

In projection mode each channel is painted twice per revolution: by the lamp on one end of the bar
and, half a turn later, by the lamp on the other end reading the pixel 1800 further on (`loop1`). The
two arms' LEDs are mounted out of phase, so one arm's LEDs fill the other's gaps (site fact) — the
two halves interlace into one image. That only works if the lamps are exactly 180° apart and the blue
pair exactly a quarter turn from the white; a degree off shows as a doubled line. The four interlace
values (`inout.osc_light_sender.interlace`, sent as `/WS/o/0..3`, firmware `cor0..3`) shift each
lamp's readout by a few pixels so the whites interlace, the blues interlace, and the blue image sits
on the white. Projection mode only: below 200 rpm `loop1` writes pixels 0 and 1800 without them.

### Sound inherits the frame

Every position Max is sent is already an azimuth, so Max cannot disagree with the cameras through
anything in this repository; the only question is where speaker 0 stands, which the layout settles.
Max inherits the show's timing from the playhead offset: the hit that starts INTRO is the beam
crossing a person.

---

## Layout

The base is a cube holding the motor and electronics, with every connection on one face — the
connection side (site fact). It is the one physical reference the fixture carries everywhere, so the
layout is defined from it. Drawing: `White Space Layout Sheet.pdf` in this folder (A the room, B the
machine).

- **Azimuth 0 is the centre of the connection side**, increasing counter-clockwise.
- **Cameras on the corners**, pointing diagonally outward, 15 cm beyond the cube's corner (lens
  0.36 m from the axis, 0.50 m up). Camera 0 at the corner counter-clockwise of the connection side,
  then counter-clockwise. Camera *i* points at 90·*i* + 45, so every seam is a face centre. The 15 cm
  keeps the speakers out of frame: the nearest speaker corner is ≈76° off a camera's axis, outside its
  63.5° half-field. The left lens sits 37.5 mm off the tripod thread — put the lens on the corner line.
- **Speakers parallel to the faces, 10 cm off them**, pointing out, not touching the fixture (site
  fact). Speaker 0 on the connection side, then counter-clockwise, so Max needs no constant.
- **Everything numbered from 0**, counter-clockwise from the connection side: `cam_0`…, `/pose/0`…,
  `white_0` / `blue_0`. (The firmware's comments count lamps from 1; internal only.)

        face:   spk 0 (az 0) — connection side · seam cam 3 | cam 0    corner: cam 0 (→ 45)
        face:   spk 1 (az 90)                  · seam cam 0 | cam 1    corner: cam 1 (→ 135)
        face:   spk 2 (az 180)                 · seam cam 1 | cam 2    corner: cam 2 (→ 225)
        face:   spk 3 (az 270)                 · seam cam 2 | cam 3    corner: cam 3 (→ 315)

**What the layout buys:** the two offsets are not per-venue tunings. The playhead offset encodes where
the reflective line sits on the head relative to the front lamp, plus the loop delay; the projection
offset encodes the same plus the firmware's quarter turn. Both are properties of the build. Re-tune
them only after the head, a strip or a lens has been remounted; in a new room, correct the placement,
not the offsets.

---

## Camera

**Role:** defines the azimuth frame. Nothing aligns the cameras; everything aligns to them.

### Hardware

Luxonis OAK-D Pro W; the app runs `color = false` and uses the left mono camera only (`SetupMono`,
`modules/oak/camera/pipeline.py`): OV9282 W, global shutter, 1280 × 800, lens 127° × 79.5° by the
spec, a little more by the units' own calibrations (see *The lens*). The lens maps angle linearly to
radius (equidistant): the factory calibrations confirm it to 0.5 % out to 75° off axis, and the warp
relies on it. The sensor table for every OAK variant in use, with Luxonis links, lives beside the
resolution tables in `modules/oak/camera/definitions.py`; opening a device logs the sensor behind
each socket and its lens.

The mono sensors carry an IR filter and do not see the light show (site fact); they use the 940 nm
flood (`camera.ir_flood_light`). Keep the dot projector off. `camera.mono_auto_exposure` applies to
all four.

### The camera frame

The warp (`warp_mesh_points`, `modules/oak/camera/definitions.py`) delivers a levelled, cylindrical
frame: a column is one azimuth at every height, a row is one elevation at every column, and the rows
are spaced by the tangent of the elevation. Column = azimuth is what the tracker assumes. Tangent rows
are for the pose: a narrow band of columns is then exactly a level pinhole camera panned to that
bearing, which is the kind of picture the detector and the pose model were trained on, and a body
keeps its proportions at any height in the frame. Rows linear in elevation would shrink a metre at
40° up by 41 % against eye level, 67 % at 55°.

A straight line is curved unless it is vertical or at the horizon. A ceiling edge `H` above the lens
on a wall `D` away sits at elevation `atan(H · cos b / D)` at bearing `b`: highest straight ahead,
falling toward the sides by tens of degrees. That is the projection, not a fault. The checks are a
vertical edge (one column, anywhere) and tape at lens height (one row).

Seven constants at the preset root define the frame, all `INIT` — baked in when the devices open, so
changed by editing the preset and relaunching:

- **`fov`** — 127°, the azimuth span of the delivered frame, quoted for the full sensor width. A
  contract, not a lens fact: the tracker's `cam_fov`, the overlaps and the azimuth of a column all
  follow from it. Each camera owns `target_fov = 360 / num_cameras`; the excess is overlap shared
  with its neighbours. It must not exceed the lens's field (`lens_fov`) or the edge columns read past
  the sensor and go black; the warp warns.
- **`lens_fov`, `lens_centre_x`, `lens_centre_y`** — the lens itself; see *The lens*.
- **`resolution`** — `P720` or `P800`. P720 is a pure vertical crop of the 800-row sensor, so the
  horizontal field and the whole column-to-azimuth mapping are identical; only the bottom of the
  window moves (4° higher, so the feet run out of frame sooner).
- **`frame_height`** — the delivered frame's rows, a multiple of 16. 0 means derived: at startup
  `main.py` sets it to the sensor's full reach for the preset's mode, tilt and lens
  (`full_frame_height`), and logs it — 848 at P720 and tilt 0, 960 at P720 and tilt 15, 1152 at P800
  and tilt 16. The rows are tangents, so the full reach needs more of them than the sensor has. A
  number in the preset is an explicit override, for when the tilt is final and a detector blob
  matching the frame is worth making. Getting it wrong costs, per direction: too many rows give a flat
  black bar of dead pixels across the top (113 rows at tilt 0 with 960); too few cut the centre of the
  top, the raised-arm zone on each camera's axis (4.5°, 45 sensor rows, at tilt 15 with 848), while
  the sides, which never reach that high, lose nothing.
- **`tilt`** — up-tilt in degrees, positive = aimed up, shared by all four. The warp re-aims the camera
  by it.

**The window is pinned at the bottom.** The last row is the sensor's lowest reach on the centre column
— `tilt − 35.1°` at P720, `tilt − 39.2°` at P800, with the lens centre 10.5 px low — and the rows run
upward from there for as many as `frame_height` gives. It is pinned there because the feet are what
the tilt rule pins — the floor-plane distance reads off the feet — and because the tangent spends its
rows at the top. At the full-reach height the top row is the sensor's top on the centre column:
−20.1° to +52.3° at P720 and tilt 15 on 960 rows, −23.2° to +57.3° at P800 and tilt 16 on 1152. The
horizon is therefore not the centre row: row 747 of 960 and row 894 of 1152 respectively
(`horizon_row` ≈ 0.78), and `frame_window` (`definitions.py`) is the one place that turns a row into
an elevation or back. The open log prints the window per camera.

**Black is where the sensor did not look.** The sensor is a rectangle in the lens's own projection,
and that is not a rectangle in azimuth/elevation, so no rectangular window fills without cutting.
Tilting up lifts the centre column by the full tilt but the edge columns by less, so the sensor's
reach falls toward the sides (the table under *Tilt — derived from the build*), and the tangent rows
magnify the difference at the top: at P720, tilt 15, 960 rows the centre column is covered top to
bottom, the seam columns (45° off axis) from row 139, the edge columns from row 256 — a black arch
across the top, deepest at the sides. `frame_coverage` says exactly which rows each column carries,
and the open log summarises it (centre, seams, edges). Downstream asks it rather than assuming the
frame's top row: a raised arm near a seam is judged against what the camera could see there.

`keystone` is the other installations' full-frame correction; it stays 0 here, and is exclusive with
`tilt`.

### The lens

Read off the four units' factory calibrations (mono CAM_B, 1280 × 800; the factory model is a pinhole
with a rational distortion polynomial, and on these lenses that polynomial keeps radius linear in
angle to 0.5 % out to 75°, so its focal and centre are the equidistant lens):

| unit                  | focal px/rad | centre x | centre y | field across 1280 px | `lens_error` |
|-----------------------|--------------|----------|----------|----------------------|--------------|
| cam …F124D9D600       | 575.4        | 615.2    | 409.9    | 127.5°               | **1.4°**     |
| cam …110AD3D200       | 567.2        | 634.3    | 407.6    | 129.3°               | 0.5°         |
| cam …31DDD2D200       | 566.7        | 632.6    | 411.7    | 129.4°               | 0.5°         |
| cam …1136D1D200       | 565.0        | 634.5    | 410.3    | 129.8°               | 0.6°         |
| the `fov = 127` model | 577.5        | 639.5    | 399.5    | 127.0°               | —            |

Modelling the lens as `fov` reads bearings ~1° short toward the seams — the constant sideways offset
in the overlap in the fault table under *Reading the panorama* — and puts the horizon 10 px ≈ 1° too
high, which is ≈ 1 m of distance error at 5 m (`TRACKING.md`, *The tracker's distance*).

One lens for all four, the mean, in the preset: `lens_fov` **128.9** (568.6 px/rad across 1280 px),
`lens_centre_x` **−10.5**, `lens_centre_y` **+10.5** (px from the frame centre, at the full sensor
mode; the 720-row crop keeps the offset). A lens per unit would remove the residuals in the last
column; a shared lens is the chosen trade, and the residual stays visible: at open each camera logs
its own lens and `camera.cam_N.readings.lens_error`, the largest bearing error it has under the shared
lens. Unit F124's optical centre sits 24 px left of the frame centre, so it keeps 1.4° at its azimuth
zero; the other three are within 0.6°. Re-read the numbers only after a lens or a unit is replaced.

The tracker's other constants, under `track.rig`: `camera_radius` (R 0.36 m — how far each lens sits
from the fixture axis, like every figure here), `camera_height` (0.50 m), and the tracked zone
`zone_min_radius` / `zone_max_radius` (R 1.5 – R 3.5). The first two are measured with a tape, the
zone is decided and then taped; none is tuned. The seam settings are in `TRACKING.md`, *Linking on a
seam*. There is no distortion correction and no person-height assumption: the projection is fixed in
the warp, and distance comes off the floor plane.

### Tilt — derived from the build

`tilt` is derived, not set by eye, from the lens height, the ring radius and the R 1.35 m inner circle
(site decision).

The reference person is 1.8 m, with an overhead hand reach of 2.2 m — raised arms are content the pose
reads. On the R 1.35 m circle, on a camera's axis, they stand 0.99 m from the lens, which puts their
hands at +59.8°, head at +52.7° and feet at −26.8°. A camera aimed up by `tilt` sees from
`tilt − 39.2°` to `tilt + 41.3°` on its centre column (P800 with the shared lens; 35.1° and 37.3° at
P720 — the sensor's own vertical field, which the frame delivers in full at the derived
`frame_height`); fitting hands and feet at R 1.35 would need 120° of it, against 80°. So the tilt is a
trade, and it is resolved in favour of the top: losing the feet degrades the distance estimate (it
extrapolates the box bottom), losing the arms loses a gesture outright.

**The rule:** the feet are in frame from R 1.5 m — the inner edge of the tracked zone (R 1.5 – R 3.5)
— and all remaining room goes to headroom.

| `resolution` | sensor field | tilt         | hands from | head from | feet from |
|--------------|--------------|--------------|------------|-----------|-----------|
| **P800**     | 79.4°        | 13           | R 1.655    | R 1.35    | R 1.355   |
|              |              | **16** ← use | R 1.52     | R 1.245   | **R 1.5** |
|              |              | 18           | R 1.435    | R 1.18    | R 1.615   |
|              |              | 20           | R 1.355    | R 1.12    | R 1.755   |
| **P720**     | 71.4°        | **12** ← use | R 1.905    | R 1.54    | **R 1.5** |
|              |              | 14           | R 1.8      | R 1.46    | R 1.615   |
|              |              | 16           | R 1.7      | R 1.385   | R 1.755   |

P720 needs less tilt because its field is 8° narrower, so the feet run out of frame sooner — and the
feet are what the rule pins. P720 cannot reach the R 1.35 inner circle at any tilt: head clipping
inside about R 1.55 is expected there, not a fault. Beyond 16° at P800 each extra degree costs more
feet than it gains reach.

The table is on-axis, and the reach falls toward the seams. The sensor's top edge lifts by less than
the tilt off axis, so what a camera sees at tilt 16 (P800, the shared lens) by bearing:

| bearing off the camera axis | top   | bottom | hands from | head from | feet from |
|-----------------------------|-------|--------|------------|-----------|-----------|
| 0° (axis)                   | 56.3° | −24.3° | R 1.5      | R 1.25    | R 1.45    |
| 30°                         | 53.8° | −24.6° | R 1.6      | R 1.3     | R 1.45    |
| 45° (seam)                  | 50.5° | −24.9° | R 1.75     | R 1.45    | R 1.45    |
| 63.5° (frame edge)          | 44.1° | −25.3° | R 2.1      | R 1.7     | R 1.4     |

The feet rule holds all round; the overhead reach is a four-leaf pattern, R 1.5 on the axes and R 1.75
at the seams. That is the mount, the same in any projection or frame height: a shorter frame can only
equalise it by cutting the middle. The frame delivers all of it at the full-reach `frame_height` (1152
rows at P800 and tilt 16); at the sensor's own 800 rows the tangent rows would cap the top at 43.7°
everywhere — hands from R 2.1, head from R 1.7 — which is why the frame is taller than the sensor.

The panel shows this live, for the configuration actually running. The tables above are the design
reference; `track.rig` is the check. After `hfov`, `vfov`, `tilt` and the frame's two edge angles it
publishes, as radii from the fixture:

| read-out     | what it is                                                              |
|--------------|-------------------------------------------------------------------------|
| `feet_from`  | nearest radius with the feet in frame — compare with `zone_min_radius`  |
| `hands_from` | nearest radius with 2.2 m raised hands in frame, on a camera's axis     |
| `hands_seam` | the same along a seam line, the worse of its two sides                  |

All three are measured against what the sensor fills per column (`frame_coverage`), not the frame's
rows, so the black arch counts. The gap between the two hands read-outs says which limit the frame is
running into. Wide: the rows reach past the sensor, so the top is the sensor's own edge, falling
toward the seams. Near zero: the rows run out first and cap the top at one angle on every column,
leaving only the ring's parallax. The studio preset (P720, tilt 15, the derived 960 rows) is the
first: feet from R 1.72, hands from R 1.67 on the axis and R 2.0 on the seam. The seam line is
searched, not read off the 45° column: a person on it near the fixture is seen by the camera ≈9°
wider than the seam's own bearing, because the camera sits 0.36 m out.

Three facts about what `tilt` moves. `angle_bottom` follows it 1:1: the frame is pinned at the
sensor's lowest reach, a lens constant below the tilt. `angle_top` does not — +1° to +1.5° per +3° of
tilt at a fixed `frame_height` (the taller the frame, the less) — because a fixed number of tangent
rows spend themselves at the top. And the row scale does not depend on it at all at a fixed
`frame_height`: that is `fov` and the frame's shape. Tilt reaches it only when `frame_height` is 0 and
the height is derived from the tilt.

### What the horizontal field allows

The vertical field decides whether a person fits; the horizontal field — `fov`, identical at P720 and
P800 — decides whether they are inside any camera's sector at all. It binds at the seams, where a
person is 45° off both neighbouring axes. Because each camera sits 0.36 m out from the centre, its
127° covers less of the room as measured from the centre, worst up close:

| R      | one camera's azimuth span | overlap at each seam                  |
|--------|---------------------------|---------------------------------------|
| 1.0 m  | 89.4°                     | −0.6° — the sectors do not meet       |
| 1.35 m | 99.4°                     | +9.4°                                 |
| 1.5 m  | 102.2°                    | +12.2°                                |
| 2.1 m  | 109.3°                    | +19.3° — the focus and parallax depth |
| 3.5 m  | 116.4°                    | +26.4°                                |

- **R 1.015 m** — a seam person's centre enters one camera. Below it they are in the gap and
  invisible, which is why R 1 m is the hard floor: a consequence of the ring radius, not a choice.
- **R 1.765 m** — a whole body (50 cm shoulders) fits inside one camera. Between the two, a seam person
  is cut on one side in each camera and the box centre leans toward the visible side, ≈3° at R 1.35 m
  — a wobble at the handover, not a failure.

On a camera's own axis the horizontal field never binds (a whole body fits from R 0.485 m), so these
are seam properties only.

### Reading the panorama

The second row is the whole ring as one 360° strip: azimuth 0 at the left edge, linear in degrees,
elevation up the side, both measured at the rig centre. The rows are the tangent of elevation, as the
camera frames' are, so a person or a ceiling edge has the same shape in the strip as in the frames
above it, and the only difference between the two is the azimuth re-projection to the rig centre. A
degree at the horizon is the same size either way; the grid's elevation lines spread toward the top.
(The strip is stitched from the frames through the tracker's published row model; `strip.strip_y`
owns the strip's own rows.) The four images are drawn on top of each other at the azimuth each camera
claims, and the tracker's own view of the same people is drawn over them on the same vertical scale:
image right and marks wrong means the distance model, not the camera.

`render.panorama.parts` is a checklist of the pieces, each independent:

| part     | what it draws                                                                  |
|----------|--------------------------------------------------------------------------------|
| `image`  | the four camera frames, stitched                                               |
| `seams`  | the dead zone (red bands), the seam rule defined on a camera's own frame       |
| `grid`   | every reference line in the strip's two axes, the degree labels and the footer |
| `marks`  | a mark per observation, and a grey line per detection a filter rejected        |
| `labels` | each mark's label: `#id cam az R distance H height`, or a rejection            |

The grid's lines are the degree lattice, the sector boundaries (orange), the camera axes (blue), the
overlap (yellow verticals), the green horizon and the yellow zone band. A mark is a line with a foot
tick at the reported distance, inside a field as wide as the rule that governs it.

**A mark is the tracker's view, not the picture, and its two axes use two depths.** Its x is that
camera's own world azimuth (`world_angle`), at `rig.parallax_radius`, never at a person's measured
distance; its field follows x onto that cylinder. What the tracker emits for a person seen by two
cameras is a blend of their views, which lies between their two lines; the show's azimuth is drawn by
the azimuth overlay. A mark's rows go through the person's own distance instead, which is what makes
the foot tick exact against the zone band (*Two axes, and the depth that varies*). The image is
stitched at `focus_radius`, set to the same R 2.1, so a line sits on its own pixels.

The strip asks four questions, and reading them in order says which number to reach for:

| what you read                                                   | what it tests                                                |
|-----------------------------------------------------------------|--------------------------------------------------------------|
| the two pictures coincide in an overlap                         | the lens and the mount: `lens_fov`, `fov`, `tilt`, roll      |
| on taped R 1.5 and R 3.5, the foot tick lands on that zone edge | the distance chain: `camera_height`, the zone, `foot_offset` |
| the gap between two lines of one colour at a seam               | how far that person is from R 2.1                            |
| whether two fields of one colour overlap                        | the linking rules (*Reading the fields*)                     |

The foot tick is the only metres on the strip, and exact (*Calibrating the metres*). The gap is a
depth indicator, not an error: zero on the R 2.1 cylinder, up to 6.7° at the zone's edges
(`TRACKING.md`, *Why the azimuth does not use the measured distance*). A gap larger than that is the
azimuth chain: `fov`, `tilt`, `camera_radius`.

A line, not a box: the box's width says nothing its azimuth does not. Its bottom end carries the foot
tick, which is the picture of the `R` beside it and the one thing on the strip that can be checked
against a tape to the pixel. `H` is that person's height in metres (`TRACKING.md`, *The tracker's
height*), and it is the one number on the strip that checks itself: a seam's two observations are at
different distances and so have different box heights in pixels, but their `H` must agree — it is
also what `seam.link_height` compares. Two labels of one colour showing different `H` means the
distance model, the levelling or a camera's roll, before any of it reaches the azimuth. The primary
view is opaque and 2 px; a passive view — another camera's view of the same person — is slightly
dimmer and 1 px, its field only outlined, so at a seam the passive field's edges show inside or beyond
the primary's fill. A label's height on the strip is a lane picked by its id — the same index its
colour comes from — so labels never collide and never move as people do, and its x always sits on its
own line.

**Nobody leaves the strip without a reason.**

- A line fading to grey, its field fading out, is an identity the tracker holds but no longer counts —
  a LOST observation: the device missed them, or a filter stopped counting them (the label then ends
  in the rejection, `past R3.5` or `small`). How far it has faded says how close it is to being
  forgotten: their pose leaves the show after `pose.tracklets.detection_timeout`, the identity at
  `lost_timeout`, the moment the line is fully grey and the field gone.
- A grey line with no field is a detection the tracker rejected, labelled with the rejection at its
  top:

| label       | rejected because                               | the setting                                         |
|-------------|------------------------------------------------|-----------------------------------------------------|
| `young`     | the device has not held the track long enough  | `track.age_filter`                                  |
| `small`     | the box is shorter than the minimum            | `track.height_filter`                               |
| `dead zone` | a new person arriving at a camera's field edge | `track.seam.dead_zone`                              |
| `past R3.5` | standing past the far edge                     | `track.zone_filter`, at `track.rig.zone_max_radius` |
| `no id`     | a new person while every world id is in use    | `num_players`                                       |

So a person walking out keeps their own mark, fading, and when it has gone grey a grey line labelled
`past R3.5` takes its place for as long as the camera still sees them.

| what you see                                              | what is wrong                                                |
|-----------------------------------------------------------|--------------------------------------------------------------|
| the overlap coincides                                     | nothing                                                      |
| aligns at head height but not at knee height              | the mount — `tilt` or roll; `camera.mount.status` says which |
| a constant sideways offset across the whole overlap       | `lens_fov` (the lens, not `fov`, which is the frame's span)  |
| a residual on one camera's seams only                     | that unit's `lens_error`; F124 shows ≈ 1.4°                  |
| a residual growing toward the frame edges on every camera | the lens is not equidistant — re-read the calibrations       |
| two lines of one colour on a seam, opening and closing    | nothing — the depth indicator, 0 at R 2.1, 6.7° at the edges |
| two lines of one colour further apart than 6.7°, anywhere | the azimuth chain — `fov`, `tilt` or `camera_radius`         |
| every line sits beside its own pixels                     | `focus_radius` differs from `track.rig.parallax_radius`      |
| the strong line flips between two cameras at a seam       | `track.seam.hysteresis`, `track.seam.hold`                   |

The image is stitched for one assumed depth, `render.panorama.focus_radius`, R 2.1 — the tracker's
parallax depth, so marks and pixels agree. It is exact there and ghosts by a bounded amount elsewhere
(+3.3° at R 1.5, −3.0° at R 3.5, per camera), so judge alignment with someone near the middle of the
room. Nothing about a person feeds the image or the marks' x, so nothing about a person can fool
either. `focus_radius` is a render setting and does not follow the zone: after changing the zone, set
it to the new `track.rig.parallax_radius`.

### Two axes, and the depth that varies

There is one space, with two axes, and everything on the strip uses them: x is azimuth at the rig
centre, y is elevation at the rig centre (`strip_y`). There is no image space — the stitch starts from
a strip column, turns it into (azimuth, elevation), and works backwards into each camera's frame,
which is what makes drawing data over pixels meaningful. Two things opt out of one axis each and
neither is a measurement: the dead zone uses x only and runs the full height, and a label's y is a
lane picked by `world_id`.

What varies is the depth assumed to get into that space. Both axes need an assumed distance to turn a
camera's view into the centre's, and each thing drawn names its own:

| drawn thing                            | x              | y                | depth                                          |
|----------------------------------------|----------------|------------------|------------------------------------------------|
| image (`image`)                        | centre azimuth | centre elevation | `focus_radius`                                 |
| dead zone (`seams`)                    | centre azimuth | full height      | `focus_radius`                                 |
| lattice, seam and axis lines (`grid`)  | centre azimuth | —                | exact                                          |
| overlap verticals (`grid`)             | centre azimuth | —                | `parallax_radius`                              |
| horizon, zone band (`grid`)            | —              | centre elevation | exact                                          |
| mark: line, foot tick, field (`marks`) | centre azimuth | centre elevation | x: `parallax_radius`, y: the person's distance |
| label (`labels`)                       | centre azimuth | lane by id       | `parallax_radius`                              |

**The rule: two things on the strip are comparable only if they share an axis and a depth.** A
misalignment on this display is a pair that shares the axis and not the depth — an overlap line off
where a mark's field actually switches, marks beside their own pixels, a foot tick off its own zone
edge. In the code the depth parameter of the shared functions is called `depth_radius`, after no one
caller's depth; `elevation_window` is the exception, because it is the strip's single y scale and
moves everything on it together.

Exact means no depth enters at all — a seam is a bearing, and a floor circle's depression is
`atan(camera_height / R)` whatever anything assumes — so those are the things a tape can check. Two
lines cross the boundary. The dead zone agrees with the picture at every depth, because the rule reads
the raw image column and the stitch places the picture through the same map. The foot tick is exact
against the zone band, because going through the person's own distance makes the lens height cancel:
`atan(tan(−atan(h/d))·d/R) = atan(−h/R)`, the zone band's own formula.

- **The overlap**, `rig.overlap` wide (31.0° on this rig), symmetric about each seam. Nothing tunable,
  and it marks one thing: where a mark's field changes width, from `seam.link_angle` inside to
  `reacquire_angle` outside. It is `angle_in_overlap`'s own threshold — a local angle of 28.3°,
  derived at the zone's far edge so the flag never under-reports (`TRACKING.md`, *Linking on a seam*)
  — projected at `rig.parallax_radius`, the depth the marks are drawn at. A local angle has no single
  position on the ring, and the same threshold drawn at the far edge instead would land 2.3° away,
  where nothing happens. It is not where the two pictures meet, which is 19.3° at R 2.1 and which the
  frames show for themselves: the flag is generous, so the yellow lines sit outside the visible
  overlap. It goes to zero below about R 1, where the sectors stop meeting (*What the horizontal field
  allows*), and the lines then vanish.
- **The zone band**, the tracked floor: a translucent band between `rig.zone_min_radius` and
  `rig.zone_max_radius`, at the depressions they subtend at the rig centre, `atan(camera_height / R)`
  — R 1.5 is −18.4° and R 3.5 is −8.1°. A floor circle of constant radius is a constant depression,
  so the zone is a band of rows, the same at every azimuth. Tape the two circles on the floor and they
  must land on the band's two edges. It is also what a person's mark is read against: a mark's line
  ends at the foot row, so someone inside the zone has that end inside the band. The band is clipped
  to the strip's window, and at the studio preset it runs off the bottom: the window bottom is −16.9°
  against R 1.5's −18.4° (the strip shows less than the frames do: `elevation_window` takes the band at
  its tightest column so no column fades to black). A fill that runs off the edge says *continues past
  here*, which a line pinned to the boundary row could not.
- **The dead zone**, `seam.dead_zone` in from each camera's own field edges, where that camera refuses
  to start a new person. It is a band in image space because the rule reads the raw local angle — the
  image column — and the stitch places the picture through the same map, so band and pixels agree at
  every depth, and it can be checked against the image anywhere. A person is born as long as one
  camera accepts them, so the region where nobody can be born is where two red bands overlap: at
  R 1.35 they do, on the seam; from about R 1.5 they no longer do.

The link rule is not a zone — it is a property of a pair of observations, not of a place — so it is
drawn as a field around each mark instead (*Reading the fields*).

`render.panorama.blend` sets how the overlap combines. `MAX` is the default and what the rest of this
procedure assumes; the others are second opinions on the same seam:

| mode          | read it for                                                       |
|---------------|-------------------------------------------------------------------|
| `MAX` / `MIN` | the plain picture; a ghost as a doubled bright (or dark) edge     |
| `AVERAGE`     | a ghost as a soft double image                                    |
| `DIFFERENCE`  | tune for black, the most sensitive; read the edges, not the level |
| `SPLIT`       | which way it is wrong — one camera red, the other green           |
| `STRIPE`      | alternating columns, so a straight edge zigzags                   |

`MIN` shows the dark double edge on bright content. Each camera runs its own auto-exposure, so in
`DIFFERENCE` a brightness mismatch lifts the whole band off black. In `SPLIT` the leading fringe is the
left camera. `STRIPE` is blind to exposure differences, so it is the mode for two cameras that disagree
on brightness.

**A free check of the whole azimuth chain:** at R 2.1 each camera spans 109.3° of the strip, not 127°.
Set `track.rig.camera_radius` to 0 and every image snaps to exactly 127° with 37° overlaps. It is a
live slider, so this exercises the entire geometry in two drags.

### Reading the fields

Each observation's line sits inside a translucent field of its own colour, the same height as the
line, as wide as the rule that decides what that observation may be joined to: `seam.link_angle`
where a second camera also sees it, `reacquire_angle` where none does. So the width says which rule
owns that part of the ring, and walking one person from mid-field to a seam visibly widens their field
as the second camera picks them up. The rules themselves are in `TRACKING.md`, *Identity* and
*Linking on a seam*.

**Read it as a pair test.** The field is the rule's angle wide rather than that much either side of
the line: both gates have the form `|Δ| ≤ angle`, so two fields each `angle` wide touch at exactly the
difference the gate allows. Two fields of one colour that overlap are two observations the tracker
will join; two colours that overlap are two people it might confuse. Fields of ±`angle` would overlap
out to twice the gate and claim links that never happen — at `link_angle` 15 they would show a link
for two views 30° apart. `modules/render/tests/test_marks.py` asserts that drawn overlap and
`Seams._observations_match` agree case for case.

Two things the field does not say:

- Both rules apply inside an overlap — a re-acquisition is tried first, everywhere — but the
  cross-camera one is the one tuned there, so it is the one drawn there.
- The re-acquire rule is in a camera's local angle, and the strip is in azimuth, so that field converts
  through the person's own distance and measures less than `reacquire_angle` against the degree grid:
  `d / (d + r)` of it on axis, so 5° reads as 4.5° at 3 m and 4.7° at 6 m. The same conversion is
  applied to the positions, which keeps the pair test valid once drawn.

The footer prints all of it — `dead 6.5°  link 15.0°/0.20  reacquire 7.0°` on the studio preset — so a
width on screen can be checked against the number that produced it.

**Setting the seam values.** One person crossing a seam keeps two overlapping fields of one colour with
their arms down; two people a metre apart do not overlap at all. What `link_angle` has to cover —
the body and the parallax residual — is tabled in `TRACKING.md`, *Linking on a seam*.

### The horizon check

The green line is elevation 0: the level plane at lens height, 0.50 m. Anything at that height lands
on it at any distance — the parallax correction scales elevations, and zero stays zero — so it is the
one row in the panorama that is exact everywhere, not only at the focus radius. It is also the zero the
tracker measures distance from (`TRACKING.md`, *The tracker's distance*).

Tape at 50 cm on the wall in front of each camera and read it against the line:

| the tape                                        | what is wrong                                       |
|-------------------------------------------------|-----------------------------------------------------|
| on the line, all the way round                  | nothing                                             |
| below the line in every camera                  | the cameras aim higher than `tilt` — raise it       |
| above the line in every camera                  | the cameras aim lower than `tilt` — lower it        |
| slants across one camera's image                | that camera is rolled                               |
| on the line in one camera, off in its neighbour | those two disagree; that seam ghosts vertically     |

Without tape: a standing person's knees are at about lens height. Standing still at a few distances
along one camera's axis, a gap that stays constant is the levelling error; a gap that grows as they
come closer is only their knee not being at 50 cm (5 cm is ≈3° at 1 m, ≈0.6° at 5 m). Judge standing
still — a stride moves the knee.

### The mount readout

The warp models `tilt` only — it assumes the camera is not rolled about its optical axis. A rolled
camera tilts the horizon, which looks exactly like a wrong `tilt` in the panorama; the image cannot
tell them apart. The cameras can:

- **`camera.cam_N.readings.tilt_measured` / `roll_measured`** — from each board's IMU.
- **`camera.mount.status`** — pinned, always on screen: the average deviation from the preset, and a
  warning naming the worst camera when any exceeds `camera.mount.tolerance` (2°). If a board has no
  IMU it says *not measured*; then check roll by eye against a vertical edge.
- **`camera.cam_N.readings.roll_offset`** — what that camera reads when level. Subtracted from the raw
  reading, so the roll shown is how far the camera has moved since it was levelled. Per camera and
  never shared: it absorbs each unit's own sensor error, and those differ (cam_2 reads −2.25° sitting
  level; see *Site facts*). Tilt has no offset — there is no reference to calibrate it against on site.
- **`camera.cam_N.readings.fov_factory`** — the field this unit's own calibration spans across the
  frame (127.5–129.8° on this rig). A check that the sensor variant is the wide one (not ~97°), never
  alarmed on.
- **`camera.cam_N.readings.lens_error`** — the largest bearing error this unit has under the shared
  lens (see *The lens*). A property of the build, so also never alarmed on; it reads 1.4° on F124 and
  under 0.6° on the others, and a different number means a unit was swapped.

Roll matters more than it looks: it doubles at the seams. Neighbouring cameras see a seam on opposite
sides of their own centres, so the same roll moves the shared content in opposite vertical directions
— ≈2.2° of vertical mismatch per 1.2° of roll. A roll common to all four does not cancel. The offset
corrects the reading, not the image.

### Calibrating the metres

Two stages that fix different things and do not interact: stage 1 removes the drift, stage 2 the
scale. Do them in order, with `image` + `grid` + `marks` + `labels` on the panorama. What the tracker
measures and why `foot_offset` is the correction is in `TRACKING.md`, *Distance and height*.

**1. Walk one person out and turn `track.foot_offset` until `H` stops drifting.** `H` is
distance-invariant by construction, so any drift is the detector's pad and nothing else. On the studio
frame, with the detector's pad at 0.03 of the frame height, a 1.8 m person prints:

| camera distance    | 1.5 m | 2.5 m | 3.8 m |
|--------------------|-------|-------|-------|
| `H`, offset 0      | 1.63  | 1.54  | 1.44  |
| `H`, offset 0.03   | 1.80  | 1.80  | 1.80  |

It is self-signing: falling as they walk away means increase, rising means decrease. The model check is
built in — if `H` is already flat but wrong, the bias is proportional rather than a fixed pad, and this
setting is the wrong shape for it: it would have to be a fraction of box height instead (the signature
table in `TRACKING.md`, *The box bottom is not the feet*). Read only the flatness at this stage. Too much offset makes people inside the zone read past
the far edge, so tune by flatness and never overshoot.

**2. Then `H`'s absolute value checks `rig.camera_height`.** Once flat, `H` is the person's real reach
in metres and nothing else can be moving it. If a known 1.80 m person reads 1.65, the lens height is
out by the same factor — remeasure it rather than tuning `H` back.

**Confirmation with tape.** Tape the R 1.5 and R 3.5 circles and stand on each: the mark's foot tick
must land on that edge of the yellow zone band, to the pixel (*Two axes, and the depth that varies*).
`R` on the label reads the taped distance at the same time — the same number as `rig.zone_max_radius`
and the footer's `zone`, with nothing to convert — and the azimuth does not move through any of it.

Judge the feet on the panorama, not on the camera row. `BBoxRenderer` there reads stage frames, whose
`BBox` the crop extractor has overwritten with the crop ROI — zoomed 1.1× and aspect-filled to 3:4 —
so its bottom sits 10–27 px below the tracker's real ROI bottom, by a distance-dependent amount. The
crop box is the right thing to draw for a crop, and the pose skeleton draws the ankles where pixel
truth about the feet is wanted.

---

## Motor and sensor

The sensor pulses once per revolution when the reflective line on the head passes it; the firmware
forwards it as `/WS/sensor/fall` (only while commanded below 200 rpm) and restarts its ring counter on
it. `MotorController` measures phase and rpm from consecutive pulses (`light/motor.py`); the phase is
raw, 0 = the pulse, offset-agnostic by design. Above 200 rpm the sensor is silent: the show anchors the
spin-up on that silence (`is_projecting`) and the spin-down on the playhead lock (`is_locked`) — see `STATES.md`.
Where the sensor or the line sit is not a calibration input; the playhead offset absorbs it.

---

## Beam mode

**Role:** the playhead is the content clock in both modes; in beam mode it is also the bar's heading
as an azimuth, because the beams are where the bar points.

**What sets it:** the playhead offset, `light.playhead.pulse_offset`, degrees. The playhead NCO tracks
the measured motor phase while locked and adds it (`light/playhead.py`). (The per-pose feature
`PlayheadOffset` is a different thing in a different namespace.)

**What depends on it:**
- `PlayheadOffset = azimuth − playhead` per pose (`pose/playhead_offset.py`): the flash layers fire on
  it, and the sound receives it (`/pose/N/playhead/offset`).
- The hit that starts INTRO is `PlayheadOffset` changing sign (`statemachine/machine.py`,
  `_detect_hit`). A wrong offset fires the intro early or late.
- The bar simulation on screen draws the four lamps at this heading.
- Max receives it as `/global/playhead`.

Calibrating it aligns all three consumers at once. The flash window is 11.5° wide in the preset.
Re-tune after changing `light.motor.beam_rpm`: the loop delay inside the offset scales with speed.

Across a spin-up and a spin-down the playhead is never reset; only its rate source changes
(`light/playhead.py`):

- *Spin-up:* it stops tracking the bar and free-runs at `beam_rpm` from wherever it was. The beam at
  azimuth θ becomes the `projection_playhead` marker at θ, continuing at the same rate. (What the wall
  shows during the acceleration is covered in `STATES.md`.)
- *Spin-down:* the bar lands at an angle unrelated to the content clock. The playhead keeps
  free-running until the sensor settles near `beam_rpm` (a two-stage re-lock gate), then `tracking`
  eases it onto the bar over roughly 1 / `tracking` ticks. Up to half a turn of re-alignment is
  physics, not calibration; the show hides it by exiting S8/S9 only once the lock is in.

---

## Projection mode

**Role:** the ring — four strips painting one 3600-pixel image around the room.

**What sets it:**
- The projection offset, `inout.osc_light_sender.projection_offset`, degrees — rotates the whole ring
  as it goes on the wire (`inout/osc_light_sender.py`), so the frame on the board stays azimuth-true.
  It absorbs the reflective line's position and the firmware's quarter turn.
- The interlace, `inout.osc_light_sender.interlace` (see *Interlacing*), ±10 px. The firmware boots
  with its own values (`cor2` 3, `cor1` 1); ours replace them on connect and once a second, so the
  firmware side is never where to tune.
- `light.brightness` and the sender's `curve` / `lower_edge` / `upper_edge` — brightness, not
  position.

**The four lamps**, relative to the front white in the bar's direction: back white +180°, `blue[0]`
−90°, `blue[R/2]` +90° — what `BEAM_LIGHT_HEADINGS` (`light/frame.py`) encodes. Standing at the
fixture facing along the front beam, `blue[0]` is on the right; the code and the fixture's labels call
it "left" (site fact), reading from the wall looking at the fixture. Which physical strip is wired to
`blue[0]` is a wiring fact to confirm (*Site facts*).

---

## Layers

**Beam layers** write the four beam lights by name — `front_white`, `back_white`, `left_blue`,
`right_blue` on `Frame.beam_lights` — and nothing else (`layers/_base_layer.py`, `BeamLayer`). No
calibration of their own: the lamp shines where the bar points, and where that is as an azimuth is the
playhead. The sender copies them into the pixels the firmware reads in beam mode (pixel 0 and 1800 of
each channel, `FIRMWARE_LIGHT_SLOT_TURNS`), which the projection offset and interlace cannot reach. A
beam layer that reacts to people (`beam_flash`, `beam_haunted`) depends on `PlayheadOffset`.

**Projection layers** draw the ring at azimuth strip positions — `pose_instrument` at each person's
`Azimuth`, `projection_playhead` at the playhead (`ProjectionLayer`). No calibration of their own: they
author in azimuth, and the sender applies the projection offset and interlace on the way out.

---

## Sound (Max)

Max spatialises over the four speakers (site fact). It receives, all azimuths produced here:
`/global/playhead`, `/pose/N/azimuth`, `/pose/N/playhead/offset`, plus the state
(`inout/osc_sound_sender.py`, `modules/inout/osc_sound.py`). With the speakers placed by the layout
there is nothing to tune on the Max side.

Two settings of our own, in `inout.osc_sound_sender`, sent in every bundle:

- **`speaker_offset`** — where speaker 0 stands, as an azimuth (`/global/speaker/offset`, radians on the
  wire). The correction stays on our side so every azimuth Max receives stays true; Max adds this one
  constant in its panner. Placed by the layout it is 0.
- **`volume`** — main volume, 0–1 (`/global/volume`).

Neither is show state, so neither is zeroed on a blackout: a fader must read true whenever it is turned,
and a calibration must not snap to 0 between shows.

**The return path:** `/WS/sound/level` (left, right) → `beam_blue_sound` → the left and right blue
lamps (named after the fixture's blue-left / blue-right; nothing to do with stereo). No alignment; the
lamps turn with the bar.

---

## Screen

Two simulations share the `ws_light` row, and the render draws whichever matches the fixture's mode
(`render/render.py`): the beam view in beam mode draws the four lamps at the playhead heading; the ring
view in projection mode shows the ring buffer as it leaves the compositor. Both are azimuth-true,
because the projection offset is applied in the sender and never touches the frame on the board.

**The check, no hardware needed:** in projection mode the ring view's playhead line must sit under the
beam view's front lamp and the tracker row's person, and a spin-up must not move it. The screen shows
the room's angles; only the wall shows what the fixture makes of them.

---

## Simulation

Nothing here applies to a simulated session, and no separate preset is needed: the offsets describe
the physical build, and a recording carries its own frame. The show compares a person's `Azimuth` from
the recording with the playhead from the simulated motor, both in one frame, so the flash, the hit, the
sound and both screen views are self-consistent.

- Existing recordings are 1280 × 720 raw clips, shot at `tilt = 0`. Set `resolution` to P720 in the
  playback preset and the whole chain derives correctly — the frame height, the window and the horizon
  row, the distance estimate and the panorama's geometry. Left at P800 the warp expects an 800-row clip
  and every frame-relative number is off (the simulator warns once).
- `camera.simulator.apply_warp` applies `tilt` to a raw clip. It assumes the clip was shot at exactly
  that tilt; capture-time tilt is not stored with clips, so old footage can carry a horizon error.

---

## Open

- **The zone's two radii are declared, not measured.** `rig.zone_min_radius` / `zone_max_radius`
  (R 1.5 – R 3.5) decide the overlap band, the parallax depth and the far edge, so they are
  load-bearing. Tape both circles on the floor and check them against the yellow zone band's edges;
  the near one is below what the strip can show at P720 / tilt 15, so the band runs off the bottom
  there and R 1.5 has to be judged in the camera frames instead. If the room's usable area is
  different, change these rather than anything derived from them.
- **The panorama counts the black arch as covered.** The stitch culls by the frame's window (the centre
  column's reach), not per column, so in the top corners of a camera's field the comparing blends
  (`AVERAGE`, `DIFFERENCE`, `SPLIT`, `STRIPE`) mix black into the count. `MAX`, the mode the procedure
  uses, is unaffected: black loses to the neighbour. Exact culling would take a per-column coverage
  texture per camera from `frame_coverage`, published by each `Camera`.
- **Roll is not modelled by the warp.** `warp_mesh_points` takes `tilt` only, so a camera that is rolled
  ghosts at its seams (≈2.2° vertical per 1.2° of roll). The mount readout says whether that is
  happening; the fix, if it is, is the tripod or a second rotation in the mesh. Tape at lens height on
  the far wall sits a few degrees off the horizon line (site fact), which is levelling to resolve.
- **A virtual camera per person for the pose** (maybe): the cylindrical crop is a level pinhole panned
  to the person; re-projecting the crop as a pinhole pitched at the person would be the most typical
  photograph the pose model could get. A per-crop warp in the crop extractor, no change to the shared
  frame. Worth an A/B on keypoint confidence at raised arms and close range.
- **A placement aid** (maybe): since placement is the room-side calibration, projection layers that put
  the sector boundaries and centres on the wall would make it easier. The IMU cannot help with azimuth
  — its magnetometer is useless next to the motor and the LED strips.

Tracking's open questions are in `TRACKING.md`, *Open*.

---

## Site facts

`R` is a radius from the fixture axis, because that is where everything here is built and taped from,
and it is what the settings, the panorama's footer and a mark's `R` all carry — one number, no halving.
The only `Ø` in this document is the `fixture:` line's own hardware: a ring or a tube is described by
its cross-section, as its supplier does, and converting those to radii would make the document harder
to check against the parts.

    reference:        the connection side of the cube                            (site fact)
    direction:        counter-clockwise seen from above                          (site fact)
    azimuth 0:        the centre of the connection side                          (rule — follows from the cameras)
    cameras:          on the corners, 15 cm beyond the cube, pointing diagonally;
                      camera 0 at the corner counter-clockwise of the connection
                      side, then counter-clockwise; camera R 0.36, height 0.50 m  (rule; taped)
    speakers:         parallel to the faces, 10 cm off, pointing out; speaker 0
                      on the connection side, then counter-clockwise             (rule)
    play zone:        R 1.35 m to R 3.5 m; hard floor R 1 m                        (site decision)
    tracked zone:     R 1.5 – R 3.5 = rig.zone_*; drives the overlap band, the
                      parallax depth and the far edge. Three notions in this doc:
                      the play zone (R 1.35 – R 3.5), the hard floor (R 1) and
                      this, the calibrated span the code acts on               (site decision)
    panorama focus:   R 2.1 = rig.parallax_radius, the zone's harmonic mean      (derived)
    seam overlap:     the flag is 28.3° of local angle, taken at the zone's far
                      edge; drawn as 31.0° of azimuth at the parallax depth,
                      where a mark's field switches. The two cameras' pictures
                      share 19.3° at R 2.1, 9.4° at R 1.35, none by R 1          (derived)
    foot_offset:      0.03 of frame height, ≈29 px on 960 rows                   (walked out on the rig)
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
