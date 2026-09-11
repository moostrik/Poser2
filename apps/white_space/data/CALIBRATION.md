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
   - Set `fov`, `resolution` and `tilt` in the preset (`tilt` from the table under *Camera*), then
     relaunch — all three are baked in when the devices open.
   - Read the pinned `camera.mount.status`. It must say *mount OK*.
   - Level each camera against a spirit level, read its roll, type it into
     `camera.cam_N.readings.roll_offset`.
   - Turn on `render.panorama.enabled`. With someone standing near the middle of the room, the
     overlaps must coincide at head *and* knee height. If not, see *Reading the panorama*.
   - Tape at 50 cm on the wall in front of each camera: it must sit on the green horizon line. See
     *The horizon check*.
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
| 1 cameras | `fov`, `resolution`, `tilt`, `camera.cam_N.readings.roll_offset`, `camera.tracker.parallax.*`, `camera.tracker.seam.*` | `camera.mount.status`; the panorama | mount OK; overlaps coincide at head and knee height |
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
**127° × 79.5°**. The lens maps angle linearly to radius (equidistant), which the published spec
confirms and the warp relies on. The sensor table for every OAK variant in use, with Luxonis links,
lives beside the resolution tables in `modules/oak/camera/definitions.py`; opening a device logs
the sensor behind each socket.

The mono sensors carry an IR filter and do not see the light show (site fact); they use the 940 nm
flood (`camera.ir_flood_light`). Keep the dot projector off. `camera.mono_auto_exposure` applies
to all four.

### The camera frame

The warp (`equirect_mesh_points`, `modules/oak/camera/definitions.py`) delivers a **levelled,
equirectangular** frame: a column is one azimuth at every height, a row is one elevation. That is
what the tracker assumes. Three constants at the preset root define it, all `INIT` — baked in when
the devices open, so changed by editing the preset and relaunching:

- **`fov`** — 127°, the lens's horizontal field, quoted for the full sensor width. Each camera owns
  `target_fov = 360 / num_cameras`; the excess is overlap shared with its neighbours.
- **`resolution`** — `P720` or `P800`. P720 is a pure vertical crop of the 800-row sensor, so the
  horizontal field and the whole column-to-azimuth mapping are identical; only the vertical field
  changes. `camera.tracker.parallax.vfov` is derived from the two (79.4° at P800, 71.4° at P720),
  never set.
- **`tilt`** — up-tilt in degrees, positive = aimed up, shared by all four. The warp re-aims the
  camera by it. The sensor never imaged the part of the levelled frame it was not pointed at, so
  the bottom `tilt / vfov` of the frame is empty (≈19 % at 15°).

`keystone` is the other installations' full-frame correction; it stays 0 here, and is exclusive
with `tilt`.

The tracker's other constants, under `camera.tracker`: `parallax.ring_radius` (0.36 m) and
`parallax.camera_height` (0.50 m) — both **measured with a tape, never tuned** — and `seam.*`
(handover in the overlap). There is no distortion correction and no person-height assumption: the
projection is fixed in the warp, and distance comes off the floor plane.

### Tilt — derived from the build

`tilt` is derived, not set by eye, from the lens height, the ring radius and the **Ø 2.7 m** inner
circle (site decision).

The reference person is **1.8 m**, with an overhead fingertip reach of **2.2 m** — raised arms are
content the pose reads. On the Ø 2.7 m circle, on a camera's axis, they stand 0.99 m from the lens,
which puts their fingertips at +59.8°, head at +52.7° and feet at −26.8°. A camera aimed up by
`tilt` sees `[tilt − vfov/2, tilt + vfov/2]`; fitting fingertips *and* feet at Ø 2.7 would need
120° of vertical field, against 79.4°. So the tilt is a trade, and **it is resolved in favour of
the top**: losing the feet degrades the distance estimate (it extrapolates the box bottom), losing
the arms loses a gesture outright.

**The rule:** the feet are in frame from **Ø 3.0 m** — the inner edge of the calibrated play zone
(Ø 3 – Ø 7) — and all remaining room goes to headroom.

| `resolution` | vfov | tilt | fingertips from | head from | feet from |
|---|---|---|---|---|---|
| **P800** | 79.4° | 13 | Ø 3.31 | Ø 2.70 | Ø 2.71 |
| | | **16** ← use | Ø 3.04 | Ø 2.49 | **Ø 3.00** |
| | | 18 | Ø 2.87 | Ø 2.36 | Ø 3.23 |
| | | 20 | Ø 2.71 | Ø 2.24 | Ø 3.51 |
| **P720** | 71.4° | **12** ← use | Ø 3.81 | Ø 3.08 | **Ø 3.00** |
| | | 14 | Ø 3.60 | Ø 2.92 | Ø 3.23 |
| | | 16 | Ø 3.40 | Ø 2.77 | Ø 3.51 |

P720 needs less tilt because its field is 8° narrower, so the feet run out of frame sooner — and
the feet are what the rule pins. P720 cannot reach the Ø 2.7 inner circle at any tilt: head
clipping inside about Ø 3.1 is expected there, not a fault. Beyond 16° at P800 each extra degree
costs more feet than it gains reach.

### What the horizontal field allows

The vertical field decides whether a person *fits*; the horizontal field — `fov`, identical at
P720 and P800 — decides whether they are inside any camera's sector at all. It binds at the
**seams**, where a person is 45° off both neighbouring axes. Because each camera sits 0.36 m out
from the centre, its 127° covers less of the room as measured from the centre, worst up close:

| Ø | one camera's azimuth span | overlap at each seam |
|---|---|---|
| 2.0 m | 89.4° | **−0.6°** — the sectors do not meet |
| 2.7 m | 99.4° | +9.4° |
| 3.0 m | 102.2° | +12.2° |
| 4.5 m | 110.5° | +20.5° — the panorama's focus depth |
| 7.0 m | 116.4° | +26.4° |

- **Ø 2.03 m** — a seam person's centre enters one camera. Below it they are in the gap and
  invisible, which is why **Ø 2.0 m is the hard floor**: a consequence of the ring radius, not a
  choice.
- **Ø 3.53 m** — a whole body (50 cm shoulders) fits inside one camera. Between the two, a seam
  person is cut on one side in each camera and the box centre leans toward the visible side, ≈3°
  at Ø 2.7 m — a wobble at the handover, not a failure.

On a camera's own axis the horizontal field never binds (a whole body fits from Ø 0.97 m), so these
are purely seam properties.

### Reading the panorama

`render.panorama.enabled` replaces the per-camera row with the four images unwrapped into one 360°
strip — azimuth 0 at the left edge, the same scale as the observation strip below it, with a degree
grid, the sector seams and camera axes marked, and a bright horizon line. In the overlaps the two
neighbouring cameras are drawn on top of each other at the azimuth each one claims.

| what you see | what is wrong |
|---|---|
| the overlap coincides | nothing |
| aligns at head height but not at knee height | the mount — `tilt` or roll; read `camera.mount.status` to tell which |
| a constant sideways offset across the whole overlap | `fov` |
| a residual growing toward the frame edges | the lens is not the equidistant one the spec describes — no knob |
| the image coincides but a person's two boxes below do not | the distance model — re-measure `ring_radius` and `camera_height`, do not tune them |
| both coincide but the primary still jumps at the seam | `camera.tracker.seam` (`reject`, `reach`, `hysteresis`) |

The image is stitched for one assumed depth, `render.panorama.focus_diameter` — **Ø 4.5 m**, the
middle of the play zone. It is exact there and ghosts by a bounded amount elsewhere (+3.9° at Ø 3,
−2.5° at Ø 7), so judge alignment with someone near the middle of the room. Nothing about a person
feeds the image, so nothing can fool it; only the *boxes* carry the tracker's per-person distance.

**A free check of the whole azimuth chain:** at Ø 4.5 m each camera should span **110.5°** of the
strip, not 127°. Set `camera.tracker.parallax.ring_radius` to 0 and every image should snap to
exactly 127° with 37° overlaps. It is a live slider, so this exercises the entire geometry in two
drags.

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
- **`camera.cam_N.readings.fov_factory`** — the field the unit declares in its factory calibration.
  A check that the sensor variant is the one `fov` assumes (127°, not 95°), never alarmed on: the
  factory model is a pinhole fit that cannot represent a 127° lens.

**Roll matters more than it looks: it doubles at the seams.** Neighbouring cameras see a seam on
opposite sides of their own centres, so the same roll moves the shared content in opposite vertical
directions — ≈**2.2° of vertical mismatch per 1.2° of roll**. A roll common to all four does not
cancel. The offset corrects the *reading*, not the image.

### The tracker's distance

Floor plane: `camera_height / tan(depression of the box bottom)`. Two limits:

- **It cannot see nearer than the picture reaches.** At the recommended tilt — 16° at P800 or 12° at
  P720 — the lowest row with picture is 23.7° below the horizon: 1.14 m from the lens, which is
  the Ø 3.0 m feet rule by construction. Anyone closer has their feet in the empty band and reads
  ≈1.14 m however close they stand. A limit, not a fault.
- **It is weak at range.** Each degree of horizon error moves the reading by ≈0.15 m at 2 m,
  ≈0.9 m at 5 m, ≈1.7 m at 7 m. A tripod within ±1° still leaves roughly ±1 m at the far wall.

It is meant for filtering and, later, for the distance sent to Max — both measured from the **rig
centre**, not the camera. It currently reads 5 m as 1.9 m; see *Open*.

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
  the playback preset and the whole chain derives correctly — `vfov`, the distance estimate and the
  panorama's geometry. Left at P800 every frame-relative number is off by 800/720 (the simulator
  warns once).
- `camera.simulator.apply_warp` applies `tilt` to a raw clip. It assumes the clip was shot at exactly
  that tilt; capture-time tilt is not stored with clips, so old footage can carry a horizon error.

---

## Open

- **Roll is not modelled by the warp.** `equirect_mesh_points` takes `tilt` only, so a camera that
  is genuinely rolled still ghosts at its seams (≈2.2° vertical per 1.2° of roll). The mount readout
  says whether that is happening; the fix, if it is, is the tripod or a second rotation in the mesh.
- **The tracker's distance reads short at range.** 5 m reads 1.9 m — the feet are placed 9° too low.
  Tape at lens height on the far wall sits only a few degrees off the panorama's horizon line, so
  levelling is part of it, not all. **Test on location**, live rig:
  1. Tape at 50 cm on the far wall, per camera — its offset from the horizon line is the levelling
     error.
  2. Floor marks at 1.5, 2, 3, 4, 5 m along one camera's axis; a person on each; read `dis`. With
     (1) subtracted, an error that grows with distance points at `camera_height` or `vfov`; one that
     shrinks with distance points at the box bottom not sitting on the feet.
  3. The same on a recording made there, to know whether clips can be trusted for this.
- **A placement aid** (maybe): since placement *is* the room-side calibration, projection layers
  that put the sector boundaries and centres on the wall would make it easier. The IMU cannot help
  with azimuth — its magnetometer is useless next to the motor and the LED strips.

---

## Site facts

    reference:        the connection side of the cube                            (site fact)
    direction:        counter-clockwise seen from above                          (site fact)
    azimuth 0:        the centre of the connection side                          (rule — follows from the cameras)
    cameras:          on the corners, 15 cm beyond the cube, pointing diagonally;
                      camera 0 at the corner counter-clockwise of the connection
                      side, then counter-clockwise; lens 0.36 m out, 0.50 m up    (rule; taped)
    speakers:         parallel to the faces, 10 cm off, pointing out; speaker 0
                      on the connection side, then counter-clockwise             (rule)
    play zone:        Ø 2.7 m to Ø 7 m; hard floor Ø 2.0 m                        (site decision)
    calibrated span:  Ø 3 – Ø 7; panorama focus Ø 4.5 m
    room:             8 × 8 m, machine in the middle                              (site fact)
    fixture:          cube 25 × 25 × 28 cm; rings Ø 25 / Ø 20 × 3.5 cm; tube Ø 20 × 154 cm;
                      head 9 × 9 cm; light from ≈ 32 cm                          (site fact)
    speakers:         25 × 25 × 33 cm, centres 35 cm from the axis                (site fact)
    camera:           OAK-D Pro W left mono, 127° × 79.5°; 10 × 3.5 × 3.5 cm body on a
                      50 cm tripod; IR filter, does not see the light             (spec / site fact)
    drawing:          "White Space Layout Sheet.pdf", two A3 pages, to scale
    blue[0]:          wired to the strip labelled "blue left" / "blue right"     (confirm)
    LED strips:       the two arms' LEDs are mounted out of phase and interlace   (site fact)

    playhead offset:    262.8°                                   (tuned — re-check with the flash)
    projection offset:  198.0°                                   (tuned — re-check with the projected line)
    front lamp at pulse: θ = 252°                                (derived from the two)
    loop delay, beam:   ≈ 50 ms (42–58) at 36 rpm                (derived — the 10.8° residual)
    interlace:          white_1 +5, blue_0 −10, blue_1 +9 px     (tuned)
    resolution / tilt:  target P800, tilt 16°                    (the table under Camera; next recording)
                        the preset's P720 is for playing back the old 720-row footage,
                        whose capture tilt is unknown
    tilt, measured:     reads as configured on all four          (IMU)
    roll offsets:       cam_0 −1.00, cam_1 −0.91,
                        cam_2 −2.25, cam_3 −0.70                 (against a level; cam_2 is a sensor error)
