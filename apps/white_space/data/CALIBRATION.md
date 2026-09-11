# White Space — Calibration

How the cameras, the machine in beam mode, the machine in projection mode and the sound come to
agree on *where* something is. Companion to `STATES.md` (the choreography) and `LAYERS.md`
(the layers). Everything here is code-verified unless marked **(site fact)** — told by the
operator — or **(deduction)** — follows from the facts but was not checked on the hardware.
All operator-facing angles in this document are **degrees, 0–360, counter-clockwise seen
from above**; the code keeps radians inside and on the wire. **The fixture firmware is not
changed**: everything here is modelled and corrected on the app side of the wire.

---

## Theory

### One frame: azimuth

Every position in the system is an **azimuth**, an angle around the room. There is exactly
one azimuth frame and **the machine defines it: azimuth 0 is the centre of the connection
side** of the cube (see *Fixed layout*). The fixture's two offsets are stated against that
zero — the pulse offset is where the front lamp points, from the connection side, at the
sensor pulse; the projection offset puts the ring's pixel 0 on the connection side — so both
are properties of the build, tuned once and carried in the preset.

In the code the frame is produced by the cameras:

    azimuth = target_fov · cam_id + local_angle − fov_overlap        (modules/tracker/panoramic/geometry.py)

so azimuth 0 is the start of camera 0's own sector, and there is deliberately **no camera
offset**: the cameras are *placed* so that camera 0's sector starts at the connection side
(the layout rule). Positioning the cameras with care is the whole of the room-side
calibration, and it has to be done anyway; a knob for it would only invite skipping that.
If the flash is ever off in a room, the fix is to move a camera, not to turn an offset.

A person's bearing leaves the tracker as `Azimuth`; the playhead is an azimuth; every layer
draws at an azimuth's strip position (`angle_to_strip_position`: azimuth / 360 × 3600
pixels); every angle Max receives is an azimuth.

### Direction

The bar turns **counter-clockwise seen from above** (site fact). The firmware's ring counter
advances with the bar and a strip index *is* that counter, so azimuth increases with the bar.
On the camera side azimuth increases with the image column, and every camera has `flip_h`
set (an `INIT` field), so the flip is part of the contract: with it, the columns must run
counter-clockwise too, and the cameras are numbered counter-clockwise (deduction — the
installation works, and an offset can shift a mirrored frame but never un-mirror it). A
mirror is invisible with one person: one crossing per turn can always be phased in. The
check for it needs two people, or one person walking along the sweep.

### Two modes, two offsets

The fixture puts light at an azimuth by two mechanisms, and both restart at the same
reference: the **sensor pulse**, once per revolution, when a reflective line on the head
passes the sensor.

- **Beam mode** (the fixture at low speed — commanded below `FIXTURE_PROJECTION_RPM` = 200 rpm; the
  firmware's one flag, `SLOW`, is true): you see the four lamps as beams sweeping the room.
  The pulse says where the bar is, the **playhead** — the content clock, alive in both modes
  — tracks it, and one number turns "angle since the pulse" into the front lamp's azimuth:
  the **playhead offset**. It is observed in beam mode — the flash lands on a person — and it
  contains whatever delay the loop has at that speed (pulse in, frame out), which is fine,
  because it is tuned at the speed it runs at.
- **Projection mode** (the fixture at high speed — commanded at or above 200 rpm, `SLOW`
  false): the bar is a blur and the image is painted from the firmware's own counter,
  restarted at the pulse, with a fixed quarter turn built in (`TEST = 900` px,
  `firmware.cpp:18`, applied at `:277-280`). One number rotates the authored ring into the
  counter's frame: the **projection offset**. It is observed in projection mode on static
  content — a line drawn at a person's azimuth lands on the person.

The playhead is the reference in both modes: in beam mode the front lamp beams it one to
one, in projection mode the same playhead is drawn as a line over the other content
(`projection_playhead`). It needs no offset of its own in projection mode, and the reason
is worth knowing. Tuning the pulse offset makes the flash *land* on the person, so the
internal playhead leads the visible beam by exactly the loop's output delay. The playhead is
never reset at spin-up, so it keeps that lead; the projected line is drawn at the internal
playhead and reaches the wall one output delay later — exactly where the beam would have
been. The hit and the sound fire on the internal playhead in both modes, so their timing
against the visible light is the same too. That is also why the projection offset must be
tuned on static content and **not** on the moving playhead line: judged at the moment of the
crossing, the line is one output delay behind, and tuning it onto the person would rotate the
whole ring by that delay. (The firmware applies a frame on the next fast revolution, so the
line may lag up to 30 ms more than the beam did — a degree or two, inside the flash window;
deduction.) A spin-up is the check: the projected line must continue where the beam was.

The two are independent and each is tuned where it is visible; there is no order between
them. They sit at opposite ends of the pipeline for a reason: the playhead offset corrects a
measurement **coming in** (the flash, the hit, the state machine and the sound all consume
the result, so it is applied at the source); the projection offset corrects an image **going
out** (nothing reads the rotated value back).

**The relation between them** — a readout, not a step. Write θ for the front lamp's azimuth at
the pulse and *d* for the loop's output delay. In projection mode the firmware paints from its
own counter, so no app delay can move the image and the alignment is pure geometry:

    projection offset = 90° − θ            (the 90° is the firmware's TEST = 900 px of 3600)

In beam mode the flash is computed at the internal playhead and lands one delay later, so
tuning it to hit the person builds that delay into the number:

    playhead offset = θ + d·rpm·6          (degrees; rpm·6 = degrees per second)

Adding them cancels θ, which is why the sum is a check on both at once:

    playhead offset + projection offset = 90° + d·rpm·6

Today: 262.8° + 198.0° = 100.8°, so **θ = 252°** and the residual 10.8° is the delay — 50 ms
at 36 rpm. Both were tuned on a slider that stepped 3.6°, so read that as 42–58 ms, which is
about what a fall message, a 30 Hz tick and a frame on the wire should cost. Two numbers found
by eye, years apart from this model, agreeing to one slider click.

The fixture switches between the two mechanisms on the **commanded** rpm the moment it
receives it (`firmware.cpp:475`), regardless of how fast the bar is actually turning.

### Interlacing: the two sides of the bar paint one image

In projection mode each channel is painted twice per revolution: by the lamp on one end of the bar
and, half a turn later, by the lamp on the other end reading the pixel 1800 further on
(`firmware.cpp:277-280`). The two lamps are not copies of each other: their LEDs have gaps,
and the two strips are mounted out of phase so that one arm's LEDs sit in the gaps of the
other's (site fact) — the two halves of a revolution **interlace** into one image, the way
the two fields of a television frame do. That only works if the two lamps are exactly 180°
apart and the blue pair exactly a quarter turn from the white pair; a mounting error of a
degree shows as a doubled line on the wall instead of an interlaced one. The four
**interlace** values (`osc_light_sender.interlace`, one per lamp, sent as `/WS/o/0..3`,
firmware `cor0..3`) shift each lamp's readout by a few pixels (1 px = 0.1°) so the two
whites interlace, the two blues interlace, and the blue image sits on the white one.
Projection mode only: the firmware ignores them below 200 rpm (`firmware.cpp:297-305`).

### Sound inherits the frame

Every position Max is sent is already an azimuth. Max cannot disagree with the cameras
through anything in this repository; the only thing left is where speaker 0 stands relative
to azimuth 0, which the fixed layout settles by placement. Max inherits the *timing* of the
show from the playhead offset: the hit that starts INTRO is the beam crossing a person.

---

## Fixed layout: the connection side is the reference

The machine's base is a cube holding the motor and the electronics, with every connection on
one face — the **connection side** (site fact). It is the one physical reference the fixture
carries everywhere, so the layout is defined from it. Drawing: `White Space Layout Sheet.pdf`
in this folder (A the room, B the machine).

- **Azimuth 0 is the centre of the connection side**, azimuth increasing counter-clockwise.
- **Cameras on the corners**, pointing diagonally outward, 15 cm beyond the cube's corner
  (lens ≈ 36 cm from the axis). **Camera 0 at the corner counter-clockwise of the connection
  side**, then counter-clockwise. Camera *i* points at 90·*i* + 45; azimuth 0, the start of
  camera 0's sector, then falls on the connection side. Every seam is a face centre. The
  15 cm keeps the speakers out of frame: the nearest speaker corner is ≈ 76° off a camera's
  axis, outside its 63.5° half-field.
- **Speakers parallel to the faces, 10 cm off them**, pointing outward, not touching the
  fixture (site fact). **Speaker 0 on the connection side**, then counter-clockwise. Speaker 0
  stands on azimuth 0, so Max needs no constant.
- **Everything numbered from 0**, counter-clockwise from the connection side: `cam_0`…,
  `/pose/0`…, `white_0` / `blue_0`. (The firmware's comments count lamps from 1; internal.)

        face:   spk 0 (az 0) — the connection side · seam cam 3 | cam 0    corner: cam 0 (→ 45)
        face:   spk 1 (az 90)                     · seam cam 0 | cam 1    corner: cam 1 (→ 135)
        face:   spk 2 (az 180)                    · seam cam 1 | cam 2    corner: cam 2 (→ 225)
        face:   spk 3 (az 270)                    · seam cam 2 | cam 3    corner: cam 3 (→ 315)

With four 127° cameras each sector is 90° and the overlap `fov_overlap` = 18.5° on each side
of every seam. What the layout buys: the two offsets stop being per-venue tunings. The
playhead offset encodes where the reflective line sits on the head relative to the front
lamp (plus the loop delay); the projection offset encodes the same plus the firmware's quarter
turn. Both are properties of the *build*. Assemble by the rule above — the cameras placed
so that camera 0's sector starts at the connection side, speaker 0 on it — and **everything
is calibrated by placement**; what remains are the checks at the end. Re-tune the offsets
only after the head, a strip or a lens has been remounted; in a room, correct placement, not
offsets.

**Camera height, tilt and the inner circle.** The tripods stay at their 50 cm minimum so the
cameras shade the light as little as possible (site fact; the light starts at 32 cm), lens
≈ 50 cm up and 36 cm out. Tilted up about **15°** with the full 800 rows, a 1.9 m person's
head is in frame from the **Ø 2.7 m** inner circle (site decision) and the feet from Ø 2.9 m;
set by eye, 12–18° all work. Near the machine the fields do
not meet: on each seam a person's centre is outside both cameras until 1.0 m out, so
**Ø 2.0 m** is the hard floor. Between Ø 2.0 m and Ø 3.5 m a seam person is cut on one side in
each camera and the box centre shifts toward the visible side, ≈ 3° at Ø 2.7 m — a wobble at
the handover, not a failure.

---

## Camera

**Role**: defines the azimuth frame. Nothing aligns the cameras; everything aligns to them.

**The hardware** (Luxonis OAK-D Pro W): the app runs `color = false` and uses **the left mono
camera only** (`modules/oak/camera/pipeline.py`, `SetupMono`): OV9282 W, global shutter,
1280 × 800, lens **127° × 79.5°**. The 127° is the preset's `fov` — an `INIT` value, because it
feeds the warp mesh that is baked when the device opens, so like `tilt` it is set in the preset
and applied at relaunch — and it is the only field angle stored: the pipeline requests
`THE_800_P`, the full readout, and the vertical field is *derived* from `fov` and the frame's
shape (`parallax.vfov` is read-only, 79.4°). The lens maps
angle linearly to radius, which is what makes that derivation hold and what the published
spec confirms. The sensor reference table for every OAK variant in use, with the Luxonis
links, lives beside the resolution tables in `modules/oak/camera/definitions.py`; opening a
device logs the sensor behind each socket. The left lens sits 37.5 mm off the tripod thread:
put the *lens* on the corner line. The mono sensors run at fixed exposure
(`mono_auto_exposure = false`) with the 940 nm flood; they carry an IR filter and do not see
the light show (site fact). Keep the dot projector off.

**Tilt.** Each camera has a `tilt` in degrees, positive = aimed up. The warp re-aims the camera
so an image column reads as one azimuth, which is what the tracker assumes; it costs the frame
edges (≈ 20 % at 15°, because it asks for a view the sensor never imaged) and is baked into the
device when it opens, so it is tuned by editing the preset and relaunching. The simulator can
apply it to a recording shot at `tilt = 0` (`camera.simulator.apply_warp`). `keystone` on the
same camera is the other installations' full-frame correction and stays 0 here — the two are
exclusive.

**What sets it** (`camera.tracker`, `camera.fov`, `camera.tilt`): `fov` (each camera owns
`target_fov = 360 / num_cameras`, the excess is shared overlap); `parallax` (`ring_radius`, the
lens distance from the axis, **0.36 m** measured, and `camera_height`, the lens height above the
floor, **0.5 m** measured — both taped, neither tuned; `vfov` is derived, not set); `seam`
(tracklet handover in the overlap); and `tilt`, shared across all four cameras. There is no
distortion correction and there is no `person_height`: the projection is fixed in the warp and
the distance comes off the floor plane.

**How to calibrate**: turn on `render.panorama.enabled`. The per-camera row is replaced by the
four images unwrapped into one 360° strip — azimuth 0 at the left edge, the same scale as the
observation strip directly below it, with a degree grid and the sector seams and camera axes
picked out. The overlaps are where the two neighbouring cameras are drawn on top of each other,
at the azimuth each one claims, so the check that could not be run before is simply *look at the
overlap*:

| what you see | what is wrong |
|---|---|
| the overlap coincides | nothing — go on to the offsets |
| aligns at head height but not at knee height | `tilt` |
| a constant sideways offset across the whole overlap | `fov` |
| a residual that grows toward the frame edges | not the equidistant lens the spec describes; no knob, and it would be news |
| the image coincides but a person's two boxes below do not | the distance model — re-measure `ring_radius` and `camera_height`, do not tune them |
| both coincide and the primary still jumps at the seam | `seam` (`reject`, `reach`, `hysteresis`) |

The image is stitched for one assumed depth, `render.panorama.focus_diameter` — **Ø 4.5 m**, the
middle of the play zone. It is exact there and ghosts by a bounded amount elsewhere: **+3.9° at
Ø 3 and −2.5° at Ø 7**, the span the correction has to cover. So judge alignment with someone
standing near the middle of the room, and read a ghost at the wall or at the rig as expected.
Nothing about a person feeds the image — no box, no pose, no estimate — so nothing can fool it;
only the *boxes* carry the tracker's per-person distance.

`fov` and `tilt` are `INIT`: they are baked into the warp when the device opens, so they are
tuned by editing the preset and relaunching, not by dragging a slider. Cameras first — both
offsets and the sound are tuned against the result.

---

## Motor and sensor

The sensor pulses once per revolution when the reflective line on the head passes it; the
firmware forwards it as `/WS/sensor/fall` (only while commanded below 200 rpm) and restarts
its ring counter on it. `MotorController` measures phase and rpm from consecutive pulses
(`light/motor.py`); the phase is raw, 0 = the pulse, offset-agnostic by design. Above 200 rpm
the sensor is silent: the show anchors the spin-up on that silence (`ring_formed`) and the
spin-down on the re-lock (`synced`), see STATES.md. Where the sensor or the line sit is not
a calibration input — the playhead offset absorbs it.

---

## Beam mode (the fixture at low speed)

**Role**: the playhead is the content clock in both modes; in beam mode it is *also* the
bar's heading as an azimuth, because the beams are where the bar points.

**What sets it**: the **playhead offset**, `light.playhead.pulse_offset`, in degrees (the
per-pose feature `PlayheadOffset` is a different thing in a different namespace). The playhead
NCO tracks the measured motor phase while locked and adds it (`light/playhead.py`). Today:
**262.8°**.

**What depends on it** — the number with the widest reach:
- `PlayheadOffset = azimuth − playhead` per pose (`pose/playhead_offset.py`): the flash
  layers fire on it, the sound receives it (`/pose/N/playhead/offset`).
- The **hit** that starts INTRO is `PlayheadOffset` changing sign (`statemachine/machine.py`,
  `_detect_hit`). A wrong offset fires the intro early or late.
- The bar simulation on screen draws the four lamps at this heading.
- Max receives it as `/global/playhead`.

**How to calibrate**: beam mode (IDLE is fine). One person stands still. Adjust
the offset until the flash fires exactly as the beam sweeps over them — equivalently
`/pose/N/playhead/offset` reads 0 at the crossing. The flash, the hit and the sound all read
the same offset, so this one adjustment aligns all three. The flash window is 11.5° wide in
the preset and the slider steps 0.1°, one ring pixel. Re-tune after changing `beam_rpm`: the
loop delay inside the offset scales with the speed.

**Across a spin-up and a spin-down.** The playhead is never reset; only its *rate source*
changes (`light/playhead.py`):

- *Spin-up*: the playhead stops tracking the bar and free-runs at `beam_rpm` from wherever it
  was. Nothing is lost — in projection mode the bar's own position is meaningless, the image
  is painted from the counter. The beam at azimuth θ becomes the `projection_playhead` ring marker
  at θ, continuing at the same rate; the projection offset makes the ring azimuth-true.
  (What the wall shows *during* the acceleration is another matter — the fixture is already
  in projection mode while the bar is still slow, see STATES.md.)
- *Spin-down*: the bar decelerates unmeasured and lands at an angle unrelated to the content
  clock. The playhead keeps free-running at `beam_rpm` until the sensor's readings settle near
  `beam_rpm` (the two-stage re-lock gate), then `tracking` eases it onto the measured bar over
  roughly 1 / `tracking` ticks. Up to half a turn of re-alignment is physics, not calibration.
  The show hides it: S8/S9 keep the wall fading and exit only once the lock is in, so by the
  time the beam is *seen* as a beam it is the tracked one. The screen's bar simulation does
  the same.

---

## Projection mode (the fixture at high speed)

**Role**: the ring — four strips painting one 3600-pixel image around the room.

**What sets it**:
- The **projection offset**, `inout.osc_light_sender.projection_offset`, in degrees — rotates
  the whole ring as it goes on the wire (`inout/osc_light_sender.py`, next to the interlace),
  so the frame on the board stays azimuth-true. It absorbs the reflective line's position and
  the firmware's quarter turn; nobody needs to know the 900. Today: **198°**.
- The **interlace** values (`osc_light_sender.interlace`, see Theory). Preset: `white_1` 5,
  `blue_0` −10, `blue_1` 9, range ±10 px. The firmware boots with its own values (`cor2` 3, `cor1` 1);
  ours replace them on connect and once a second, so the firmware side is never where to tune.
- `light.brightness` and the sender's `curve` / `lower_edge` / `upper_edge` — brightness, not
  position.

**The four lamps**, from the firmware's sampling offsets, relative to the front white in the
bar's direction: back white +180°, `blue[0]` −90°, `blue[R/2]` +90°. This is what
`BEAM_LIGHT_HEADINGS` (`light/frame.py`) encodes for the beam lights too. Standing at the
fixture facing along the front beam, `blue[0]` is on the right; the code and the fixture's
labels call it "left" (site fact), which reads from the wall looking *at* the fixture. Which
physical strip is wired to `blue[0]` is a wiring fact — worth one look.

**How to calibrate**: projection mode with a layer that draws *static* content at a person's
azimuth — `pose_instrument` (its anchor line) or a test layer. One person stands still.
Adjust the projection offset until that line is on them. Not the moving playhead line: it
trails the internal playhead by the output delay (see Theory), and tuning on it would rotate
the whole ring by that delay. Then the interlace, with any thin line: adjust until it is
single on the wall, not doubled — the whites against each other, the blues against each
other, then the blue image onto the white. Finally a spin-up: the projected playhead line
must continue where the beam was.

---

## Layers

**Beam layers** write the four beam lights by name — `front_white`, `back_white`, `left_blue`,
`right_blue` on `Frame.beam_lights` — and nothing else (`layers/_base_layer.py`, `BeamLayer`).
No calibration of their own: the lamp shines where the bar points, and
where that is *as an azimuth* is the playhead. The sender copies the four values into the
pixels the firmware reads in beam mode (pixel 0 and 1800 of each channel,
`FIRMWARE_LIGHT_SLOT_TURNS`); the projection offset and the interlace provably cannot reach them.
What a beam layer *does* depend on is `PlayheadOffset` when it reacts to people
(`beam_flash`, `beam_haunted`).

**Projection layers** draw the ring at azimuth strip positions — `pose_instrument` at each
person's `Azimuth`, `projection_playhead` at the playhead (`ProjectionLayer`). No calibration
of their own either: they author in azimuth and the light sender applies the projection offset
and the interlace on the way out.

---

## Sound (Max)

Max spatialises over the four speakers (site fact). It receives, all azimuths produced here:
`/global/playhead`, `/pose/N/azimuth`, `/pose/N/distance`, `/pose/N/playhead/offset`, plus
the state (`inout/osc_sound_sender.py`, `modules/inout/osc_sound.py`). With speaker 0 on
azimuth 0 and the speakers numbered counter-clockwise there is nothing to tune on the Max
side. Check: in IDLE, have Max voice `/global/playhead`; the sound must follow the searchlight
around the room.

**Two settings of our own**, in `inout.osc_sound_sender`, sent in every bundle:

- **`speaker_offset`** — where speaker 0 stands, as an azimuth, degrees 0–360 in steps of 1
  (`/global/speaker/offset`, radians on the wire like every azimuth). A speaker stand is a
  fuzzy target, so the correction stays on *our* side: every azimuth Max receives stays true,
  Max adds this one constant in its panner, and the number travels with the preset. Placed by
  the fixed layout it is **0** and Max needs nothing.
- **`volume`** — the main sound volume, 0–1 (`/global/volume`).

Neither is show state, so neither is zeroed on a blackout the way the playhead and the motor
mode are: a fader must read true whenever it is turned, and a calibration must not snap to 0
between shows.

**The return path**: `/WS/sound/level` (left, right) → `beam_blue_sound` → the left and right
blue lamps (site fact: named after the fixture's blue-left / blue-right; nothing to do with
stereo). No alignment; the lamps turn with the bar.

---

## Screen (the render's light row)

Two simulations share the `ws_light` row and the render draws whichever matches the
fixture's mode (`render/render.py`): the **beam view** in beam mode draws the four lamps at
the playhead heading; the **ring view** in projection mode shows the ring buffer as it leaves
the compositor. Both are azimuth-true, because the projection offset is applied in the light
sender and never touches the frame on the board.

**The check** needs no hardware: in projection mode the ring view's playhead line must sit
under the beam view's front lamp and the tracker row's person, and a spin-up must not move it.
The screen shows the room's angles; only the wall shows what the fixture makes of them.

## Simulation

Nothing here applies to a simulated session, and no separate preset is needed. The offsets
describe the physical build; a recording carries its own frame. The show never compares a
physical angle with anything — it compares a person's `Azimuth` from the recording with the
playhead from the simulated motor, both in one frame — so the flash, the hit, the sound
offsets and both screen views are self-consistent. The simulator can apply `tilt` to a
recording shot at `tilt = 0` (`camera.simulator.apply_warp`), which is how a tilt value is
tried against footage. Existing recordings are 720 rows and are being retired: they run through
the 800 pipeline mechanically, but their geometry is not tuned.

---

## Procedure

1. **Cameras** — placed by the layout (camera 0's sector starts at the connection side); `fov`
   and `tilt` set in the preset as lens and mount constants; `ring_radius` the measured 0.36 m.
   Whether the cameras agree is not visible today; the stitched panorama (planned) is the
   check. First, always.
2. **Playhead offset** — beam mode, one person stands still, the beam is on them as the
   playhead crosses them (the flash).
3. **Projection offset** — projection mode, the same person, a static line drawn at their
   azimuth (`pose_instrument`) is on them — not the moving playhead line, which trails by the
   output delay; then the interlace until the line is single; then a spin-up, the playhead
   line continues where the beam was.
4. **Speakers** — placed by the layout; nothing to tune. Listen in IDLE.

Steps 2 and 3 in either order. **Re-check without re-tuning**: a spin-up — the projected
line continues where the beam was; the flash on the first person at IDLE → INTRO; the sound
on the beam.

## Where the settings live

No gathered calibration panel: camera calibration is involved enough to stay in the camera
tab, and each offset stays with the code that applies it — the pulse offset with the
playhead under `light`, the projection offset and the interlace with the sender under
`inout`, `speaker_offset` with the sound sender. The steps above, with their settings, tools
and one-line instructions:

| step | settings | tool / readout | instruction |
|---|---|---|---|
| 1 cameras | `fov`, `cam_N.tilt`, `distortion.*`, `parallax.*`, `seam.*` | placement; the stitched panorama (planned) | "Place by the layout; set `fov` and `tilt` in the preset; the stitch shows the rest." |
| 2 playhead | `playhead.pulse_offset` (with `tracking`, `speed_smoothing`) | beam mode, `beam_flash`; `/pose/N/playhead/offset` live | "One person stands still; turn until the beam is on them at the crossing." |
| 3 projection | `projection_offset`, interlace `white_0/1`, `blue_0/1` | projection mode, `pose_instrument` (static line at the person) | "Same person; turn until the projected line is on them; adjust the interlace until it is single. Then a spin-up: the playhead line continues where the beam was." |
| 4 speakers | `speaker_offset` | IDLE, Max voicing `/global/playhead` | "Speaker 0 on azimuth 0; the sound follows the beam." |

**Units**: degrees for every *angle* an operator reads or turns, 0.1° step (one ring pixel) —
both offsets and the playhead and motor-phase readouts. `speaker_offset` is degrees too, in
steps of 1: a speaker stand is not placed to a tenth of a degree.
Radians stay the internal and wire unit. The interlace is **not** an angle and stays in
pixels: the firmware shifts a pixel index, an integer, one LED step at a time, so pixels are
its true unit (1 px = 0.1° is a remark, not a conversion), and its ±10 px range stays.

## Open decisions

- **Done**: beam/projection naming throughout; both offsets in degrees with the ring rotation
  in the light sender; `speaker_offset` + `volume`; 800 rows; `fov` per app, `tilt` in degrees
  on the real lens model, `keystone` kept for the other installations, `vfov` derived; the
  stitched panorama and the observation strip — the camera check described under Camera.
- **Hardware verification** (next): the stitch has only been checked against the arithmetic, not
  against a wall. It needs footage shot at 800 rows — the old 1280 × 720 clips are vertically
  mis-scaled and can confirm the layout but not `tilt` or `fov`. Confirm the sign of `tilt`
  first: positive should *improve* an up-aimed camera. Then record the four `tilt` values, `fov`
  and `ring_radius` here as build constants beside the two light offsets.
- **A placement aid for the cameras** (later): since placement *is* the room-side
  calibration, it deserves a good way of doing it — probably projection layers that put
  the sector boundaries and centres on the wall so each camera can be aimed against them,
  and a readout that says where a person at the connection side lands. Out of scope for now.

## Site facts

    reference:    the connection side of the cube                              (site fact)
    direction:    counter-clockwise seen from above                            (site fact)
    azimuth 0:    the centre of the connection side                            (rule — follows from the cameras)
    cameras:      on the corners, 15 cm beyond the cube, pointing diagonally;
                  camera 0 at the corner counter-clockwise of the connection
                  side, then counter-clockwise; lens ≈ 50 cm up, ≈ 15° up      (rule)
    speakers:     parallel to the faces, 10 cm off, pointing out; speaker 0
                  on the connection side, then counter-clockwise               (rule)
    play zone:    Ø 2.7 m to Ø 7 m; hard floor Ø 2.0 m                          (site decision)
    room:         8 × 8 m, machine in the middle                                (site fact)
    fixture:      cube 25 × 25 × 28 cm; rings Ø 25 / Ø 20 × 3.5 cm; tube Ø 20 × 154 cm;
                  head 9 × 9 cm; light from ≈ 32 cm                            (site fact)
    speakers:     25 × 25 × 33 cm, centres 35 cm from the axis                  (site fact)
    camera:       OAK-D Pro W left mono, 127° × 79.5°; 10 × 3.5 × 3.5 cm body on a
                  50 cm tripod; IR filter, does not see the light               (spec / site fact)
    drawing:      "White Space Layout Sheet.pdf", two A3 pages, to scale
    blue[0]:      wired to the strip labelled "blue left" / "blue right"       (confirm)
    playhead offset for this build:   262.8°                                   (tuned — re-check with the flash)
    projection offset for this build: 198.0°                                   (tuned — re-check with the projected line)
    front lamp azimuth at the pulse:  θ = 252°                                 (derived from the two — see Theory)
    loop delay in beam mode:          ≈ 50 ms (42–58)                          (derived — the 10.8° residual at 36 rpm)
    interlace:    white_1 +5, blue_0 −10, blue_1 +9 px                          (tuned)
    LED strips:   the two arms' LEDs are mounted out of phase and interlace     (site fact)
