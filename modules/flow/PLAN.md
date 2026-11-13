# The flow's plan

What is next for the flow, the one place that says it: the main lines in order, each line's items, the
details that belong to no line, and what is parked. How each stage works is in its own document
([FLOW.md](FLOW.md) lists them); an item links to the section it changes. **(deduction)** marks what is
inferred, **(site fact)** what the operator saw.

## Main lines

| #   | Line                                                                      | State                                                         | Documents                                                    |
|-----|---------------------------------------------------------------------------|---------------------------------------------------------------|--------------------------------------------------------------|
| 1   | [The 3D fluid's input](#1-the-3d-fluids-input)                            | built; the depth read to choose                               | [VIEW.md](VIEW.md), [BRIDGE_3D.md](BRIDGE_3D.md)             |
| 2   | [The 3D fluid's look at 4K](#2-the-3d-fluids-look-at-4k)                  | next                                                          | [FLUID_3D.md](FLUID_3D.md), [TIMING.md](TIMING.md)           |
| 3   | [One unit across the stages](#3-one-unit-across-the-stages)               | not started                                                   | [FLOW.md](FLOW.md)                                           |
| 4   | [The apps on deep_flow's input](#4-the-apps-on-deep_flows-input)          | planned: A callbacks, B white_space, C smile, D hd_trio       | [INPUT_THREADS.md](INPUT_THREADS.md)                         |
| 5   | [The 2D flow follows the 3D](#5-the-2d-flow-follows-the-3d)               | not started                                                   | [FLOW.md](FLOW.md)                                           |
| 6   | [A camera without depth](#6-a-camera-without-depth)                       | not started                                                   | [FLOW.md](FLOW.md), [VIEW.md](VIEW.md)                       |
| 7   | [Recording and -sim](#7-recording-and--sim)                               | built; checks open                                            | [RECORDING.md](../recording/RECORDING.md), [INPUT_THREADS.md](INPUT_THREADS.md) |
| 8   | [The camera package cleaned](#8-the-camera-package-cleaned)               | built; the apps' runs open                                    | [CAMERA.md](../camera/CAMERA.md)                             |

The 3D fluid's lines are weighed by [its goal](FLUID_3D.md#the-goal): sharp, translucent ink patterns, layers
distinct.

## 1. The 3D fluid's input

The input as good as it gets, before the 4K density's heavy side. Built: every capture reaches the motion
([INPUT_THREADS.md](INPUT_THREADS.md)), the mask's edge on the image's and the depth read across a surface
([VIEW.md](VIEW.md)), one density and its fine density ([FLUID_3D.md](FLUID_3D.md#the-density)), each output at
its readings' depth ([BRIDGE_3D.md](BRIDGE_3D.md#the-z-positions)).

- The depth read: `view.depth_read` holds three, `SURFACE`, `MASK` and `BILATERAL`
  ([VIEW.md](VIEW.md#the-depth-read)). One stays, chosen by eye on the edges recording, and the other two go
  with `bilateral_contrast`.
- The whole chain before the fluid, camera to bridge, timed as one at the preset: what each pass costs against
  what it gives the fluid.
- The dye's amount weighed by the image's brightness, and z's share in the dye's amount: discussed, not
  designed.

## 2. The 3D fluid's look at 4K

### The velocity for the fine density

The 4K amount is carried by the 288 × 512 velocity read straight between its cells, 7.5× coarser, whose
kinks show as the cells in a sharp density. A tricubic read where the amount is carried: 8 trilinear reads
for 1 at each of 199 M texels a step, twice that with MacCormack **(deduction)**; kept if the look gains
for its cost.

### Swirls at more than one size

The confinement keeps swirls alive at one size, `vorticity_radius`: fine wisps or big eddies, not both.
More than one radius, each with its own strength (multi-scale vorticity confinement), keeps fine and large
at once. Starting with two: the curl at both radii in the same pass (six more velocity reads a cell), the
force from both curls in one pass (a curl volume more), their sum added to the faces once. About a third
to a half more of the confinement's 0.57 ms at the preset, 0.2 to 0.3 ms **(deduction)**, measured as it
lands. The fine density's turbulence already adds swirls below the grid to the dye, so what a second
radius adds to the look is judged against it, and by whether the layers stay apart: the confinement
also carries one layer's motion into the next.

### The fade and the evaporation

Each field's `fade_time` (seconds to 1%, a share at a time) and `evaporation_time` (seconds a full value takes
to reach 0, a fixed amount at a time) do not yet give what is wanted **(site fact)**
([FLUID_3D.md](FLUID_3D.md#the-fade-and-the-evaporation)); what each should do for the look is the first
question.

### The temperature

Whether the temperature needs a field of its own. Buoyancy needs only a scalar the water is lighter or
heavier by: the density alone can be it (ink sinks), and the temperature adds currents apart from the dye
(warm clear water rising), the textbook smoke's soot and heat. Off in the preset (`buoyancy` 0), and nothing
is injected into it then.

Where it is kept, its advection moves to the density's: today it is carried mid-step, after the
confinement and before the projection, by a velocity not yet free of divergence; the textbook carries every
scalar by the same projected velocity (Fedkiw 2001, Bridson), and the buoyancy reads the temperature of the
step before.

Making the density sink needs no temperature: `temperature.sink` (widths/s² per unit of density, along
gravity) weighs dyed water wherever it is, with `buoyancy` 0, in the same pass, its direction `gravity_x`,
`_y`, `_z`. It sits in the temperature's panel though it is the density's weight: if the temperature goes,
`sink` and the gravity move to a panel of their own or the density's.

### Details

- The body force's acceleration: tested, tweaked and made useful.
- The velocity loses its energy faster than expected **(site fact)**, with the plain step on the velocity: the
  plain step's own loss (a swirl keeps 52% of its speed in 2 s, MacCormack 62%), the drag now over the body's
  whole depth, or another cause; MacCormack with the clamp on the velocity is the first thing to try.
- The particles: worked on, and how they combine with the fine density.
- The masking in the bridge: the mask gates the velocity in the motion (`use_mask`) and the density's color in
  the bridge; in the bridge alone it is one gate in one stage.
- The blobs' and the surface's p95: a sprite that grows to twice `radius` on a strong drop fills four times
  the pixels. The cut at 2 sigma and the lower cap ([TIMING.md](TIMING.md#the-particles-1)) are the fix, judged
  on the look.
- The run-to-run spread of the pose's load, 1 to 1.5 ms of the step between two runs of one configuration:
  the person detection's rate is not fixed.

## 3. One unit across the stages

Every length in the flow is in widths: the motion's velocity (widths/s of the camera's image), the bridge's
readings, the fluid's inputs (`spread`), its look (`opacity`, `fog`), the box's depth, the monitor's scale.
In portrait the width is the smaller side; in landscape it is the longer, and the same settings would
then mean less against what is seen. One unit across the stages, the smaller side, keeps a preset's look
and feel when the box turns: in portrait nothing changes. It reaches past the fluid (the motion, the
bridge, the monitor), so it is one rule for all of them, made where each length is made.

## 4. The apps on deep_flow's input

Four parts, in this order: A deep_flow's callbacks become hand-overs; B white_space comes back on that input; C smile
is checked; D hd_trio comes back after the 2D fluid ([line 5](#5-the-2d-flow-follows-the-3d)). The rules they keep:
a callback stores or enqueues what it receives and returns, a receiver does its work on its own thread, the camera's
thread runs only the camera, and every drop is counted on the panel ([INPUT_THREADS.md](INPUT_THREADS.md)).

### A. deep_flow's callbacks as hand-overs

Four places run a receiver's work on another's thread (read on `deep-flow` at 5c74ad1b):

| Thread                        | Work it runs that is not its own                                                                         |
|-------------------------------|----------------------------------------------------------------------------------------------------------|
| the camera's                  | `RecordingSync` matching under its locks, from `Recorder.submit_frame` and `submit_tracklets`            |
| the uploader's                | `CaptureSync.submit_image` and `submit_depth`: the matching and the capture's emit                       |
| the uploader's                | `_on_image`: `PosesFromTracklets.process`, the RAW stage's box steps, `PoseInference.submit`, `expect_poses` |
| the pose runner's callback    | the RAW emit, CLEAN, SMOOTH, `CaptureSync.submit_poses`, the crossing into LERP (-sim: `RecordedPoses`')  |

Five steps, each one commit, checked by its tests and, on your go, by one run:

1. `RecordingSync` gets its own thread: its submits append to one inbox and return; its thread matches and hands
   each capture to the recorder. The inbox is not bounded: a recording needs every part and the matching is cheap;
   `HELD` after it stays.
2. `CaptureSync` matches on its own thread (it has one, for the deadlines): its submits append to an inbox, its
   thread drains it, waits for the next deadline in the same wait and emits; `_emit_lock` goes.
3. `PosesFromTracklets` gets its own thread: `submit(timestamp)` enqueues the sync (2 waiting, the oldest dropped
   and counted); its thread runs `process`, so the RAW stage's box steps, the crops and the model submit run there.
   `_on_image` keeps the submit and the camera interval; `expect_poses` comes from the new thread, by a callback
   with the sync's time and the poses asked per camera (one camera per sync in deep_flow, several in white_space).
4. The pose pipeline runs the steps after an async step on a thread of its own: the async step's results are
   enqueued (4 waiting, the oldest dropped and counted, `PosePipeline.dropped`); its thread runs the RAW emit,
   CLEAN, SMOOTH and the crossing. `RecordedPoses`' thread then only enqueues.
5. The six runners (pose, optical flow, segmentation; ONNX and TRT) lose their callback thread, which only came
   because the pipeline ran on it: they call their callbacks from the inference thread, which now only enqueue. A
   batch replaced before it ran is counted (`dropped` on the model's settings) instead of lost silently when the
   runner's `Queue(maxsize=2)` is full. The optical flow predictor, which calls its callbacks under its lock, copies
   the list under it and calls outside it. After step 4, not before.

Latency **(deduction)**: a hand-over is a notify and a wake, tens of µs on an idle core, up to the GIL's 5 ms
switch when the handing thread keeps running Python. A capture with its poses gets two more (steps 2 and 3, the
model's to SMOOTH +1 −1); against that the camera's and the uploader's threads no longer match or run the pose
step. Each step's run compares `CAPTURE_LATENCY`, `POSE_LATENCY_RAW`, `POSE_LATENCY_SMOOTH` and the drops, back
to back, on a new recording or live.

Decision: `CameraSync` (the frames' and the tracklets' sync, used by white_space and hd_trio) decides a sync in its
submit, on the caller's thread, and calls its callbacks under its lock. Recommended: it gets its own thread, as the
other syncs in steps 1 and 2, its submits only enqueue, and it calls its callbacks outside its lock; one more
hand-over before white_space's pose step. The other way: it stays inline as a cheap decision over a few dozen
timestamps, with only the lock fixed, and the rule then reads that a sync's decision is part of its store.

### B. white_space back

white_space (four mono OAK-D cameras at 23 fps, the panoramic tracker, the light) runs on `.venv` (depthai 2, the
`oak` backend), no depth; on this branch it stops at start. It moves onto A's result:

1. Each camera hands its frames and tracklets to the uploader (`Uploader(settings)`, `submit_frame`,
   `submit_tracklets`) and the recorder only.
2. The uploader's images go to the -sim detector, the board's `set_camera_image` (the previews) and the frames'
   `CameraSync`, whose sync is one uploaded image per camera, fired on the uploader's thread once every camera's
   image is on the GPU; its callback is `PosesFromTracklets.submit` (A3), which crops every camera's newest image
   (`PoseInference(crop, pose, uploader)`, its crops at the model's size already); the crops go to the board.
3. The uploader's tracklets go to the tracker (`submit_cam_tracklets`), the board, the tracker journal and the
   tracklets' `CameraSync`, each a store or an enqueue (the journal's `submit_cam_tracklets` checked for it); the
   sync's callback is `tracker.update`, which sets the tracker's event.
4. The recorder as [RECORDING.md](../recording/RECORDING.md#open) plans it for every app, and the oak backend's
   capture time stamped as the oak4's is (RECORDING.md Open 2): white_space's old show recordings keep playing,
   new ones are the new layout.
5. The light's calibration layer (`test_calibration.py`) reads the uploaded image instead of the CPU frame the
   camera's thread puts on the board (`_store_video_frame` and the board's video images go): it reduces its band of
   rows to one row on the GPU, queues the row's copy into pinned memory with an event, and uses on a later tick the
   newest row whose event has passed, so nothing waits.
6. The clip camera (`oak.ClipCamera`, a recording played through an OAK) hands over as the camera does.
7. The drops on the panel as deep_flow's: the uploader's, the pose step's, the pipeline's, the runners'.
8. The probe (`scripts/cadence_probe.py`) takes the app as an argument: the camera's thread per frame, the syncs
   per second against the cameras' rate, the pose rate per camera, the pose latency (RAW, SMOOTH), the drops.

Done when white_space starts on this branch, its camera threads only hand over (under 1 ms per frame at p95),
every sync runs the pose step, the pose latency and the light's tick are no worse than on `main` (live, back to
back), and its tests pass. One uploader thread serves the four cameras (92 frames a second); the probe says
whether it keeps up, and one per camera follows only if not.

### C. smile checked

smile imports `modules.camera`, `gl`, `face`, `video`, `inout`, `settings`, `panel` and `utils`, not `inference`
or `board`, and was committed today (its document, `SMILE.md`). The check: whether it runs on this branch, which of
its threads run a receiver's work in a callback (its camera's frame callback into the render and the face finder),
and what moving it onto the rules would take; reported, nothing changed.

### D. hd_trio back, after the 2D fluid

hd_trio comes back once [line 5](#5-the-2d-flow-follows-the-3d) has rebuilt the 2D fluid, on B's wiring. Known
now: its crops (the pose model's at its input size, RVM's at 768 × 1024 from the same region, both by
`CropExtractor`), its recordings as before (mp4 chunks any player plays, stored mirrored, a pose file per chunk,
RECORDING.md's plan), and its pose-following center layers, which stay. Planned when it starts.

### Details

- One line per stream, average and 1% low: the panel's rates as frame-time analysis gives them (RTSS,
  CapFrameX), one pinned line per stream, Render, Video, Captures and Motion,
  `<name>  <average> fps  low <1% low>`, both from one record of each stream's frame times. The average says how
  many frames came (a count for the cadence: 50 when none is lost); the 1% low, the rate over the slowest 1% of
  frames, says how steady, catching hitches without being set for a whole window by one, as the minimum (1 / the
  worst gap) is. It replaces the minimums (`Video min`, `Captures min`, `Motion min`) and gives the render the
  same two numbers; a frame-time line beside it only if asked.
- The color in one kernel: the color's conversion, straightening and mirroring as one CUDA kernel: from the
  camera's NV12 23 GPU ops become 2, about 0.4 ms less GPU a frame **(deduction)**. After the live run of
  [line 7](#7-recording-and--sim), which measures it on the camera.
- The previous crop: the optical flow on the crops (`enable_prev_crop`, off in deep_flow, so it runs nothing)
  does not wait for the crops' event, and its pairing of the current and the previous image is to be checked
  when it is used.
- No captures: while no capture comes (the camera stalls, -sim is paused) the bridge keeps its last output and
  the fluid keeps taking it, where the motion holds its last velocity. What the flow should do then is not
  decided.

### Rules it touches

- A callback stores or enqueues what it receives and returns: A's four places and B's wiring.
- A receiver does its work on its own thread: the recording sync, capture sync, pose step and pipeline get one.
- A callback list has its own lock, copied under it and called outside it: every list A touches.
- The camera's thread runs only the camera: A1, B1.
- A consumer that needs every item gets a callback; one that needs the latest reads the board: B2, B3.
- `submit` enqueues deferred work, `process` transforms and emits synchronously: both syncs' `submit_*`,
  `PosesFromTracklets.submit`.
- Log exceptions with context, never fail silently: drops counted on the panel.
- Treat latency as a first-order concern: each step's run compares the latency series.
- Update all call sites, remove obsolete pathways: the runners' dispatch thread goes, white_space's video images go.
- Plans and commits: one commit per checked step, built on your go.

## 5. The 2D flow follows the 3D

The 2D system (`GradientFlow`, `bridge_2d`, `Fluid2D`, hd_trio) runs as on `main`
([FLOW.md's constraints](FLOW.md#constraints)). It follows the 3D's principles once the 3D is finished:

- The 2D fluid: it adds the injected velocity times dt (`Fluid2D.add_velocity`), as Stam's Stable Fluids
  adds a source; but the bridge's velocity is a velocity, in view widths per second like the
  fluid's, so it is taken as an acceleration, `velocity.gain` acts as a rate, and the fluid stays
  near 0.05 to 0.1 of the motion's speed **(deduction)**, a monitor needing a scale of 20 to 30 to
  show it. The 3D fluid takes the boundary by a drag and the body force's acceleration times dt
  ([FLUID_3D.md](FLUID_3D.md#the-inputs)); the 2D fluid is to follow it: the drag, the two systems,
  the one grid.
- The 2D motion on the Lucas–Kanade solve, in widths/s with its velocity gate, in place of `GradientFlow`.
- The 2D bridge rebuilt on the 3D bridge's filters, the three copied classes (`DensityBridge`,
  `TemperatureBridge`, `VelocityMagnitude`) shared again.
- hd_trio's density fade from motion: its wiring pass writes `density.fade_time` every tick from the pose's
  motion, a settings field written per tick. A per-tick value on the fluid layer would take it off the
  settings.

Each 2D stage gets its document as it is rebuilt (`BRIDGE_2D.md`, `FLUID_2D.md`).

## 6. A camera without depth

- The depth's mask with another mask source: deep_flow's mask is the depth's ([VIEW.md](VIEW.md#the-mask-and-the-depth-mask));
  a mask from RVM, for a camera without depth, would come back from hd_trio. With it the motion's `depth_mask`,
  the depth's own cut that gates z, is still the view stage's, and RVM's mask, on the crop, would have to be
  cut into the view's window.
- A depth for a 2D camera: the 3D fluid puts a pixel in at its z position and nothing where it has none
  ([FLUID_3D.md](FLUID_3D.md#the-bodys-depth)), so a 2D camera gives it nothing. A depth made from the image
  (a monocular depth model) on the view would give the motion its z and z position as a depth camera's does.

## 7. Recording and -sim

- deep_flow live: the cadence with the camera and a player in view; the view's outputs on the monitors; the
  depth's edges on the measure recording (z's step inside the body, z's noise at the edge and inside); the dye
  at drag 0.
- The recorded tracklets' checks: [RECORDING.md's Open](../recording/RECORDING.md#open).

## 8. The camera package cleaned

Built: recording is `modules/recording`, the cameras' sync `modules/inference`'s `CameraSync`, the OAK4's two
scripts in `scripts/`, one subject per file, `CameraHealth`, `KeypointJoin` and `oak.ClipCamera`, the public
boundary cut to what other packages use, and the reasons in [CAMERA.md](../camera/CAMERA.md). Checked by the
import and guard tests; the apps' runs and an OAK-D's clip path are the user's.

## The motion's details

- The false z at a limb's edge: a limb moving sideways before the body reads false z at its edges: the leading
  edge carries 43% of a sweep's z, the trailing edge 12%. The cause is the depth's ramp at the limb's edge. The
  camera's temporal filter is off and its confidence does not sharpen the ramp. Accepted for now; the cue not
  yet tried is a forward-splat count of the x/y motion (0 = uncovered, above 1 = covered).
- The range: `edge`, `tolerance` and `dead_zone` are in metres and tuned at 0.65–1.5 m. Stereo noise grows
  with the square of the distance, so at 2–3.5 m they may need to be in disparity **(deduction)**. This needs
  a recording at that distance.
- The depth velocity's `track`: its memory is named `track` in the code (`DepthVelocity._track`, the monitor's
  `z.track`), a word that says nothing on its own; it holds the followed disparity, its rate and its anchor.
  To be renamed by what it holds, with the monitor's port.
- A soft limit on the z speed: a guard to try on the motion beside its finite guard
  ([MOTION.md](MOTION.md#the-guard)): z as measured up to a knee (about 1.5 m/s), compressed above it toward a
  maximum (about 3 m/s), the speed gate (6 m/s) behind it; a curve per reading, no time filter, checked first
  against the spread of z speeds on a recording.
- A finer depth would give the mask and z a finer edge at the source; the camera's sizes are a given
  ([RESOLUTION.md](RESOLUTION.md#the-rules)).

## Parked

Not worked on until asked for; each with what is known.

- The 4K upscale: depth in 3D at the grid and detail in 2D at 4K, a march at the grid's size, one
  edge-aware upscale of the flat result and a 4K detail layer advected by the grid's velocity; what
  was found is in [FLUID_3D.md](FLUID_3D.md#the-densitys-detail).
- The particles' weight: the particles splatted into an `r32ui` grid by fixed-point
  `imageAtomicAdd` (it needs integers), converted to the density volume the buoyancy reads.
- The sparse grid: each step a pass marks the 8 × 8 × 8 bricks whose velocity or density is above
  a floor, grown by one brick, into a list with an atomic counter; every pass but the pressure then
  runs `glDispatchComputeIndirect` over the list, one workgroup a brick, and a brick off the list
  keeps its values. The volumes stay dense: 24 GB holds them, so GVDB's atlas, which saves memory,
  buys nothing here. The multigrid's coarse levels stay whole, an eighth and less of the work. Its
  gain is the share of bricks off the list; first one cheap pass counts the active share on a
  recording, and the grid is built only if it is under half.
- MacCormack's cost at 4K: a 4K texel reads 41 texels a step against semi-Lagrangian's 16 (the
  predictor, the velocity again, the forward trace, the cell, the limiter's eight), 3.7 ms for the
  4K × 16 density. Two ways down: MacCormack on the velocity and semi-Lagrangian on the 4K density,
  a setting; or the predictor reading its eight texels itself and keeping their min and max for the
  limiter, about a fifth less **(deduction)**.
- The density's detail carried by a flow map (BiMocq², Qu et al. 2019): the coordinates moved on
  the coarse grid and the 4K density read through them, baked every half second; its benefit to be
  shown first.
- Vortex particles shed by the body for swirls below the grid (He et al. 2011; Weißmann and
  Pinkall 2009).
- Each field panel showing its own input gains as well, the same values as `input`'s: shared
  between sibling panels, which `share` cannot do (it runs parent to child), so either hidden fields
  at the top shared down to both, or a share between panels. In a field panel the system's name is
  dropped as in `input`.
- The camera as a standalone OAK app: the depth's alignment, the PC's now, then needs a home.
