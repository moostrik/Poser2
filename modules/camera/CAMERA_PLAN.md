# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F). Who uses what, from the code: the contract
(`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture match) is read by
every app and by inference, recording, tracker and render. The backends are each one kind of camera, used now by
the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, `vimba` and `webcam` by smile. That is
general code with one user, which stays.

Built: 1, Vimba's user set is a setting, `VimbaCameraSettings.user_set` (empty loads none), `UserSet2` in smile's
four presets; 2, the four backends' callback lists have their lock, registered under it, copied under it, called
outside it; 3, a camera's number is `cam_id` on the contract and every backend (the replay camera's too), Vimba's
serial is `serial`, the package spells American, its tests time with `perf_counter`; 4, the tilt correction
fits the picture's height to the frame's, with the angle not capped.

## A. Depth confidence to the motion

Neural Depth also gives a confidence per depth pixel (`NeuralDepth.confidence`). Today only the depth is taken, and
the confidence is used once, as the camera's cut (`DepthSettings.confidence`, 125 by default, 51 in deep_flow's preset).
Probed on the OAK4 (2026-11-18, `NEURAL_480`, 100 frames, no IR light): the confidence is 480 × 300 RAW8 like the depth,
with the depth's sequence number and timestamp; higher is surer, on the cut's 0–255 scale (Luxonis docs). It ran 73 to
255, median 230, so the preset's 51 cuts nothing. 2.2% of pixels have no depth at every cut, 0 included; theirs is lower
(median 158 against 231) but overlaps the valid pixels' range, so no one value parts them.
The motion is to weight each depth sample by it (Flow Motion's line, in [FLOW_PLAN.md](../flow/FLOW_PLAN.md#c-the-motion)),
so it goes the depth's whole way: camera, contract, alignment, recording, view, motion. It rides with the depth at each
step, never as a frame of its own, so it can never be paired with another frame's depth.

- **A1. The camera.** The oak4 pipeline takes `neural.confidence` beside `neural.depth`, both into one `Sync` node, so
  one message holds a frame's depth and its confidence and the backend needs no join **(deduction: the two come off one
  node with one timestamp)**. The DEPTH `CameraFrame` gets `confidence: np.ndarray | None`, uint8 on the depth's view;
  None from every other backend. OAK4.md's Depth section, with the probe's numbers. It adds 7 MB/s on the
  link at 50 fps **(deduction: 480 × 300 bytes a frame)**, room USB has.
- **A2. The alignment.** `DepthAligner` moves the confidence with the depth that wins a cell: the atomic minimum's key
  becomes the depth's float bits with the confidence in the lowest 8, so the nearest point brings its own confidence in
  the same one pass. That costs the depth 15 bits of precision, under 0.1 mm at 3 m, against the recording's 3 mm steps.
  Both paths, the kernels and the torch ops, the tests holding one to the other. `GpuDepth` gets `confidence`, 0 to 1,
  0 where no depth lands. INFERENCE.md's upload.
- **A3. The recording and -sim.** A recording keeps it as a third stream in the chunk's `.mkv`: 8-bit HEVC on the
  iGPU (`hevc_qsv`, the confidence in Y over a neutral chroma), lossy as the depth is, since a weight a few levels off
  does no harm; the depth's `keyframe_interval` and no B-frames, so a seek finds all three streams whole. The replay
  camera hands it on. Not in the depth stream's empty chroma:
  the iGPU's `hevc_qsv` encodes 12 bits only as 4:2:0, which halves the confidence to 240 × 150, and 4:4:4 only up to
  10 bits, which would take the depth from 3 mm to 12 mm steps (checked 2026-11-18). A recording without it plays with
  confidence None. This step is the recording thread's ([RECORDING_PLAN.md](../recording/RECORDING_PLAN.md)); it builds
  on A1's field.
- **A4. The view and the motion's input.** The depth's texture carries the confidence in green, the metres staying in
  red, so every stage on the way keeps the two together: the crop cuts both, the upscale blends both with the same
  weights, the fill keeps the confidence where the depth is the camera's and gives 0 where it filled. With confidence
  None, green is 1, so the motion runs as now on older recordings. `DepthView` and `LucasKanadeFlow3D.update` then get
  it with the depth; the motion's use of it is Flow Motion's. A monitor shows it. VIEW.md's depth.

The recordings measured so far have no confidence, so the motion's A/B needs a new recording.

## Open

- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.

## The rules file

`camera.md`, line by line:
- "The camera's thread runs only the camera": true of every app's wiring (the cameras' callbacks only queue); about
  this module; checkable in the apps' `main.py`.
- The two-venv line: true (`oak` and `oak4` call `require_backend` on import, which raises in the wrong venv); about
  this module. Its naming of `oak_backend()` went to the rules thread and is fixed (212331ae).
- Missing: nothing; the contract's registration before `start()`, which clashed with CLAUDE.md's callback-list lock,
  went with step 2.

## Rules it touches

- What one app decides sits in that app (`modules/` holds what is general): 1.
- A callback list has its own lock: 2.
- Use the glossary's terms (`cam_id`, no `camera_id`, American spelling): 3.
- Read `perf_counter` for every stamp and duration: 3.
- Open questions and unbuilt work sit in the plan, not the document: 3.
- A document describes what is built: 4 (CAMERA.md's and OAK4.md's tilt correction).
