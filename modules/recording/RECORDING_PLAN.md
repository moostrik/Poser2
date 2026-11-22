# The recording plan

What is next for recording and -sim, in order. How it works is in [RECORDING.md](RECORDING.md); the apps' move to
the recorder is in [the refactor's plan](../../REFACTOR_PLAN.md). **(deduction)** marks what is inferred, **(site
fact)** what was told, **(seen)** what is read in the code.

## A. Checks waiting for the camera

deep_flow live, with the camera and a player in view:

- A1. A new recording: its counts on the panel, its tracklets and latencies in the metadata, its first frames with
  their tracklets (`Recorder.expect_parts`), and its playback as before.
- A2. The cadence; the view's outputs on the monitors; the depth's edges on the measure recording (z's step inside the
  body, z's noise at the edge and inside); the dye at drag 0.
- A3. -sim on the new recording with the recorded tracklets against live on the same scene, by
  `scripts/cadence_probe.py`, and against -sim with the detector. Whether the live 99.4% of
  [INPUT_THREADS.md](../flow/INPUT_THREADS.md#the-cadence) had a player in view is not known, likely yes **(site
  fact)**, so the live number is taken anew.
- A4. The pose points and boxes over the image, live and in a replay, to see the tracklets straightened with their own
  frame and the board's poses as live's.
- A5. The device's model looks at the frame unstraightened, which for a few degrees of tilt finds the same players
  **(deduction)**.

## B. The recorder in the other apps

With the apps' refactors ([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)), each tested with its refactored app:

- B1. hd_trio: its wiring (the tracklets from each camera as they come, `expect_parts` per camera, `expect_poses` per
  camera and set), as white_space's is wired, and its pose `.h5` cut at the recorder's split.
- B3. A white_space recording of the new layout played by hand, beside the old ones, which play as they are.

## D. Questions for the recording review (the refactor's A6)

- D2. Settled (2026-11-22): pose's `Recorder` becomes `PoseRecorder`, in [POSE_PLAN.md](../pose/POSE_PLAN.md) D1.

## H. Drops shown to the user

- H1. The player's hand-over (`player/frame_hand_over.py`, `FrameHandOver.submit_frame`) drops frames in two ways:
  frames left without their other stream, counted in `_unpaired` and only logged at info in
  `Player._update_readouts` **(seen)**; and a frame older than one gone out, not counted at all **(seen)**. Both are
  counted in one count, `take_unpaired` becomes `take_dropped`, and `Player._update_readouts` adds it to a READ field
  on `PlayerSettings`, `dropped_frames` ("Frames dropped since the play started: unpaired or out of order"), set to 0
  where a folder's play starts (`Player.play`, the start button and a folder change), not at a chunk's start or a
  loop. The info log goes; the panel holds the count. Why: a part that drops an item shows the count to the user,
  and the second drop is not even counted. READ fields are not written to settings files **(seen,
  `BaseSettings.to_dict`)**, so no settings file changes. Tests: the player tests, and one in
  `test_frame_hand_over.py` that hands a frame older than one gone out and finds it counted.
- H2. The replay camera's detection queue (`replay/camera.py`, `Camera._on_detection`) keeps `DETECTIONS_KEPT` (4)
  and drops the oldest past it, counted in `dropped_detections` and one log warning, shown nowhere **(seen)**. The
  limit came with F2 (cdbd69c7) beyond its step, and rests only on the queue line that was removed **(site fact)**. The
  queue loses its limit and its drop: every detection reaches the tracker, so a replay tracks the same from run to
  run. `DETECTIONS_KEPT`, `dropped_detections` and the warning go; nothing drops, so nothing is shown. A detection
  is a few boxes, so an unbounded queue costs little **(deduction)**. RECORDING.md's sentence on the drop follows.
  Touches `replay/camera.py` and RECORDING.md; no app code, no settings files. Tests: in
  `replay/tests/test_camera.py` the drop-oldest test becomes one that the tracker gets every detection, on the
  camera's thread.

## The rules it touches

- Run a test of the app or a GPU harness only on the user's go, and only what the plan names: the checks of A
- Similar things share the noun and differ by a qualifier: D2 (built in POSE_PLAN.md D1)
- A part that drops an item shows the count to the user: H1, H2 (nothing dropped)
