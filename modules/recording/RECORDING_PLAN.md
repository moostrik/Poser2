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

## The rules it touches

- Run a test of the app or a GPU harness only on the user's go, and only what the plan names: the checks of A
- Similar things share the noun and differ by a qualifier: D2 (built in POSE_PLAN.md D1)
