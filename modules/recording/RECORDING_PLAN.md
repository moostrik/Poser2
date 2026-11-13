# The recording plan

What is next for recording and -sim, in order. How it works is in [RECORDING.md](RECORDING.md); the apps' move to
the recorder is in [the app refactor](../../apps/APPS_PLAN.md). **(deduction)** marks what is inferred, **(site
fact)** what was told.

## 1. Checks waiting for the camera

deep_flow live, with the camera and a player in view:

- A new recording: its counts on the panel, its tracklets and latencies in the metadata, its first frames with
  their tracklets (`Recorder.expect_parts`), and its playback as before.
- The cadence; the view's outputs on the monitors; the depth's edges on the measure recording (z's step inside the
  body, z's noise at the edge and inside); the dye at drag 0.
- -sim on the new recording with the recorded tracklets against live on the same scene, by
  `scripts/cadence_probe.py`, and against -sim with the detector. Whether the live 99.4% of
  [INPUT_THREADS.md](../flow/INPUT_THREADS.md#the-cadence) had a player in view is not known, likely yes **(site
  fact)**, so the live number is taken anew.
- The pose points and boxes over the image, live and in a replay, to see the tracklets straightened with their own
  frame and the board's poses as live's.
- The device's model looks at the frame unstraightened, which for a few degrees of tilt finds the same players
  **(deduction)**.

## 2. The recorder in the other apps

With the apps' refactors ([APPS_PLAN.md](../../apps/APPS_PLAN.md)), each tested with its refactored app:

- hd_trio: its settings (`MP4`), its wiring (the tracklets from each camera as they come, `expect_parts` per camera,
  `expect_poses` per camera and set), as white_space's is wired, and its pose `.h5` cut at the recorder's split.
- The OAK-D's capture time: the oak backend (depthai 2) stamps a frame's arrival, so hd_trio and white_space record
  USB's jitter in their times. Oak4's `capture_time` shows the way; it needs an OAK-D to test.
- A white_space recording of the new layout played by hand, beside the old ones, which play as they are.
- The glossary's `Encoder` for `CoderFormat` and `CoderType`: a rename through `modules/video` and the three apps.

## 3. The poses the pose model drops

In -sim, with the recorded tracklets and the pose model on, about one frame in six with a player gets no poses (107
of 596 on `hands`, [RECORDING.md](RECORDING.md#measured)): the pose model's TensorRT runner keeps one batch in
flight and one waiting (`pose/runner_trt.py`), and a batch that comes while one waits replaces it. At 50 fps a frame
comes every 20 ms, so the host's 34 ms keeps up only while batches overlap, and drops when one runs long. With the
recorded tracklets -sim loads the GPU as live does, its one addition, the decoding, on the iGPU and the CPU, so live
drops about as many **(deduction)**; no live number exists.

In order:

1. Live, with a player in view: a new recording's `missed_poses` and its frames with poses, the live drop rate.
2. In -sim on the same scene, the host's 34 ms split: RTMPose's own time (the predictor's inference timer), the
   wait for the crops' event behind the GPU's queue, and the wait in the runner's waiting slot, each per frame.
3. From that split, what stops the drops: RTMPose's own time over 20 ms means a faster model or input; a long wait
   for the crops means the pose work queued behind the fluid's (its own stream's priority, or the order of the work
   on the GPU); a long wait in the slot means the runner's hand-over. Each is weighed against the poses' precision
   (RTMPose stays FP32).

## The rules it touches

- Use the terms in glossary.md: `CoderFormat` and `CoderType` stay until their rename
- Keep presets and settings schemas in sync with code changes: hd_trio's `container` in its preset
- Run a test of the app or a GPU harness only on the user's go, and only what the plan names: the checks of 1 and 3
