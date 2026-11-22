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

- B1. hd_trio: its settings (`MP4`), its wiring (the tracklets from each camera as they come, `expect_parts` per camera,
  `expect_poses` per camera and set), as white_space's is wired, and its pose `.h5` cut at the recorder's split.
- B2. The OAK-D's capture stamp is built (`oak_device/capture_clock.py`); its live check with an OAK-D is the camera plan's F1.
- B3. A white_space recording of the new layout played by hand, beside the old ones, which play as they are.

## C. The split test that sometimes fails

- C1. `test_a_split_keeps_the_streams_in_step_across_chunks` (`tests/test_round_trip.py`) fails about 2 runs in 5: the
split lands 1 to 3 frames early on one stream. No frame is lost (50 of 50 in all three streams), so it is the split's
timing, not the recording's data. Looked at when the refactor's A6 reaches the recording module.

## D. Questions for the recording review (the refactor's A6)

None open.

## F. The tree and what it exports

- F2. `__init__.py` exports only what other packages use. It keeps `ChunkStream`, `FrameMetadata`, `PoseArrays`,
  `Container`, `read_metadata`, `make_folder_name`, `Player`, `PlayerSettings`, `Recorder`, `RecorderSettings` and
  `replay` with `Camera` and `Tracker`. Out, used by no other package on the PC's deep-flow **(seen, grep)**:
  `ChunkInfo`, `ChunkMetadata`, `merge_chunk`, `finish_unmerged`, `make_file_name`, `make_chunk_name`,
  `RecordingFolder`, `Timeline`, `RecordingSync`, `RecordedCapture`, `FramePoses`, `frame_poses`, `PlayClock`,
  `ClipFrameCallback`, `LoopCallback`, and replay's `PersonDetector`, `BoxTracker`, `Track`, `LOST_FRAMES`,
  `MIN_OVERLAP`, `assign`, `overlap`. The package's tests import them from their files. Why: "Its `__init__.py` holds
  what other packages use". Tests: the recording tests, the guard tests, pyright.
- F3. The metadata file's name is made once, by a `make_metadata_name(cam_id, chunk)` in `chunk.py`, for
  `Recorder._metadata_path` and `metadata_beside`, which each build `<cam>_M_<chunk>.jsonl` themselves **(seen)**.
  Why: "Inside a package, code that does the same is written once". Tests: the recording tests.

## G. Text

- G1. Leftovers: the comment "The letter in a clip's file name" above `STOP_WAIT` in `recorder.py` goes, its labels
  being `chunk.CHUNK_STREAM_LABELS`; the log lines "VideoRecorder started" and "VideoRecorder stopped" say
  "Recorder"; `PlayerSettings.video_frame_types` reads "Frame types to play", not "to record" **(seen)**. Why:
  "Remove obsolete pathways"; one term everywhere; a description says what the value is.
- G2. Reasons and measurements leave the docstrings for RECORDING.md, which holds them already: `recorder.py`'s
  module docstring (keyframes, confidence), `recording_sync.py`'s (RECORDING.md's rules table), `FrameLines`'
  measurement (RECORDING.md's Measured), `chunk.py`'s "a chunk recorded before them reads them as off", and the
  markers' paragraph in `player.py` and `recording_folder.py` **(seen)**. Why: "A docstring says what the thing is"; "Reasons sit in the package's document, history in the
  commit". Check: none, text only.
- G3. RECORDING.md's link to `RECORDING_PLAN.md#3-the-poses-the-pose-model-drops` goes: the plan has no such topic
  **(seen)**. Why: a document describes what is built. Check: none, text only.

## The rules it touches

- Keep settings files in sync with code changes: hd_trio's `container` in its settings file
- Run a test of the app or a GPU harness only on the user's go, and only what the plan names: the checks of A
- Its `__init__.py` holds what other packages use: F2
- Inside a package, code that does the same is written once: F3
- Remove obsolete pathways; one term everywhere; `Field` descriptions: G1
- A docstring says what the thing is; reasons sit in the package's document: G2
- A document describes what is built: G3
