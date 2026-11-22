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

## D. Questions for the recording review (the refactor's A6)

- D2. Settled (2026-11-22): pose's `Recorder` becomes `PoseRecorder`, in [POSE_PLAN.md](../pose/POSE_PLAN.md) D1.
- D3. `RecordingSync`'s inbox has no limit, and the `Recorder` drops a whole new capture past `HELD` (1 GB)
  **(seen)**, by RECORDING.md's completeness first. The code stays. Whether the rule "When that part can't keep up,
  its queue drops the oldest item and counts it" needs a limit on where it applies is with the Rules thread.

## E. From the scan of 2026-11-22

Proposals; nothing is built.

- E1. `FrameHandOver.__init__` takes `num_cams` and never reads it **(seen)**; it goes, and `Player` and
  `test_frame_hand_over` stop passing it. Why: "Remove obsolete pathways". Tests: the recording tests.
- E2. `Player` binds `_on_refresh_path` to `PlayerSettings.video_path`, a `Field.INIT` value, which cannot change after
  `initialize()` **(seen)**; the bind and its unbind go. Why: leftovers; an INIT field has no change to hear. Tests:
  the recording tests.
- E3. `ChunkInfo`'s fields are listed twice, as `info_line`'s JSON and as `merge_chunk`'s tags **(seen)**; one
  function gives both their values, so a field added lands in both. `finish_unmerged` names the merged chunk
  `f'{cam}_{chunk}.mkv'` itself instead of by `make_chunk_name`, and its glob `f'{cam}_{label}_{chunk}.*'` rebuilds
  `make_file_name`'s pattern **(seen)**. A stem `<cam>_<x>_<chunk>` is parsed apart three times, by
  `metadata_beside`, `finish_unmerged` and `RecordingFolder.find` **(seen)**: one parse in `chunk.py` for the three.
  Why: "Inside a package, code that does the same is written once". Tests: the recording tests.
- E4. player.py's `Message` is a plain class with an `__init__`, and its `state` holds a `MessageType` **(seen)**: a
  frozen `@dataclass` with `kind`. `LoopCallback` also types the jump callbacks **(seen)**: one alias per job, or
  `Callable[[], None]` for both. `import logging` follows the relative imports in player.py and recorder.py
  **(seen)**. Why: "Use `@dataclass` for plain value objects"; a name says what the thing is; PEP 8. Tests: the
  recording tests.
- E5. replay's protocols take untyped callbacks: `PlayerFeed.add_frame_callback`, `add_jump_callback`,
  `PersonDetector.add_detection_callback` **(seen)**; they take `ClipFrameCallback`, `Callable[[], None]` and
  `Callable[[Detection], None]`, so pyright checks `replay.Camera`'s callbacks against what calls them. Why: "Use a
  `Protocol` for what a consumer needs from a collaborator", typed. Tests: pyright, the recording tests.
- E6. Reasons leave the docstrings that G2 did not name: `replay/camera.py` ("so `-sim` needs no OAK plugged in", "so
  the uploader straightens them as it does live", "so nothing but the hand-over runs on the player's"),
  `chunk_readers.py` (the unsynced rig, "So cameras that dropped unlike frames start and end at one moment"),
  `play_clock.py` ("so a frame comes as late after its capture as it came live"), `frame_hand_over.py` ("so the
  chunk after the one playing can start beside it"), `chunk.py` ("which re-encodes nothing and takes moments",
  "where any tool can read it"), `recorder.py`'s module docstring ("so the correction can still be tuned on them"),
  `PersonDetector` ("Named here so the contract does not import the inference package"), `FrameHandOver.__init__`
  (VIDEO first: "its readings set up what the depth of the frame uses"), and the comments on `HELD` ("some 6 s of
  four cameras") and `KEPT` ("320 ms at 50 fps, three times the slowest pose"), which repeat RECORDING.md's constants
  table **(seen)**; RECORDING.md takes any it does not hold yet, the `PersonDetector` and VIDEO-first reasons among
  them **(seen: it holds neither)**. `chunk.py`'s comment above `PoseArrays` says "the app converts"; `recording_sync.pose_arrays`
  converts **(seen)**: the sentence goes. `PlayerSettings.fps` reads "Simulation FPS"; it paces a clip without
  recorded times and turns the playhead's frames into seconds **(seen)**. `RecorderSettings.fps` reads "Camera frame
  rate", without its unit; `PlayerSettings.in_out` ("Plays between the two handles; each recording keeps its own")
  says neither the value (two fractions of the length) nor its unit; `RecorderSettings.recording` reads "Recording"
  **(seen)**. Why: "A docstring says what the thing is"; "Keep `Field` descriptions to one short line of panel text:
  what the value is and its unit"; docs.md "State each fact in one place". Check: none, text only.
- E7. B2 leaves this plan: it says what is built and points at the camera plan's F1, which holds the check
  **(seen)**. B1 drops "its settings (`MP4`)", and "The rules it touches" drops its line on hd_trio's `container`:
  hd_trio's `studio.json` and `umu.json` hold `"container": "MP4"` already **(seen)**. Why: a plan holds only what is
  left.
- E8. `PlayerSettings.replay` ("Play on the host without an OAK device") leaves the player's settings for each app's
  camera settings: `OakGroup.replay` in `apps/hd_trio/settings.py` and in `apps/white_space/settings.py`, beside
  `sim_enabled`, an INIT field with the same description. Today hd_trio's and white_space's `main.py` read
  `setup.camera.simulator.replay` to choose the camera and to set `device_tracklets`; the `Player` never reads it
  **(seen)**. Their `main.py` read `setup.camera.replay`; `scripts/white_space_probe.py` sets
  `root.setup.camera.replay` and its docstring says `replay` **(seen: it sets `simulator.replay` today)**. In the
  settings files under `data/settings`, the key moves from `camera.simulator` to `camera`: hd_trio's
  `setup/studio.json` and `setup/umu.json`, white_space's `setup/studio.json` and `setup/public.json` **(seen)**.
  deep_flow's and low_flow's `setup/studio.json` hold `"replay": true` under `simulator` too, and their `main.py`
  build `replay.Camera` in -sim without reading it **(seen)**: the key goes from both files. What a key the settings
  no longer have does on load is not checked **(deduction: the files are kept in sync anyway)**. Why: "Settings
  connect the user and a component"; the choice of camera is the app's, not the player's. Tests: the recording
  tests, and each of hd_trio and white_space started briefly with `-sim` and `replay` on (no camera needed).
- E9. `RecordingSync`'s inbox length shows on the recorder's panel as a READ field, `RecorderSettings.sync_queued`
  ("Parts waiting in the sync's inbox"), set beside the other counts in `Recorder._update_missed`; `RecordingSync`
  gives the length as a property (`queued`), as `waiting` gives it with the cameras' waiting lines **(seen:
  `waiting` adds `len(self._inbox)` and every camera's `waiting`)**. The inbox alone is shown, since each camera's
  waiting line holds at most `KEPT` by design **(seen)**. A warning is logged once a recording when the inbox passes
  `4 * KEPT * num_cams` parts (256 for four cameras): four parts come per frame (frame, tracklets, asked, poses) **(seen:
  `_Part`)**, so past it the sync's thread is a whole `KEPT` window behind on every camera, some 320 ms at 50 fps,
  where captures leave without their parts anyway **(deduction)**. Why: the inbox is not bounded, and nothing shows it
  growing during a long recording. Tests: the recording tests.

## F. Code (the scan of 2026-11-22, second pass)

Proposals; nothing is built.

- F1. `chunk.finish_unmerged` catches an exception per chunk, logs it and goes on; `Recorder._merge_chunk` logs its
  own failure. Today one `OSError` (a `mkdir`, a move or an unlink) ends the `RecorderUnmerged` thread and leaves the
  chunks after it, and `_merge_chunk` has no `try` **(seen)**; `merge_chunk` and `keep_chunk` log their own failures
  already **(seen)**. Why: "A thread's loop catches an exception per item, logs it and goes on with the next"; "Log
  exceptions with context; never fail silently". Tests: the recording tests.
- F2. `replay.Camera._on_detection` only queues the detection and sets the camera's `Waker`; the box tracker runs on
  `replay.Camera`'s own thread, beside the recorded tracklets. Today the detector's callback runs the whole SORT
  tracker (Kalman and the Hungarian assignment) on the detector's thread **(seen)**. It matters only with the
  detector on, off in deep_flow's settings file **(seen in RECORDING.md)**. Why: "A callback only hands the item over
  and returns". Tests: the replay tests, the recording tests.
- F3. `RecordingFolder.chunks` becomes `last_chunk`: it is the last chunk's number, as its docstring says, while
  `Player.get_num_folder_chunks` and `Player.num_chunks` call it a count **(seen)**; they follow. Why: "Name a concept
  after what it is in the domain, and use that one term everywhere it appears". Tests: the recording tests.
- F4. A `Protocol` for what `FrameHandOver` needs of `ChunkReaders` (`chunk`, `streams`, `frame`, `info`, `clock`)
  **(seen)**, so `test_frame_hand_over`'s `FakeReaders` meets it and its five `# pyright: ignore[reportArgumentType]`
  go **(seen)**. Why: "Use a `Protocol` for what a consumer needs from a collaborator, an `ABC` for a family of
  classes that share code"; "Suppress only when a third-party stub is provably wrong or the type system cannot express
  the code, never to hide an error in our own code". Tests: pyright, the recording tests.

## G. Text (one commit)

- G1. RECORDING.md's first table names `modules/camera/sync/capture_match.py`, which is `modules/camera/capture_match.py`
  since the camera's sync folder went **(seen)**. Its sentence "A chunk records it when its first depth frame has
  one" follows the keyframe sentence, so "it" reads as the keyframes **(seen)**: it names the confidence. Why: "When
  renaming or removing a section, state or setting, search the documents for references to it"; "Write plain
  declarative sentences, without emphasis phrases". Check: none, text only.
- G2. `num_cams` stands a column out of its block in `RecorderSettings` and `PlayerSettings`, left from the
  `num_cameras` rename **(seen)**: aligned. Why: PEP 8 and the block's own layout. Check: none, text only.

## The rules it touches

- Keep settings files in sync with code changes: hd_trio's `container` in its settings file; the `replay` key, E8
- Run a test of the app or a GPU harness only on the user's go, and only what the plan names: the checks of A; E8's starts
- Settings connect the user and a component, not two components: E8
- When that part can't keep up, its queue drops the oldest item and counts it: D3 (with the Rules thread), E9
- Similar things share the noun and differ by a qualifier: D2 (built in POSE_PLAN.md D1)
- Remove obsolete pathways: E1, E2
- Inside a package, code that does the same is written once: E3
- Use `@dataclass` for plain value objects; PEP 8: E4
- Use a `Protocol` for what a consumer needs from a collaborator: E5
- A docstring says what the thing is; `Field` descriptions: E6
- A plan holds only what is left (plans.md, "A built step leaves the plan"): E7
- A thread's loop catches an exception per item, logs it and goes on with the next: F1
- Log exceptions with context; never fail silently: F1
- A callback only hands the item over and returns: F2
- Name a concept after what it is in the domain, and use that one term everywhere it appears: F3
- Suppress only when a third-party stub is provably wrong or the type system cannot express the code, never to hide an error in our own code: F4
- Use a `Protocol` for what a consumer needs from a collaborator: E5, F4
- Keep `Field` descriptions to one short line of panel text: what the value is and its unit: E6
- When renaming or removing a section, state or setting, search the documents for references to it: G1
