# Recording and playback

How a camera's frames, its tracklets and the host's poses are recorded and played back, so `-sim` runs an app on a
recording without a device, with the camera's own tracklets and live's poses, at their timing. This document holds
the design and its reasons; how each part works is in its module's docstring:

| Part                         | Module                                                          |
|------------------------------|-----------------------------------------------------------------|
| what a chunk holds           | `container/chunk.py`                                            |
| a capture's parts together   | `recorder/recording_sync.py`, `../sync/capture_match.py`        |
| writing                      | `recorder/recorder.py`, `modules/video/stream_writer.py`        |
| playing                      | `player/player.py` and [its parts](#the-players-parts)          |
| the camera of a recording    | `../replay/camera.py`, `../replay/tracker.py`                   |
| the poses of a recording     | `modules/inference/recorded_poses.py`                           |

The threads a frame goes through after it are deep_flow's ([INPUT_THREADS.md](../../flow/INPUT_THREADS.md)).

**(deduction)** marks what is inferred from the code, **(site fact)** what was told; the rest is in the code.

## What is recorded

A recording holds what the camera delivered, nothing straightened and nothing mirrored, and only what its playback
plays. The camera cannot straighten its color frame itself (the oak4's `Warp` takes grey frames only, and the depth
is aligned on the GPU, where it is straightened with the color), so straightening and mirroring are the uploader's,
live and in -sim alike, and a recording made before them plays through the same path as the camera. The tracklets
are stored as the camera gave them for the same reason: the uploader straightens them with the homography of the
frame they were found in (`Uploader.submit_tracklets`).

Every part carries its capture time (`timestamp`) and the moment the backend handed it over (`arrival`), on
`perf_counter`. Their difference, the latency, is stored with the part, so a playback hands each part as late after
its capture as it came live.

A recording is a folder named after its start time (and the panel's `name`), with one `<cam>_<chunk>.mkv` per
camera and chunk, with depth or color only:

| Stream / part | Holds                                                                                   |
|---------------|-----------------------------------------------------------------------------------------|
| stream 0      | color, the camera's encoder (`video_encoder`, `video_format`)                           |
| stream 1      | depth when recorded, uint16 mm, `DEPTH_ENCODER`, the same `keyframe_interval`           |
| tags          | the info line's values                                                                  |
| attachment    | the metadata, one JSON line each (`.jsonl`)                                             |

| Where      | What                                                                                     |
|------------|------------------------------------------------------------------------------------------|
| info line  | width, height, the lens (`intrinsics`), `fps`, the depth's alignment, `depth`            |
| info line  | `flip_h`: the frames are still to be mirrored, as in `CameraFrame`                       |
| per frame  | its number, `timestamp` (the capture)                                                    |
| per frame  | tilt and roll (°), exposure (µs), ISO (`FrameReadings`), null without                    |
| per frame  | `video_latency` (s): the frame's arrival minus its capture                               |
| per frame  | the tracklets: `id`, `status`, `roi`, `age`, `keypoints` (null without a pose model)     |
| per frame  | `tracklet_latency` (s): their arrival minus their frame's capture                        |
| per frame  | `poses`: per `track_id`, the box (x, y, width, height) and the points (17 × x, y, score) |
| per frame  | `pose_latency` (s): the poses' arrival minus the frame's capture                         |

- One line per frame, always, so a line's place is its frame's number.
- The tracklets are null for a frame none came for and an empty list for a frame with no player.
- The poses are null for a frame the model gave none (no player, or a batch it dropped), so a replay drops the
  frames live did.
- The poses are the RAW stage's: each pose's crop as its box and the pose model's points in it. The box goes with
  the points, so a replay does not depend on the box smoother and predictor reproducing live's box: the predictor
  reads time, and a replay's stamps are not live's arrivals.
- Not recorded: the homography and the tilt offset (a playback makes its own, from the recorded tilt and the live
  offset), the model chain (the keypoints say whether a pose model ran), the camera's settings, the depth's latency
  (the player hands a frame's depth with its color).
- The in and out markers are the recording's, in `in_out.json` in its folder, not in the chunks: a recording is
  many chunk files, ffmpeg changes a file's tags only by writing the whole file anew, and `mkvpropedit`, which
  edits them in place, is not installed.
- A recording of the layout before this one plays wrong: its `flip_h` meant "mirrored before storing".

## Recording

deep_flow records live and in -sim alike: a recording played is recorded again as the replay camera hands it on,
with the poses the app finds in it, so a recording without poses gets them by being played once with the pose
model on. The host detector's tracklets are not a camera's and stay out. The recorder sits behind the live path:
the app hands a camera's frames and tracklets to the uploader first.

The rules the recorder keeps, each with its reason:

| Rule                                                                 | Why                                                                                   |
|----------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Completeness first: a capture waits for its parts                    | a recording is played many times; the live `CaptureSync` puts latency first instead   |
| No wait is timed: a part of a later frame says this one's is missing | every stream comes in the order of capture, so the order settles it without a clock   |
| At most `KEPT` captures wait per camera                              | the one case order cannot settle: a stream that stopped                               |
| A part a camera never sent is not waited for                         | a camera without depth, tracklets or poses records beside one with them               |
| A frame nobody asked poses for does not wait for them                | the app says per image whether the pose model was asked (`Recorder.expect_poses`)     |
| Each camera is synced under its own lock, in its own order           | the order between cameras is not kept live either                                     |
| A capture is dropped whole, before any part of it is written         | a chunk's color, depth and metadata then count alike, so frame `n` is one capture     |
| One bound for what waits for the encoders, in bytes, for all cameras | a bound per queue grows with the cameras                                              |
| A split cuts every camera at one capture time                        | the cameras' chunks start and end at one recorded moment, so a play runs through it   |
| On stop, the captures waiting are settled by the frames after it     | their tracklets and poses still come, after the last frame taken                      |

What a capture went without is counted on the panel (`missed_depth`, `missed_color`, `missed_tracklets`,
`missed_poses`, `dropped`) and warned of by the recorder's thread, at most once a second; what came before a
camera's first capture of a recording, a few frames at every start, is counted apart and not warned of. An encoder
that dies ends its writer and the recording; a recording that cannot start says why once and is given up. What a
crash left unmerged is merged beside the next recording.

| Constant            | Value | What                                                                              |
|---------------------|-------|-----------------------------------------------------------------------------------|
| `CAPTURE_TOLERANCE` | 2 ms  | the parts of one capture carry timestamps this close (oak4: 0.12 ms apart)        |
| `KEPT`              | 16    | captures waiting per camera: 320 ms at 50 fps, three times the slowest pose       |
| `STOP_WAIT`         | 1 s   | a stopping recording takes the parts its last captures wait for                   |
| `SPLIT_WAIT`        | 1 s   | a split waits for a camera's first capture after the cut, then cuts it anyway     |
| `HELD`              | 1 GB  | what waits for the encoders: some 6 s of four cameras with depth **(deduction)**  |

## Playback

`Player` plays a recording folder between the panel's in and out markers, on one clock, the recorded time
(`PlayClock`). The rules it keeps, each with its reason:

| Rule                                                                       | Why                                                                              |
|----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| A frame comes at its recorded capture plus its `video_latency`             | as late after its capture as live, and a frame lost live leaves its gap          |
| One origin for every camera and stream, from the earliest first frame      | cameras that started apart or dropped unlike frames play in step                 |
| A frame's streams go out together, stamped with its capture on the clock   | downstream takes time from the frame, as live                                    |
| A reader that falls behind moves the origin for all, as far as it is late  | one stall moves the clock once, and the cameras stay together                    |
| The next chunk carries the clock on and plays as soon as it is loaded      | its readers hold each frame until its moment, so no camera's first frame is late |
| A loop, a seek or another folder starts the clock anew, and is told        | the replay camera and `RecordedPoses` drop what waits (`add_jump_callback`)      |
| Each camera's frames go out under its own lock, in chunk and frame order   | a camera never waits for another's; a frame older than one gone out is dropped   |
| A camera plays the streams its chunk has                                   | a color-only camera plays beside one with depth                                  |
| The markers, a seek and a scrub are places on camera 0's frames            | the other cameras play the same recorded moment, found by time                   |
| A stop ends a reader's decoder                                             | a decoder that stalled cannot hold the player                                    |

The cameras of a rig run unsynced, so a camera's frame of one round is up to half a frame period from camera 0's.
Each camera plays its frames captured from half a period before camera 0's first frame of the range to half a
period after its last; a range from a chunk's start or to its end takes every camera's whole, where the recorder
cut them all at one time. How far the cameras play apart is `drift`, the recorded capture times of their last
frames gone out, in ms. A clip without metadata plays at `fps`.

deep_flow's player decodes color to NV12 (`Player(nv12=True)`), as the camera hands it: half the bytes of BGR
through the pipe, and live's upload path. A chunk's metadata is read once and kept for the chunk playing, the one
loaded and the last played, and its lines are parsed as the frames are read, so no load holds the interpreter.

### The player's parts

| File                            | Class                         | Holds                                                           |
|---------------------------------|-------------------------------|-----------------------------------------------------------------|
| `player/player.py`              | `Player`                      | the thread, its states and messages, the settings, the readouts |
| `player/play_clock.py`          | `PlayClock`                   | the recorded time of a play                                     |
| `player/chunk_readers.py`       | `ChunkReaders`                | one chunk's readers of every camera, its metadata and its clock |
| `player/frame_hand_over.py`     | `FrameHandOver`               | each camera's frames handed over in order, the drift            |
| `container/recording_folder.py` | `RecordingFolder`, `Timeline` | the folders, a folder's chunks and markers, places on them      |

## The replay camera

`replay.Camera` is a `CameraSource` without a device. Its frames come as the camera delivered them: `flip_h` as
recorded, depth frames with the recorded alignment, and with `tilt` the homography of the recorded tilt plus the
live offset on the recorded lens. That matches live's to within `TILT_STEP` (0.25°): live makes it from the IMU's
newest tilt, the recording keeps the tilt read with the frame **(deduction)**.

Its tracklets come one of two ways, by `PersonDetector.enabled` (off in deep_flow's preset):

| Detector | Tracklets                                                                     | Against live                                |
|----------|-------------------------------------------------------------------------------|---------------------------------------------|
| off      | the recorded ones, at their frame's stamp plus their `tracklet_latency`       | the same messages at the same timing        |
| on       | YOLOv8n on the uploaded image through `replay.Tracker`, marked `straightened` | another model, timing and GPU load          |

The recorded ones go out from the camera's own thread, so nothing but the hand-over runs on the player's.

## The recorded poses

`RecordedPoses` (`modules/inference`) is the RAW stage's async step in -sim, around `PoseInference`. With
`recorded_poses.enabled` on (deep_flow's default) the pose model does not run: the crops are cut for what reads
them (the optical flow, hd_trio's masks, the previews), and each pose gets its recorded box and points by
`track_id`, handed on after the recorded `pose_latency`. The `track_id`s come from the host tracker on the recorded
tracklets, the same tracker on the same messages as live, so they match the recording's **(deduction)**. With it
off the pose model runs as live, with live's GPU load, its poses differing a little every play. It is apart from
`pose.enabled`, which the preset shares between live and -sim.

A recording without poses plays boxes without points while it is on, as one without tracklets plays no players with
the detector off. The pose recorder of `modules/pose` (`Recorder`, HDF5 per stage) is another thing: it records a
stage's features for the show to play, on the wall clock, apart from the video.

## The camera module's layout

`modules/camera` is split by role, as capture libraries are (RealSense, ffmpeg, GStreamer): the data types, the
devices, the calibration, the matching by time, the file's format, the recorder and the player.

| Folder                 | Role                         | Holds                                                               |
|------------------------|------------------------------|---------------------------------------------------------------------|
| the root               | the contract                 | `source`, `frame`, `tracklet`, `settings`, `camera_check`, `depthai_backend` |
| `oak`, `oak4`, `replay`, `vimba`, `webcam` | the devices | a backend each                                                    |
| `geometry`             | the calibration              | the lens, the warp, the tilt (`geometry`, `tilt_correction`)        |
| `sync`                 | matching by capture time     | `capture_match` (shared with `modules/inference`'s `CaptureSync`), `frame_trigger` |
| `recording/container`  | the file's format            | `chunk`, `recording_folder`                                         |
| `recording/recorder`   | writing                      | `recorder`, `recording_sync`                                        |
| `recording/player`     | playing                      | `player`, `play_clock`, `chunk_readers`, `frame_hand_over`          |

`modules/camera/__init__.py` is the one public boundary and exports every folder's names; each folder has its tests
beside it, and `recording/tests` holds the tests through the real encoders (`test_round_trip`, `test_rig`) and what
the recording tests share (`recordings.py`).

- The devices sit at the root, not under a `backends` folder: `CLAUDE.md` names `modules/camera/oak/` and
  `modules/camera/oak4/`.
- Recording sits inside `modules/camera`, not in a module of its own: the replay camera and the oak simulator use
  the player, so the two modules would import each other.
- `FrameTrigger` matches nothing and hands no frames on: it fires one callback per round of the cameras' frames,
  when the round's last camera delivers, for hd_trio's and white_space's tracker and poses.

## Measured

| What                                                    | Value                        | On                                              |
|---------------------------------------------------------|------------------------------|-------------------------------------------------|
| tracklets after their frame's capture, live             | about 33 ms **(site fact)**  | P800, oak4, 50 fps                              |
| `video_latency`, median                                 | 29.7 ms                      | `20261112-124918_hands`                         |
| the host, from a frame's arrival to its poses, in -sim  | median 34 ms, p95 65 ms      | the same, pose model on                         |
| parsing a chunk's metadata at once                      | 8.3 ms                       | 597 lines with poses                            |
| a player's poses in the metadata                        | about 400 bytes a frame      | five decimals **(deduction)**                   |
| the detector in -sim                                    | 1 ms a frame on the GPU      | YOLOv8n                                         |

By `scripts/pose_recording_probe.py`, on `20261112-124918_hands` (P800, one player throughout), 12 s each:

| Run                                            | Frames | With poses           | Pose latency median / p95 / max (ms) |
|------------------------------------------------|--------|----------------------|--------------------------------------|
| played with the pose model on, recorded again  | 594    | 488 (82%)            | 64.5 / 97.5 / 111                    |
| that recording played with its recorded poses  | 463    | 463, all as recorded | 72.3 / 105 / 114                     |

The frames without poses are batches the pose model dropped ([Open 3](#3-the-poses-the-pose-model-drops)), which
the replay drops alike. In the replay every pose had its recorded box and points, and the poses came about 8 ms
later after their capture than recorded: -sim's own delay on top of the recorded one. The recorded latencies hold
the live host's own delay too (the callback waits for Python's lock, up to 5 ms under a busy render)
**(deduction)**.

## Open

### 1. Checks waiting for the camera

- A new recording with a player in view: its counts on the panel, its tracklets and latencies in the metadata.
- -sim on it with the recorded tracklets against live on the same scene, by `scripts/cadence_probe.py`, and against
  -sim with the detector. Whether the live 99.4% of [INPUT_THREADS.md](../../flow/INPUT_THREADS.md#the-cadence) had a
  player in view is not known, likely yes **(site fact)**, so the live number is taken anew.
- The pose points and boxes over the image, live and in a replay, to see the tracklets straightened with their own
  frame and the board's poses as live's.
- The device's model looks at the frame unstraightened, which for a few degrees of tilt finds the same players
  **(deduction)**.

### 2. The other apps' recordings

1. The oak backend (depthai 2) stamps a frame's arrival, not its capture, so hd_trio and white_space record and
   replay with USB's jitter in their times. Its capture time as oak4's `capture_time` makes it, with the apps'
   move ([PLAN.md's line 4](../../flow/PLAN.md#4-the-other-apps-on-deep_flows-input)).
2. hd_trio and white_space wire the camera's tracklets to their tracker directly, not through the uploader, and do
   not say `Recorder.expect_poses`: they pose in rounds, and say it per camera and image when they move.

### 3. The poses the pose model drops

In -sim, with the recorded tracklets and the pose model on, about one frame in six with a player gets no poses (107
of 596 on `hands`): the pose model's TensorRT runner keeps one batch in flight and one waiting
(`pose/runner_trt.py`), and a batch that comes while one waits replaces it. At 50 fps a frame comes every 20 ms, so
the host's 34 ms keeps up only while batches overlap, and drops when one runs long. With the recorded tracklets
-sim loads the GPU as live does, its one addition, the decoding, on the iGPU and the CPU, so live drops about as
many **(deduction)**; no live number exists.

In order:

1. Live, with a player in view: a new recording's `missed_poses` and its frames with poses, the live drop rate.
2. In -sim on the same scene, the host's 34 ms split: RTMPose's own time (the predictor's inference timer), the
   wait for the crops' event behind the GPU's queue, and the wait in the runner's waiting slot, each per frame.
3. From that split, what stops the drops: RTMPose's own time over 20 ms means a faster model or input; a long wait
   for the crops means the pose work queued behind the fluid's (its own stream's priority, or the order of the work
   on the GPU); a long wait in the slot means the runner's hand-over. Each is weighed against the poses' precision
   (RTMPose stays FP32).

### 4. Small faults

- A camera's first frame of a recording is written without its tracklets and poses: nothing has said yet that the
  camera sends them. To fix: the recorder is told at its start which parts each camera sends.
- Two recordings started in one second get one folder name. To fix: the name takes a counter when the folder
  exists.
