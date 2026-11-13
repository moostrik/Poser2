# Recording and playback

How a camera's frames, its tracklets and the host's poses are recorded and played back, so `-sim` runs an app on a
recording without a device, with the camera's own tracklets and live's poses, at their timing. This document holds
the design and its reasons; how each part works is in its module's docstring:

| Part                         | Module                                                               |
|------------------------------|----------------------------------------------------------------------|
| what a chunk holds           | `container/chunk.py`                                                 |
| a capture's parts together   | `recorder/recording_sync.py`, `modules/camera/sync/capture_match.py` |
| writing                      | `recorder/recorder.py`, `modules/video/stream_writer.py`             |
| playing                      | `player/player.py` and [its parts](#the-players-parts)               |
| the camera of a recording    | `replay/camera.py`, `replay/tracker.py`                              |
| the poses of a recording     | `modules/inference/recorded_poses.py`                                |

The threads a frame goes through after it are deep_flow's ([INPUT_THREADS.md](../flow/INPUT_THREADS.md)). What is
next is in [RECORDING_PLAN.md](RECORDING_PLAN.md).

**(deduction)** marks what is inferred from the code, **(site fact)** what was told; the rest is in the code.

## What is recorded

A recording holds what the camera delivered, nothing straightened and nothing mirrored, and only what its playback
plays. The camera cannot straighten its color frame itself (the oak4's `Warp` takes grey frames only, and the depth
is aligned on the GPU, where it is straightened with the color), so straightening and mirroring are the uploader's,
live and in -sim alike, and a recording made before them plays through the same path as the camera. The tracklets
are stored as the camera gave them for the same reason: the uploader straightens them with the homography of the
frame they were found in (`SerialUploader.submit_tracklets`).

Every part carries its capture time (`timestamp`) and the moment the backend handed it over (`arrival`), on
`perf_counter`. Their difference, the latency, is stored with the part, so a playback hands each part as late after
its capture as it came live.

A recording is a folder named after its start time (and the panel's `name`), with a counter (`-2`, `-3`) when a
recording of that second has the name already. Its chunks are in the recorder's `container`:

| `container` | Files per camera and chunk                        | Holds                                             |
|-------------|---------------------------------------------------|---------------------------------------------------|
| `MKV`       | `<cam>_<chunk>.mkv`                               | color, depth when recorded, the metadata attached |
| `MP4`       | `<cam>_C_<chunk>.mp4` and `<cam>_M_<chunk>.jsonl` | color, and the metadata beside it; no depth       |

An `MP4` chunk's clip plays in any player, h264 at a constant `fps`; a frame lost live is a frame missing from it, so
another player shows the rest a frame early **(deduction)**, while ours plays the recorded times from the `.jsonl`.
With depth asked for, `MP4` records color and warns once. A clip without its `.jsonl`, the layout of the older
recordings, plays at `fps`, its frames as stored (`flip_h` False). An `MKV` chunk holds:

| Stream / part | Holds                                                                                   |
|---------------|-----------------------------------------------------------------------------------------|
| stream 0      | color, the camera's encoder (`video_encoder`, `video_format`)                           |
| stream 1      | depth when recorded, uint16 mm, `DEPTH_ENCODER`, the same `keyframe_interval`           |
| tags          | the info line's values                                                                  |
| attachment    | the metadata, one JSON line each (`.jsonl`)                                             |

The metadata, attached or beside, is the same lines:

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
| A camera's first capture waits for the parts the app says it sends   | before its first tracklet nothing shows them coming (`Recorder.expect_parts`)         |
| A frame nobody asked poses for does not wait for them                | the app says per image whether the pose model was asked (`Recorder.expect_poses`)     |
| Each camera is synced in its own order                               | the order between cameras is not kept live either                                     |
| The parts are matched on the sync's own thread, from one inbox       | the threads they come on only queue them, a camera's among them                       |
| The sync's inbox is not bounded                                      | a recording needs every part; `HELD` bounds what waits after the match                |
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

## The package's layout

`modules/recording` uses a camera and is not one: it depends on `modules/camera`, never the other way. A camera
that feeds recorded clips through its device (`oak.ClipCamera`) takes them by the camera's `ClipSource` protocol.

| Folder      | Role                   | Holds                                                      |
|-------------|------------------------|------------------------------------------------------------|
| `container` | the file's format      | `chunk`, `recording_folder`                                |
| `recorder`  | writing                | `recorder`, `recording_sync`                               |
| `player`    | playing                | `player`, `play_clock`, `chunk_readers`, `frame_hand_over` |
| `replay`    | the recording's camera | `camera`, `tracker`                                        |

Each folder has its tests beside it; `tests` holds the tests through the real encoders (`test_round_trip`,
`test_rig`, `test_mp4`) and what the recording tests share (`recordings.py`).

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

The frames without poses are batches the pose model dropped
([RECORDING_PLAN.md](RECORDING_PLAN.md#3-the-poses-the-pose-model-drops)), which
the replay drops alike. In the replay every pose had its recorded box and points, and the poses came about 8 ms
later after their capture than recorded: -sim's own delay on top of the recorded one. The recorded latencies hold
the live host's own delay too (the callback waits for Python's lock, up to 5 ms under a busy render)
**(deduction)**.

