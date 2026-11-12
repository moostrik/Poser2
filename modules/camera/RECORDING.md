# Recording and playback

How a camera's frames, its tracklets and the host's poses are recorded and played back, so `-sim` runs an app on a
recording without a device, with the camera's own tracklets and live's poses, at their timing. The modules'
docstrings own the details: `recorder.py`, `recording_sync.py`, `capture_match.py`, `chunk.py`, `player.py`,
`replay/camera.py`, `replay/tracker.py`, `modules/inference/recorded_poses.py`. The threads a frame then goes through are deep_flow's ([INPUT_THREADS.md](../flow/INPUT_THREADS.md)).

"Code" is what the files show; **(deduction)** marks what is inferred from them, **(site fact)** what was told.

## What the camera hands over

A camera hands its frames (`CameraFrame`) and its tracklets of one frame (`CameraTracklets`) as it delivered them:
nothing straightened, nothing mirrored. Each carries its capture time (`timestamp`) and the moment the backend
handed it over (`arrival`), on `perf_counter`; their difference is its latency, so nothing downstream reads a clock
for it. A frame carries the homography it is still to be straightened with and `flip_h`, the mirror still to do.

The camera cannot straighten its color frame itself: the oak4's `Warp` takes grey frames only, and the depth is
aligned on the GPU, where it is straightened with the color. So the uploader straightens and mirrors both, and the
tracklets with them: `Uploader.submit_tracklets` only queues, and the uploader's thread moves the tracklets with the
homography and mirror of the frame they were found in (noted by capture time in `submit_frame`, a dropped frame's
included; the newest frame's when it is no longer noted, `straightened_tracklets`), ahead of the frames waiting,
and hands them to the tracker and the board. The device's model looks at the frame unstraightened, which for a few
degrees of tilt finds the same players **(deduction)**. Live on the P800 the tracklets come about as often as the
frames, about 33 ms after their frame's capture **(site fact)**.

Tracklets found on the uploaded image are in its frame already (`straightened`): the host detector's in -sim.
hd_trio and white_space still wire the camera's tracklets to their tracker directly; they are off on this branch and
take the uploader's with [PLAN.md's line 4](../flow/PLAN.md#4-the-other-apps-on-deep_flows-input).

## Recording

deep_flow records live and in -sim alike: a recording played is recorded again as the replay camera hands it on (its
frames with their readings and lens, its recorded tracklets; the host detector's tracklets are not a camera's and
stay out), with the poses the app finds in it, so a recording without poses gets them by being played once with the
pose model on.

`Recorder` takes the camera's frames (`submit_frame`) and tracklets (`submit_tracklets`) on the camera's threads and
hands them to its `RecordingSync`, which brings the parts of one capture together, completeness first: the color
frame, its depth when DEPTH is in `video_frame_types`, and the tracklets of a camera that sends them, matched by
capture time within `CAPTURE_TOLERANCE` (2 ms, `CaptureStream`). Each camera is synced apart, under its own lock, and
its captures go out in its own order; the order between cameras is not kept, as it is not live. A camera that sent
no depth by its `KEPT`-th color frame records color only, so a camera without depth records beside one with it. Every complete capture waits for its tracklets, in
the order of capture, and a newer capture never ends the wait; a camera that never sent tracklets is not waited
for. No wait is timed: each stream comes in the order of capture, so a part of a later frame says this frame's will
not come, and the capture goes out without it. At most `KEPT` captures (16, 320 ms at 50 fps, three times the
slowest pose measured) wait per camera, beyond which the oldest goes out without what it waits for: the one case
order cannot settle, a stream that stopped. On stop no frame is taken any more, and the captures still waiting are
settled by the tracklets and poses of the frames after the stop, which keep coming, for at most `STOP_WAIT` (1 s,
run out only by a stream that stopped with the recording). The recorder's thread writes what the sync sends out.

The host's poses come too: the RAW stage's output, each pose's crop as its box (`BBox`) and the pose model's points
in it with their scores (`Points2D`), stamped with the image's capture by the crops (`CropExtractor.process`). The
box goes with the points, so a replay does not depend on the box smoother and predictor reproducing live's box: the
predictor reads time, and a replay's stamps are not live's arrivals. The app hands them over (`Recorder.submit_poses`
through `modules.inference.frame_poses`, the camera contract knowing no pose), stamping their arrival there, and
says for every image whether the pose model was asked poses for it (`Recorder.expect_poses`, from what
`PosesFromTracklets.process` returns). A capture waits to know, and for the poses asked for, settled by the order
as the tracklets are; a frame nobody asked poses for does not wait.

A waiting capture keeps its parts as they come, so newer ones cannot push them out of their stream. What a capture
went without is counted on the panel (`missed_depth`, `missed_color`, `missed_tracklets`, `missed_poses`: asked
for, a batch the model dropped) and warned of by the recorder's thread, at most once a second (`MissedParts`
counts where the parts come, on a camera's thread under the sync's lock, and its owner logs); what came before a
camera's first capture of a recording, a few frames at every start, is counted apart and not warned of. The live `CaptureSync`
(`modules/inference`) shares the matching and the counting and puts latency first instead: its newest capture wins
and a late part is left out.

The recorder sits behind the live path: deep_flow hands a camera's frames and tracklets to the uploader first. What
waits for the encoders, in the recorder's queues and in its writers (`StreamWriter.queued_bytes`), is bounded by one
count of bytes for all cameras (`HELD`, 1 GB); beyond it a capture is
dropped whole before any part of it is written, and counted (`dropped`), so a chunk's color, depth and metadata
always count alike. A writer drops nothing of its own. An encoder that dies (an error, a full disk) ends its writer, which drops what it
holds and says so, and the recording ends; a recording that cannot start says why once and is given up. The queue
is emptied at every start, so nothing of one recording reaches the next, and what a crash left is merged on a
thread of its own beside the new recording.

A recording is a folder named after its start time (and the panel's `name`), split into chunks by the session. A
split cuts every camera at one capture time, the newest color frame any camera handed in when it was asked for: each
camera's captures up to it end its chunk, the later ones open the next, so the cameras' chunks start and end at one
recorded moment; a camera with no capture after the cut within `SPLIT_WAIT` (1 s) is cut anyway. Each chunk ends as
one `<cam>_<chunk>.mkv`, with depth or color only, merged by stream copy when the chunk ends
(`chunk.merge_chunk`):

| Stream / part | Holds                                                                                   |
|---------------|-----------------------------------------------------------------------------------------|
| stream 0      | color, the camera's encoder (`video_encoder`, `video_format`)                           |
| stream 1      | depth when recorded, uint16 mm, `DEPTH_ENCODER`, the same `keyframe_interval`           |
| tags          | the info line's values                                                                  |
| attachment    | the metadata, one JSON line each (`.jsonl`)                                             |

A chunk holds nothing straightened and nothing mirrored, and only what its playback plays:

| Where      | What                                                                                    |
|------------|-----------------------------------------------------------------------------------------|
| info line  | width, height, the lens (`intrinsics`), `fps`, the depth's alignment, `depth`           |
| info line  | `flip_h`: the frames are still to be mirrored, as in `CameraFrame`                      |
| per frame  | its number, `timestamp` (the capture)                                                   |
| per frame  | tilt and roll (°), exposure (µs), ISO (`FrameReadings`), null without                   |
| per frame  | `video_latency` (s): the frame's arrival minus its capture                              |
| per frame  | the tracklets: `id`, `status`, `roi`, `age`, `keypoints` (null without a pose model)    |
| per frame  | `tracklet_latency` (s): their arrival minus their frame's capture                       |
| per frame  | `poses`: per `track_id`, the box (x, y, width, height) and the points (17 × x, y, score) |
| per frame  | `pose_latency` (s): the poses' arrival minus the frame's capture                        |

One line per frame, always, so a line's place is its frame's number. The tracklets are null for a frame none came
for and an empty list for a frame with no player; the poses are null for a frame the model gave none (no player,
or a batch it dropped), so a replay drops the same frames live did. A player's poses take about 400 bytes a frame,
at five decimals **(deduction)**. Not recorded: the homography and the tilt offset (a playback
makes its own, from the tilt and the live offset), the model chain (the keypoints say whether a pose model ran),
the camera's settings, the depth's latency (the player hands a frame's depth with its color). Recordings made
before this layout play wrong: their `flip_h` meant "mirrored before storing", and their readings are not read.

## Playback

`Player` plays a recording folder between the panel's in and out markers, every camera's chunks in step, on one
clock, the recorded time (`PlayClock`): at the start of a play it sets an origin, and every reader of every camera
hands frame `n`, its color and, where its chunk holds it, its depth together, at the origin plus its recorded
capture time minus the cameras' earliest first frame's plus its `video_latency`, so a frame comes as late after its capture as it came live and a frame lost live leaves
its gap. The frame is stamped with its capture on that clock, its streams with one stamp. The chunk after the one
playing carries the clock on (`carry_on`) and shares its origin, so the recorded time runs on across chunks; it
starts when the first reader of the playing chunk has played its last frame, beside the rest, so a camera whose
chunk ended first goes on without waiting for the others to end. A loop, a seek or another
folder starts it anew, and the player says so (`add_jump_callback`): the replay camera and `RecordedPoses` drop what
waits for the frames before. Each stream has its own `StreamReader`, on its own thread, which takes its moments from
the clock (`FrameSchedule`), never earlier than the frame before; a reader that falls behind by more than a frame
moves the origin for all of them, as far as its frame is still behind, so one stall moves it once. Each camera's
frames go out under its own lock, in the order of chunk and frame, with the streams its chunk has, and a frame older
than one gone out is dropped, counted with the frames left
without their other stream and logged by the player's thread, which also writes the panel's readouts. How far the
cameras play apart is `drift`: the recorded capture times of their last frames gone out, in ms.

The markers, a seek and a scrub are places on camera 0's frames, and every other camera plays the same recorded
moment: it starts at its first frame captured at or after camera 0's first frame of the range, and ends after its
last at or before camera 0's last (`ChunkReaders`), so a camera that dropped unlike frames live starts and ends
where camera 0 does. A stop ends a
reader's decoder, so a decoder that stalled cannot hold the player. A clip without metadata plays at `fps`.

The player is composed of parts, each in its own file of `modules/camera`:

| File                   | Class                         | Holds                                                               |
|------------------------|-------------------------------|---------------------------------------------------------------------|
| `player.py`            | `Player`                      | the thread, its states and messages, the settings' callbacks, the readouts |
| `play_clock.py`        | `PlayClock`                   | the recorded time of a play                                         |
| `recording_folder.py`  | `RecordingFolder`, `Timeline` | the folders found, a folder's chunks, frame counts and markers' file; the markers and places on the timeline |
| `chunk_readers.py`     | `ChunkReaders`                | one chunk's readers of every camera and stream, its metadata and its clock |
| `frame_pairing.py`     | `FramePairing`                | each camera's frames handed over in order under its own lock with one stamp, the chunks handed over, the unpaired count, the drift |

deep_flow's player decodes color to NV12 (`Player(nv12=True)`), as the camera hands it: half the bytes of BGR
through the pipe, and live's upload path. A chunk's metadata is read once per chunk and kept for the chunk playing,
the one loaded and the last played past a stop, so a seek in it reads none; its lines are parsed as the frames are read (`FrameLines`: 8.3 ms for 597 lines with poses at
once, a minute's chunk about 40 ms), so no load holds the interpreter for it.

The recorded latencies hold the live host's own delay (the callback waits for Python's lock, up to 5 ms under a
busy render), and -sim adds its own, so a frame arrives a little later after its stamp in -sim than live
**(deduction)**.

The in and out markers are the recording's, in `in_out.json` in its folder, not in the chunks: a recording is many
chunk files, ffmpeg changes a file's tags only by writing the whole file anew, and `mkvpropedit`, which edits them in
place, is not installed.

## The replay camera

`replay.Camera` is a `CameraSource` without a device. Its frames come on the player's reader threads, as the camera
delivered them: `flip_h` as recorded, depth frames with the recorded alignment, and with `tilt` the homography of
the recorded tilt plus the live offset on the recorded lens, which matches live's to within `TILT_STEP` (0.25°):
live makes it from the IMU's newest tilt, the recording keeps the tilt read with the frame **(deduction)**.

Its tracklets come one of two ways:

- The recorded ones, while the person detector is off (`PersonDetector.enabled`, off in deep_flow's preset): each
  frame's at its stamp plus their `tracklet_latency`, stamped with the frame's, as the camera gave them, so the
  uploader straightens them as it does live. They go out from the camera's own thread, so nothing but the hand-over
  runs on the player's. A jump in the recorded time (a loop, a seek) drops what still waits.
- The host detector's, while it is on: YOLOv8n on the uploaded image (`modules/inference/detection`), its boxes
  through `replay.Tracker` (the device's tracker in software, SORT's shape), out as tracklets marked `straightened`.
  They differ from live's in their model, their timing (the detector's thread) and the GPU load (1 ms a frame).

## The recorded poses

`RecordedPoses` (`modules/inference`) is the RAW stage's async step in -sim, around `PoseInference`. It hears the
player's frames (`set_clip_frame`, a `ClipFrameCallback`) and keeps each frame's recorded poses by its capture on
the play's clock, the stamp the crops give the stage's poses. With `recorded_poses.enabled` on (deep_flow's default),
it cuts the crops for what reads them (`PoseInference.cut_crops`: the optical flow on the crops, hd_trio's masks,
the crop previews) without the pose model, sets each pose's box and points from the recording by `track_id`, and
hands them on after the recorded `pose_latency`, from its own thread. A frame the recording has no poses for sends
none; a pose whose `track_id` it has not goes on without points, counted and warned of. The `track_id`s come from
the host tracker on the recorded tracklets, the same tracker on the same messages as live, so they match the
recording's **(deduction)**. With it off the pose model runs as live, with live's GPU load, its poses differing a
little every play; it is apart from `pose.enabled`, which the preset shares between live and -sim. A jump in the
recorded time (a loop, a seek) drops what waits. A recording without poses plays boxes without points while it is
on, as one without tracklets plays no players with the detector off.

The pose recorder of `modules/pose` (`Recorder`, HDF5 per stage, hd_trio's session and intro sequence) is another
thing: it records a stage's features for the show to play, on the wall clock, apart from the video.

## Open

### 1. Checks of the recorded tracklets

Built 2026-11-12 in six commits: the contract (`CameraTracklets`, `arrival`), the straightening of the tracklets
into the uploader, the shared matching under `CaptureSync`, the recorder (`RecordingSync`, the counts, the frames
as delivered, one chunk for every recording, the metadata with the tracklets), the player on one clock, the replay
camera's recorded tracklets with `PersonDetector.enabled`. Tested without the camera: the units above, and a
recording through the real encoders and the player, its frames played on the recorded time.

Waiting for the camera:

- a new recording with a player in view, its counts on the panel, its tracklets and latencies in the metadata;
- -sim on it with the recorded tracklets against live on the same scene, by `scripts/cadence_probe.py` (captures
  the motion measures, the tick's p95, the capture latency), and against -sim with the detector. Whether the live
  99.4% of [INPUT_THREADS.md](../flow/INPUT_THREADS.md#the-cadence) had a player in view is not known, likely yes
  **(site fact)**, so the live number is taken anew;
- the pose points and boxes over the image, live and in -sim, to see the tracklets straightened with their own
  frame.

### 2. Checks of the recorded poses

Built 2026-11-12 in one commit, tested without the camera: the chunk's poses in and out, the recorder waiting for
the poses asked for and not for a frame nobody asked, the poses asked for that never came counted, `RecordedPoses`
playing the recorded box and points by `track_id` after their latency, a missing `track_id`, a jump, and `enabled`
off leaving the pose model's poses.

Measured by `scripts/pose_recording_probe.py`, on `20261112-124918_hands` (P800, one player throughout), 12 s each:

| Run                                            | Frames | With poses           | Pose latency median / p95 / max (ms) |
|------------------------------------------------|--------|----------------------|--------------------------------------|
| played with the pose model on, recorded again  | 594    | 488 (82%)            | 64.5 / 97.5 / 111                    |
| that recording played with its recorded poses  | 463    | 463, all as recorded | 72.3 / 105 / 114                     |

The 106 frames without poses are batches the pose model dropped under -sim's load **(deduction)**, which the
replay drops alike. In the replay every RAW pose had its recorded box and points, no `track_id` was missed, and the
poses came about 8 ms later after their capture than recorded: -sim's own delay on top of the recorded one. The
first run waited for the poses up to a timed 150 ms, which the order of the streams has replaced since.

Waiting for the camera: the board's poses over the image in a replay against live.

### 3. The other apps' recordings

1. The oak backend (depthai 2) stamps a frame's arrival, not its capture, so hd_trio and white_space record and
   replay with USB's jitter in their times. Its capture time as oak4's `capture_time` makes it, with the apps'
   move ([PLAN.md's line 4](../flow/PLAN.md#4-the-other-apps-on-deep_flows-input)).
2. `Recorder.expect_poses` is wired in deep_flow; hd_trio and white_space pose in rounds, and say it per camera and
   image when they move.

### 4. The poses the pose model drops

In -sim, with the recorded tracklets and the pose model on, about one frame in six with a player gets no poses (107
of 596 on `hands`): the pose model's TensorRT runner keeps one batch in flight and one waiting
(`pose/runner_trt.py`), and a batch that comes while one waits replaces it, dropped. The poses' latency in -sim
splits into the camera's own (`video_latency`, median 29.7 ms, recorded live and replayed) and the host's, from the
frame's arrival to its poses (median 34 ms, p95 65 ms: the upload, the crop and RTMPose). At 50 fps a frame comes
every 20 ms, so a host step of 34 ms keeps up only while batches overlap, and drops when one runs long.

With the recorded tracklets -sim loads the GPU as live does: its one addition, decoding the recording, runs on the
iGPU and the CPU **(deduction)**. So live drops about as many **(deduction)**; no live number exists yet.

The plan, in order:

1. Live, with a player in view: a new recording's `missed_poses` and its frames with poses, the live drop rate.
2. In -sim on the same scene, the host's 34 ms split: RTMPose's own time (the predictor's inference timer), the
   wait for the crops' event behind the GPU's queue, and the wait in the runner's waiting slot, each per frame.
3. From that split, what stops the drops: RTMPose's own time over 20 ms means a faster model or input; a long wait
   for the crops means the pose work queued behind the fluid's (its own stream's priority, or the order of the work
   on the GPU); a long wait in the slot means the runner's hand-over. Each is weighed against the poses' precision
   (RTMPose stays FP32).

