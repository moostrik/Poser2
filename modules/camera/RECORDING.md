# Recording and playback

How a camera's frames are recorded and played back, so `-sim` runs an app on a recording without a device. The
modules' docstrings own the details: `recorder.py`, `chunk.py`, `player.py`, `replay/camera.py`,
`replay/tracker.py`. The threads a frame then goes through are deep_flow's
([INPUT_THREADS.md](../flow/INPUT_THREADS.md)).

"Code" is what the files show; **(deduction)** marks what is inferred from them.

## Recording

`Recorder` is a camera frame callback (`Recorder.submit_frame`) on the camera's thread, which only queues; its own
thread writes. It records what the camera delivered, not straightened, so the correction can still be tuned on a
recording; a frame with `flip_h` it mirrors on the CPU before encoding (`mirrored_image`).

A recording is a folder named after its start time (and the panel's `name`), split into chunks by the session. With
depth in `video_frame_types` each chunk ends as one `<cam>_<chunk>.mkv`, merged by stream copy when the chunk ends
(`chunk.merge_chunk`):

| Stream / part | Holds                                                                                   |
|---------------|-----------------------------------------------------------------------------------------|
| stream 0      | color, the camera's encoder (`video_encoder`, `video_format`)                           |
| stream 1      | depth, uint16 mm, `DEPTH_ENCODER`, the same `keyframe_interval` as the color            |
| tags          | the info line's values                                                                  |
| attachment    | the metadata, one JSON line each (`.jsonl`)                                             |

A color frame waits for its depth (`WAITING_VIDEO`, 8 frames), paired by timestamp within `CAPTURE_TOLERANCE`
(2 ms); a color frame whose depth never comes is left out, uncounted and unlogged, as is everything before the first
depth, so both streams count alike. Without depth a chunk is a color clip `<cam>_C_<chunk>.mp4` and has no metadata.

The metadata:

- the info line, first, what stays the same through a chunk (`ChunkInfo`): the frame's width and height, the lens
  (`intrinsics`), `flip_h` (here: the frames were mirrored before they were stored), `fps` and the depth's alignment
  as the camera gave it;
- a readings line per frame with readings: its number in the chunk, its `timestamp` (the capture time, live
  `perf_counter`) and `FrameReadings` (tilt and roll in degrees, exposure in µs, ISO).

A frame without readings writes no line, and `parse_metadata` takes the lines' order as the frames' numbers, so a
frame without readings would shift every reading after it onto the wrong frame **(deduction)**. oak4 gives every
color frame readings, so it does not happen live.

Not recorded: the camera's tracklets (boxes, ids, keypoints) and when the frames and tracklets arrived.

## Playback

`Player` plays a recording folder between the panel's in and out markers, every camera's chunks in step. Per chunk
and camera it reads the metadata and starts one `StreamReader` per stream (color, depth), each on its own thread,
decoding with ffmpeg and pacing itself on `perf_counter` at `PlayerSettings.fps` from its own start
(`StreamReader._play`). After a stall longer than a frame a reader starts its schedule anew. The player collects a
frame's color and depth by frame number and hands them to its frame callbacks, color first, with the frame's
readings; the cameras' readers drift apart by frame numbers (`Player.get_drift`), not by time.

So what playback paces on is the frame number at `fps`, not the recorded timestamps: a recording with uneven
capture (a frame lost live) plays even **(deduction)**.

The in and out markers are the recording's, in `in_out.json` in its folder, not in the chunks: a recording is many
chunk files, ffmpeg changes a file's tags only by writing the whole file anew, and `mkvpropedit`, which edits them in
place, is not installed.

## The replay camera

`replay.Camera` is a `CameraSource` without a device. It has no thread: its frames come on the player's reader
threads, its tracklets on the detector's.

- Frames: each color frame is stamped `perf_counter()` at hand-over, its depth takes the color's stamp. With
  readings and a lens, `TiltCorrection` straightens from the recorded tilt plus the live `tilt.offset`, so the
  frames carry a homography as live; the lens and the depth's alignment are mirrored, as the frames were stored.
- Tracklets: the recording has none, so the app gives the camera a host `PersonDetector`
  (`modules/inference/detection`, YOLOv8n on the uploaded, straightened and mirrored image) and a `replay.Tracker`
  (the device's tracker in software, SORT's shape). The detector's boxes go through the tracker and out as the
  contract's tracklets, stamped with the image's timestamp. Without a detector the camera is frames only.

deep_flow makes the detector when `cam_0.model_chain` is not `OFF`; the detector then runs on every uploaded image.
Live, the device's model looks at the frame unstraightened and unmirrored (stretched to its input, `ImageManip`),
and `oak4.Camera._tracklet` straightens and mirrors its boxes and keypoints on the host to match the frames the GPU
straightens, with the homography current when they arrive: that of a later frame than the one they were found in
when the tilt moved past `TILT_STEP` between the two **(deduction)**.

In -sim the tracklets therefore differ from live in three ways: another model (YOLO on the host, not the device's),
other timing (the detector's thread, not the device's latency) and other GPU load (the detector's 1 ms a frame).

## Open

### 1. Plan B: the recorded tracklets

Agreed 2026-11-11: the camera's tracklets are recorded with the frames, and -sim plays them back after their
recorded latency, so -sim gets live's tracklets, their timing and live's GPU load, without the host detector.

Whether the live 99.4% (captures the motion measures, [INPUT_THREADS.md](../flow/INPUT_THREADS.md#the-cadence)) had a
player in view is not known, likely yes **(site fact)**; with none in view live ran no poses, so the comparison
below takes a new live run with a player in view.

Built as six commits, in this order:

1. the contract: `CameraTracklets` and `arrival`, in every backend, both trackers, the board and the apps;
2. the straightening and the mirror of the tracklets off the camera, into the uploader;
3. the match, and `CaptureSync` on it;
4. the recorder: `RecordingSync`, its counts, the frames as delivered, one chunk for every recording, the metadata
   with the tracklets;
5. the player on one clock;
6. the replay camera's recorded tracklets and `PersonDetector.enabled`.

**What a recording holds**

Nothing straightened and nothing mirrored: the frames and the tracklets as the camera delivered them, and what a
playback needs to do with them what live does. A recording stores only what its playback plays.

Every recording is the same chunk, with depth or color only (`video_frame_types`, color only an option as now):
`<cam>_<chunk>.mkv`, the color stream, the depth stream when it is recorded, the info line in the tags and the
metadata attached.

| Where      | What                                                                                    | Now      |
|------------|-----------------------------------------------------------------------------------------|----------|
| info line  | width, height, the lens (`intrinsics`), `fps`, the depth's alignment (null without)     | kept     |
| info line  | `flip_h`: the frames are still to be mirrored, as in `CameraFrame`                      | changed  |
| per frame  | its number, `timestamp` (the capture)                                                   | kept     |
| per frame  | tilt and roll (°), exposure (µs), ISO (`FrameReadings`)                                 | kept     |
| per frame  | `video_latency` (s): the frame's arrival minus its capture                              | new      |
| per frame  | the tracklets: `id`, `status`, `roi`, `age`, `keypoints` (null without a pose model)    | new      |
| per frame  | `tracklet_latency` (s): their arrival minus their frame's capture                       | new      |

Not recorded: the homography and the tilt offset (a playback makes its own, from the tilt and the live offset), the
model chain (the keypoints say whether a pose model ran), the camera's settings, the depth's latency (the player
hands a frame's depth with its color). Recordings made before are not converted: their `flip_h` meant "mirrored
before storing", and they play mirrored wrong.

**The straightening and the mirror, off the camera**

The camera straightens and mirrors nothing. It hands its frames and its tracklets as delivered, and the uploader,
which straightens and mirrors the image, does the tracklets, with the homography and `flip_h` of the frame they
were found in. The device's model keeps looking at the frame unstraightened, which for a few degrees of tilt finds
the same players **(deduction)**. The camera cannot straighten its color frame itself: the oak4's `Warp` takes grey
frames only, and the depth is aligned on the GPU, where it is straightened with the color.

- `CameraTracklets`, the contract's tracklets of one frame, as the device gave them: `cam_id`, `timestamp` (their
  frame's capture), `arrival` and the `CameraTracklet`s; `TrackletCallback` takes one.
- `CameraFrame` and `CameraTracklets` carry `arrival`, `perf_counter` stamped by the backend at hand-over beside
  `timestamp`, so nothing downstream reads a clock for a latency.
- oak4's `_tracklet` and `_poses_callback` neither straighten nor mirror.
- `Uploader.submit_tracklets`, a tracklet callback beside `submit_frame`: it only queues. `submit_frame` notes each
  color frame's homography and `flip_h` by capture time, for the last few frames, a frame its queue drops included.
  The uploader's thread straightens and mirrors the tracklets with those of their frame
  (`geometry.straightened_tracklets`: `corrected_roi`, `corrected_keypoints`, the mirror) and hands them to its
  tracklet callbacks: the tracker and the board. Tracklets whose frame is no longer noted take the newest frame's.
- The camera's tracklet thread then only queues, into the uploader and the recorder.
- The host detector in -sim looks at the uploaded image, straightened and mirrored, so the replay camera's tracklets
  from it go to the tracker and the board as they are, not through the uploader.

Live on the P800 the tracklets come about as often as the frames, about 33 ms after their frame's capture
**(site fact)**, so the uploader keeps the homography of more frames than that spans.

**Two syncs on one match**

`CaptureSync` (`modules/inference`) and the recorder both match the parts of one capture by capture time, for
opposite ends. `CaptureSync` is live and puts latency first: the newest capture wins, a part is waited for briefly
and only while its stream keeps up, a late part is left out. The recorder puts completeness first: every capture
waits for its parts, in order, as long as the slowest takes, and a newer capture never ends the wait; the device's
tracklets arrive about 33 ms after their frame's capture, more than a frame at 50 fps **(site fact)**, which
`CaptureSync`'s rules would leave out of nearly every capture. So two syncs, on what they share:

- The match, in `modules/camera` beside `CAPTURE_TOLERANCE`: `Stamped`, the protocol of what it matches (`cam_id`,
  `timestamp`; `CameraFrame`, `CameraTracklets`, `GpuImage`, `GpuDepth` fit it); per camera a few of each part by
  capture time, found within `CAPTURE_TOLERANCE`; and the counts of what goes missing per camera and part, with a
  warning naming them at most once a second while they rise. Its name is not decided.
- `CaptureSync` on the match, its rules as they are. It is the input thread's; the change is made with it.
- `RecordingSync` (name not decided), in `modules/camera`, on the match: color required, depth required when
  recorded, the tracklets waited for when the app wires them (`RecorderSettings.tracklet_wait`, 100 ms, three times
  their latency); several captures waiting at once, each out when its parts are in or its wait ends, in order of
  capture; its own thread for the waits, no submit blocks.
- Counted: a color frame without its depth, a depth without its color, a capture written without its tracklets.
  The recorder's panel shows the counts beside `recording`. What comes before a recording's first complete capture
  is counted apart and not warned of: a few frames at every start.

**The recording side**

- `Recorder.submit_frame` and `Recorder.submit_tracklets` (a tracklet callback) only hand their parts to the
  `RecordingSync`, in place of `waiting_video` and `WAITING_VIDEO`; a capture it sends out goes on the recorder's
  queue.
- Frames stored as delivered: no `mirrored_image`, `flip_h` in the info line as the frames carry it.
- A color-only recording is merged as one `.mkv` with its metadata too (`merge_chunk` with the color stream alone).
- One metadata line per frame, always: readings null when the frame has none, so a line's place is its frame's
  number; `video_latency` and `tracklet_latency` from the parts' `arrival`; null tracklets for a capture written
  without them, an empty list for a frame with no player.
- `chunk.parse_metadata` gives one `FrameMetadata` per frame (timestamp, readings, the latencies, the tracklets) in
  place of the readings list.
- Wired in deep_flow: the camera's tracklets to `Uploader.submit_tracklets` and `Recorder.submit_tracklets`, the
  uploader's to the tracker and the board. The contract's change is made in every backend, tracker and app in its
  commit; hd_trio and white_space are off on this branch
  ([INPUT_THREADS.md Plan 1](../flow/INPUT_THREADS.md#1-white_space-and-hd_trio-on-deep_flows-input)) and follow with
  them.

**The replay side**

- `Player` on one clock, the recorded time: at the start of a play (a loop, a seek) it sets an origin, and every
  reader of every camera hands frame `n`, color and depth together as now, at origin plus (its recorded `timestamp`
  minus the first frame's) plus its `video_latency`, in place of each reader's own schedule at `fps`. `StreamReader`
  knows no metadata: the player gives each reader its schedule, a function from the frame's number to its moment,
  never earlier than the frame before it. A reader's stall moves the origin for all of them, so the cameras and the
  streams stay together.
- The recorded latencies hold the live host's own delay (the callback waits for Python's lock, up to 5 ms under a
  busy render), and -sim adds its own, so a frame arrives a little later after its stamp in -sim than live.
- A frame's stamp is its capture moment on the play clock, origin plus its recorded time: `ClipFrameCallback`
  passes it, and `replay.Camera` stamps with it, so a frame arrives as late after its stamp as it did live.
- `replay.Camera` hands its frames with the recording's `flip_h`, and makes their homography from the recorded tilt
  and the live offset on the recorded lens, as oak4 does live; `mirrored_intrinsics` and `mirrored_alignment` leave
  it. The homography matches live's to within `TILT_STEP` (0.25°): live makes it from the IMU's newest tilt, the
  recording keeps the tilt read with the frame **(deduction)**.
- `replay.Camera` hands a frame's recorded tracklets as `CameraTracklets` with that frame's stamp, as the device gave
  them, so the uploader straightens and mirrors them as it does live; at the frame's stamp plus their
  `tracklet_latency`, from a timer thread of its own (nothing on the reader's thread but the hand-over), and none for
  a frame with null tracklets. A loop or a seek empties what waits there, so no tracklets of before the jump come
  after it.
- `PersonDetector.enabled` (settings `person_detector`): on, the camera takes the detector's tracklets, as now; off,
  the detector looks at nothing and the camera sends the recorded ones. deep_flow's preset turns it off.
- The presets take the two new settings, `recorder.tracklet_wait` and `person_detector.enabled`.

Tests: the match (parts found by capture time, the counts per part); `CaptureSync`'s tests passing on it;
`RecordingSync` (several captures waiting, out in order, a capture out when its tracklets come or its wait ends, a
newer capture not ending a wait, color without depth counted, the start counted apart); `straightened_tracklets`
against `corrected_roi` and the mirror; the uploader's tracklets (straightened with their own frame's homography
after it changed, a dropped frame's still known); a metadata line with tracklets and latencies in and out; the
recorder (frames stored unmirrored, a color-only chunk with its metadata, the chunk's end writing what waits); the
player's schedule (recorded times and latencies, never out of order, a stall moving the origin); the replay
camera's tracklets after their latency, with their frame's stamp, emptied by a loop.

Measured after the replay side, by `scripts/cadence_probe.py`, on a new recording with a player in view: -sim with
the recorded tracklets against live on the same scene (captures the motion measures, the tick's p95, the capture
latency), and against -sim with the detector.
