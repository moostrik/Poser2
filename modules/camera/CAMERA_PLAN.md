# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F; again 2026-11-21 and 2026-11-22). Who uses
what, from the code: the contract (`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`,
the capture match) is read by every app and by inference, recording, tracker and render. The backends are each one
kind of camera, used now by the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, low_flow
and smile, `vimba` and `webcam` by smile. That is general code with one user, which stays.

**(seen)** is read in the code, **(deduction)** inferred.

## Open

- Where the OAK-D's 1.1% clip loss is made. The recording thread counted about 4,104 frames sent per OAK-D and 4,060
  back over 180 s of white_space -sim (recording 20261124-110331_probe_F1, about three loops) **(site fact)**, and
  guessed the frames in flight at each loop jump are lost. Between the clip input's send and the camera's video
  callback nothing is lost, at the loop jumps included: over 181 s, three loops, `studio.json` with `replay` false and
  `CLIP_QUEUE_SIZE` 1, each camera's frames sent equal its frames back (4,159, 4,160, 4,160, 4,156), `dropped_clips`
  is 0, every 4 s window around a jump has as many back as sent within the one in flight, video latency 35 ms and
  tracker latency 119 ms (2026-11-24) **(seen)**. The loss is counted after the camera's hand-over or before the
  send; where the recording thread counted, and against what, is to be asked before anything is changed. The
  simulator's `playhead` steps back about 9.4 s every 10 s of play, outside the loops **(seen, cause not checked)**.
- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.
- Flow Motion's A/B of the depth's confidence needs a recording with it and a player moving. No recording with a
  confidence is kept: the two probes of 2026-11-21 were made to measure the encoder and are deleted.
- How to count the drops inside depthai, of both OAKs. The detector's input is non-blocking
  (`setBlocking(False)` in both `pipeline.py`) and skips the frames it has no time for. The OAK4's depth `Sync`
  (`add_depth`) lets go of a depth or confidence without its partner within `CAPTURE_TOLERANCE` **(deduction)**.
  depthai reports no count of either **(deduction)**. The panel shows them only as a rate: `tracker_fps`, and the
  depth rate against the video's **(seen)**. A gap in a stream's sequence numbers at the hand-over would count them,
  if ImageManip and Warp carry the numbers through, which is not checked. The OAK-D's clip input counts its own
  (`dropped_clips`). The host's output queues (`maxSize=1`, non-blocking) lose nothing: their
  callback takes every message, and what the queue overflows has already gone to the callback **(deduction)**.
- Feeding the OAK4 a recording, when an OAK4 app needs it; no OAK4 app is fed one, and depth is not fed back
  (Matthias, 2026-11-23 and 2026-11-24). One input node, its queue from `createInputQueue()`, as the frames source in
  place of `add_camera`'s (OAK4.md, The backend); `oak4.Camera(settings, cam_id, clips: ClipSource | None = None)`
  as the OAK-D's camera with clips, with no Neural Depth, no sensor controls and no IMU; the fed frame stamped with its recorded capture, as the OAK-D's. The Player is
  built with `nv12=True` in the OAK4 apps **(seen)**, so a clip frame needs no conversion, only its strides and planes.
  First a probe with the OAK4 attached, on Matthias's go: the crops and their keypoints land in the frame (an
  `ImgTransformation` made from the recorded lens and size, through `FrameCropper` and `remapPointTo`), a stamp set on
  the host frame comes back through `capture_time`, and 720p NV12 sent at 60 fps holds the rate **(deduction, not
  run)**. deep_flow and low_flow have no `replay` switch; their -sim always runs `replay.Camera` **(seen)**.

## Rules it touches