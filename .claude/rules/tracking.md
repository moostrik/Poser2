---
description: Host trackers on top of the camera's ObjectTracker
paths:
  - "modules/tracker/**"
---
# Tracking Guidelines

## Two tiers

- Each camera runs depthai `YoloDetectionNetwork` → `ObjectTracker` (`modules/camera/oak/pipeline.py`, `TRACKER_TYPE` in `definitions.py`); read them before reasoning about identity
- The camera's `ObjectTracker` owns detection, per-frame association, short-term LOST bridging under the same id, and device ids; it may hand one person two tracks (NMS double boxes, id churn)
- A host tracker (`modules/tracker/`) works on those device tracks; it adds what one camera cannot know, plus the same-camera identity the device fails to settle

## Rules

- Do not add motion models or per-frame assignment to a host tracker
- The camera pipeline stays fixed; treat a within-camera identity problem with the host's same-camera gates (`same_cam_angle`, `reacquire_angle`)
- A device id is valid for one device track only: after `REMOVED` its next holder is a new person
- Break ties by an id handed out in order, never by a timestamp: `time.time()` has ~16 ms resolution on Windows
- In tests, send LOST and REMOVED as the device would
