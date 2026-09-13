---
description: Host trackers on top of the camera's ObjectTracker
paths:
  - "modules/tracker/**"
---
# Tracking Guidelines

## Two tiers

- Each camera runs depthai `YoloDetectionNetwork` → `ObjectTracker` (`modules/oak/camera/pipeline.py`, `TRACKER_TYPE` in `definitions.py`); read them before reasoning about identity
- The camera's `ObjectTracker` owns detection, per-frame association, short-term LOST bridging under the same id, de-duplication and device ids
- A host tracker (`modules/tracker/`) works on those device tracks; it adds only what one camera cannot know

## Rules

- Do not add motion models, per-frame assignment or de-duplication to a host tracker
- Treat a within-camera identity problem as a device-tracker setting first (`TRACKER_TYPE`, YOLO thresholds)
- A device id is valid for one device track only: after `REMOVED` its next holder is a new person
- Break ties by an id handed out in order, never by a timestamp: `time.time()` has ~16 ms resolution on Windows
- In tests, send LOST and REMOVED as the device would
