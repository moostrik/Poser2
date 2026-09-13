---
description: The two-tier people tracking stack — camera ObjectTracker and host panoramic tracker
paths:
  - "modules/tracker/**"
  - "modules/oak/camera/**"
---
# Tracking Guidelines

## Two tiers

- Each camera runs depthai `YoloDetectionNetwork` → `ObjectTracker` (`modules/oak/camera/pipeline.py`, `TRACKER_TYPE` in `definitions.py`); read them before reasoning about identity
- The camera's `ObjectTracker` owns detection, per-frame association, short-term LOST bridging under the same id, de-duplication and device ids
- The host tracker (`modules/tracker/panoramic/`) owns cross-camera identity only: seam links, world merges and splits, the emitted view, re-linking a REMOVED device track

## Rules

- Do not add motion models, per-frame assignment or de-duplication to the host tracker
- Treat a within-camera identity problem as a device-tracker setting first (`TRACKER_TYPE`, YOLO thresholds)
- A device id is valid for one device track only: after `REMOVED` its next holder is a new observation
- Show layers and the state machine read poses, never tracklets; presence is a pose existing, set by `pose.tracklets.detection_timeout`
