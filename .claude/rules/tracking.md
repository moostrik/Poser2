---
description: The two-tier people tracking stack — camera ObjectTracker and host panoramic tracker
paths:
  - "modules/tracker/**"
  - "modules/oak/camera/**"
---
# Tracking Guidelines

The design and its data are in `apps/white_space/data/TRACKING.md`.

## Two tiers

- Each camera runs depthai `YoloDetectionNetwork` → `ObjectTracker` (`modules/oak/camera/pipeline.py`, `TRACKER_TYPE` in `definitions.py`); read them before reasoning about identity
- The camera's `ObjectTracker` owns detection, per-frame association, short-term LOST bridging under the same id, de-duplication and device ids
- The host tracker (`modules/tracker/panoramic/`) owns cross-camera identity only: seam links, world merges and splits, the emitted view, re-linking a REMOVED device track

## Rules

- Do not add motion models, per-frame assignment or de-duplication to the host tracker
- Treat a within-camera identity problem as a device-tracker setting first (`TRACKER_TYPE`, YOLO thresholds)
- A device id is valid for one device track only: after `REMOVED` its next holder is a new observation

## Geometry

- Compare cameras only by world azimuth; edge, overlap, dead-zone, hysteresis and re-acquisition tests use the camera's local angle
- Correct the world azimuth at the fixed `rig.parallax_radius`, never at a measured distance
- Keep behavioural gates in degrees, fractions or seconds, never in metres; the measured distance only feeds read-outs and the far-edge filter
- Check `height_is_measured` on both readings before comparing two heights

## Identity

- One camera never has two active views in one world: links, re-acquisition and collapse refuse it, and `split_worlds` repairs it
- Observations are immutable; a world id changes only through `ObservationStore` (`add`, `merge_worlds`, `detach`)
- Apply the intake filters before any identity branch; a filtered observation already tracked goes LOST with its newest box, and its `last_active` does not advance
- Never filter on freshness in the tracker; staleness is the consumer's call
- Never emit a LOST primary while another view of that world is active
- Break ties by `obs_id`, never by a timestamp: `time.time()` has ~16 ms resolution on Windows

## Threads and tests

- Write the `Rig` only on the tracker thread (`RigSync.apply`); other threads read the published READ fields or `column_to_azimuth`
- Drive tracker tests synchronously through `_add_tracklet` / `_update_and_notify`, and send LOST and REMOVED as the device would
