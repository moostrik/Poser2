---
description: The panoramic tracker — cross-camera identity on a ring of cameras
paths:
  - "modules/tracker/panoramic/**"
---
# Panoramic Tracker Guidelines

The panoramic tracker owns cross-camera identity only: seam links, world merges and splits, the emitted view, re-linking a REMOVED device track.

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
- Break ties between observations by `obs_id`

## Threads and tests

- Write the `Rig` only on the tracker thread (`RigSync.apply`); other threads read the published READ fields or `column_to_azimuth`
- Drive tracker tests synchronously through `_add_tracklet` / `_update_and_notify`
