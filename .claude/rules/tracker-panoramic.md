---
description: The panoramic tracker — cross-camera identity on the camera rig
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

- Two active views from one camera are one person while their fresh boxes agree within `same_cam_angle` (the per-tick collapse re-merges converged pairs; `split_worlds` separates beyond 1.5×); intake joins only LOST anchors (`reacquire_angle`), never a simultaneous track — a one-shot match against the store can be stale. Cross-camera links and collapse still refuse a camera's occupied world
- Observations are immutable; a world id changes only through `ObservationStore` (`add`, `merge_worlds`, `detach`)
- Player slots are assigned and freed only inside `ObservationStore`; the tracker emits only slotted worlds, keyed and id-stamped by slot
- Apply the intake filters before any identity branch; a filtered observation already tracked goes LOST with its newest box, and its `last_active` does not advance
- Never filter on freshness in the tracker; staleness is the consumer's call
- Never emit a LOST primary while another view of that world carrying dead-zone weight is active; a view inside `seam.dead_zone` does not take over from one that carries weight
- Break ties between observations by `obs_id`

## Threads and tests

- Write the `Rig` only on the tracker thread (`RigSync.apply`); other threads read the published READ fields or `column_to_azimuth`
- Drive tracker tests synchronously through `_add_tracklet` / `_update_and_notify`
