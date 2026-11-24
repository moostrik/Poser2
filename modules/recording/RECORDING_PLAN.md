# The recording plan

What is next for recording and -sim, in order. How it works is in [RECORDING.md](RECORDING.md); the apps' move to
the recorder is in [the refactor's plan](../../REFACTOR_PLAN.md). **(deduction)** marks what is inferred, **(site
fact)** what was told, **(seen)** what is read in the code.

## A. The device's model

The device's model looks at the frame unstraightened, which for a few degrees of tilt finds the same players
**(deduction)**; nothing to run.

## B. The recorder in the other apps

With the apps' refactors ([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)), each tested with its refactored app:

- B1. hd_trio: its wiring (the tracklets from each camera as they come, `expect_parts` per camera, `expect_poses` per
  camera and set), as white_space's is wired, and its pose `.h5` cut at the recorder's split.
- B3. A white_space recording of the new layout played by hand, beside the old ones, which play as they are.

## C. The player's readouts

- C1. The playhead and `current_chunk` follow the frame shown. Now `_follow` sets `play_chunk` to the next chunk as
  soon as it is loaded, while the ending chunk's frames still go out, and `update_gui` adds that chunk's start to the
  shown frame's number (`player.py`, `_follow`, `update_gui`) **(seen)**; on `20261124-102304_Test` the playhead read
  1:57 of 1:17.4 with `current_chunk` 1 while the image looped right **(site fact, 2026-11-24)**, and
  `scripts/cadence_probe.py`, which waits on the playhead, measured nothing on 1–31 s and needed 61–91 **(seen)**.
  How: `FrameHandOver` keeps camera 0's last (chunk, frame) gone out (its `_handed_over`), and `update_gui` reads
  both from it; `play_chunk` stays the player's own. Why: the panel and the probe read where the recording is.
  Test: a player unit test with two timed chunks, the playhead within the first chunk while the second is followed;
  the probe on 1–31 s.

## D. Questions for the recording review (the refactor's A6)

- D2. Settled (2026-11-22): pose's `Recorder` becomes `PoseRecorder`, in [POSE_PLAN.md](../pose/POSE_PLAN.md) D1.

## The rules it touches

- Similar things share the noun and differ by a qualifier: D2 (built in POSE_PLAN.md D1)
- Fix a problem where it is made, not where it shows: C1
