# The recording plan

What is next for recording and -sim, in order. How it works is in [RECORDING.md](RECORDING.md); the apps' move to
the recorder is in [the refactor's plan](../../REFACTOR_PLAN.md). **(deduction)** marks what is inferred, **(site
fact)** what was told, **(seen)** what is read in the code.

The recorder in the other apps is in their plans: [hd_trio's](../../apps/hd_trio/HD_TRIO_PLAN.md) A1,
[white_space's](../../apps/white_space/WHITE_SPACE_PLAN.md) A1.

## B. A stop during a split

- B1. Proposed. Every camera's chunk `n` covers the same captures, a stop during a split included. What happens now:
  in `20261124-110331_probe_F1` (white_space live, 4 OAK-Ds, 2026-11-24) the session's 10 s split was asked at
  capture 372851.82 and the stop came about one frame later. Cams 1 and 3 each had one capture after the split time,
  so they cut a 1-frame chunk 006; cams 0 and 2 had none and end at 005 **(seen: the chunk metadata)**. That goes
  against RECORDING.md's "a split cuts every camera at one capture time". The cause: a camera cuts only at its first
  capture after `_split_at` (`_write_queued`), or at `SPLIT_WAIT` (`_update_recording`); `_stop_recording` writes
  what is queued and finishes each camera's current chunk, so a split still pending at the stop leaves the cameras
  that have not cut yet one chunk short **(seen)**. How: the cameras cut together, and a capture after the split
  time waits for that cut. In `_write_queued`, with a split pending, a camera's first capture after `_split_at` is
  taken out of its queue and held, one per camera, instead of being written; that camera's queue is read no further,
  so the captures after it stay in its queue. Once every camera holds one, or at `SPLIT_WAIT`, every camera is cut
  in one pass (`_split_camera`), its held capture is written into the new chunk and its queue is read again. At a
  stop with a split pending the split is dropped: the held captures go into the current chunk, then the queues, so
  every camera ends at the same chunk with every capture written. A held capture's bytes are subtracted from
  `_queued_bytes` when it is written, not when it is taken, so `HELD` bounds it as it bounds the queues. RECORDING.md
  follows: the `SPLIT_WAIT` line says how long the cameras wait for each other at a split, and the row "A split cuts
  every camera at one capture time" says the cameras cut in one pass, a stop during a split included. Why: a
  recording keeps everything **(site fact)**, and the player plays chunk `n` of every camera together, so a chunk
  only some cameras have plays out of step or not at all **(deduction)**. Why not put a cut camera's new chunk back
  into its old one at the stop: that joins two encoded clips and renumbers the metadata, at the stop, for what
  holding a second of captures does without **(deduction)**. Cost: the captures of a split's wait, at most
  `SPLIT_WAIT` of frames per camera, wait in the queues and are written that much later; nothing at the stop but
  writing them **(deduction)**. Left as it is: a camera with no capture for all of `SPLIT_WAIT` is cut empty, and if
  it sends nothing until the stop its new chunk has no frame and is deleted, so it ends one chunk short; that camera
  stopped delivering, and its chunks show it. Test: in `tests/test_rig.py`, through `recordings.record` (the
  recorder's thread and the iGPU encoders, skipped without them, as the rig tests are), two cameras of `cam_frame`s
  30 captures each; the split fired and waited on until `split_requested` clears, as in `test_round_trip`'s split
  test. Case one: one capture after the split time on camera 1 only, then the stop: each camera's folder holds chunk
  0 alone, its `stream_frames` and metadata every capture handed in. Case two: one capture after the split time on
  both, then more on both, then the stop: both end at chunk 1, cut in the same pass, chunk 1's metadata starting at
  those captures.

## The rules it touches

- Fix a problem where it is made, not where it shows: B1
