# The video module's plan

## A. A clip's frame count

`StreamReader.frame_count` gives an mp4's stored frame count, but estimates an mkv's from its duration × fps, which
can read one short. The recording package counts each chunk's frames with it (`recording_folder.py`). Looked at when
the apps' plan F reaches the video module.

## C. The ring in RAM

The ring holds 10 to 13 s of 4K in free video memory; this PC has 96 GB of RAM, 74 GB free (measured 2026-11-19),
which holds about 80 s at 60 fps in 60 GB (deduction).

- C1. A setting `memory`: HOST (the default) or VIDEO (for short takes), read at Start. In HOST the one-second
  blocks are pinned host memory (`cudaMallocHost`), and `ScreenCapture.flush` copies last tick's pixel buffer device
  to host into the slot, on the capture's stream, without waiting; the unmap orders it before GL writes the buffer
  again. The encode reads a slot from RAM directly after `synchronize`, with no download. A setting `ram_reserve`
  (GB of RAM left free for Windows and the app) beside `reserve` (video memory); in HOST the block check and
  `max_duration` read free RAM through psutil against `ram_reserve`. `duration` by time, look back and its 1.2 ×
  margin work as in VIDEO, on RAM. The `ScreenRing` protocol gains what the recorder needs to tell the two apart.
  Both apps' presets only if a non-default value is needed. VIDEO.md: the ring's two memories and their numbers.
  Check: the recorder's unit tests with the fake ring in both modes. One commit.
- C2. One -sim run in deep_flow at 4K, 60 Hz with v_sync: tick time idle, recording in HOST, idle, and a 30 s take
  with its encode (encode time, frames, fps, the file plays); then one take that runs until `ram_reserve` stops it,
  to show Windows grants the pinned blocks and the app runs on while it is held. Check of C1, and of the drawing
  through `ScreenRecordingCompositor`, in the app; a fix found here is one more commit only if a test checks it.

Why one-second blocks also here: Windows handles many small pinned allocations better than one very large one
(deduction). The copy is about 750 MB/s over PCIe at 4K60, on a link of about 25 GB/s, and the CPU does not touch it
(deduction, C2 measures it).

## D. In and out before saving

- D1. A state HELD: after Stop, or Save in look back, the take stays in the ring instead of being encoded; Start
  and the look-back recording wait until it is discarded. An `in_out` timeline on the recorder's panel
  (`Widget.timeline` with its playhead, drag and seek fields, as the player's), over the held take's length by its
  frames' times, and two buttons: `save` encodes the frames between in and out and keeps the take held, so a second
  part can be saved; `discard` frees the ring (in look back: empties it and records on). The buttons set Events,
  the render tick acts. VIDEO.md: the held take and its states. Check: unit tests of HELD, the in/out trim by time,
  a second save and discard, with the fake ring. One commit.
- D2. A preview: `ScreenCapture.preview`, an RGBA Fbo of the recording's size, filled on the render tick from the
  slot under the dragged handle through an unpack pixel buffer in the `Tensor` texture's way (registered
  write-discard, copied device to device; in HOST a plain upload from the pinned slot), then the NV12 planes back to
  RGB by a shader beside `nv12.py`. The recorder says which slot; the preview's GL objects are made and freed with
  the capture's. `ScreenRecordingCompositor` draws `preview` over the main window while a handle is dragged.
  VIDEO.md: the preview. Check: the shader in a GPU harness against a known NV12 frame; one -sim run, a held take
  trimmed by eye and saved, its frame count matching in..out. One commit.

The preview is there so a cut is made on what is seen, not on seconds alone.

## The rules it touches

- Fix a problem where it is made: A.
- Update all affected call sites in the same change: C1 in deep_flow and low_flow.
- Keep presets and settings schemas in sync: C1.
- Components that bind unbind on teardown: D1 (save, discard, in_out).
- Bound callbacks never write their own field: D1.
- A setting follows the quantity it controls: C1, `ram_reserve` beside `reserve`.
- Use a Protocol for what a consumer needs: C1, `ScreenRing`.
- Minimize CPU-GPU syncs: C1, D2.
- All GL calls on the render thread: D2; the recorder's thread touches only CUDA memory, RAM and the pipe.
- Avoid transient resource churn: D2, the preview made once with the capture.
- Shaders get checked in a GPU harness: D2.
- Name a shader's source after its `.py`: D2.
- A document describes what is built: VIDEO.md in C1, D1, D2.
- Prefer the design with the lower latency, measured in the app: C2.
- One plan step checked by a run or test is one commit: C1, C2, D1, D2.
