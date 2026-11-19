# The video module's plan

## A. A clip's frame count

`StreamReader.frame_count` gives an mp4's stored frame count, but estimates an mkv's from its duration × fps, which
can read one short. The recording package counts each chunk's frames with it (`recording_folder.py`). Looked at when
the apps' plan F reaches the video module.

## B. The screen recorder's own panel, and one recording draw

The panel gives every child of the root settings a tab, so the recorder gets one by sitting at the root. deep_flow
and low_flow each hold their own copy of the recording draw (`_draw_screen_recording`, `_on_snap`); D2 would add a
third piece to both, so the draw goes to one place first.

- B1. Move `screen_recorder` from `RenderSettings` to the root `Settings` in deep_flow and low_flow. The recording
  keeps recording each app's `layer.main`. Each render takes the group through its constructor and keeps its `snap`
  bind and unbind. The keys move in both apps' `settings.json` in the same change, as the step's own preset change.
  VIDEO.md says where the recorder's settings sit. Check: guard tests, deep_flow and low_flow tests, both presets
  load with the moved keys. One commit.
- B2. `ScreenRecordingCompositor` in `modules/render/general/screen_recording/`: it owns the `ScreenCapture` and the
  `ScreenRecorder`, and draws the layers it is handed (layer objects, each fit or filled at its own shape, added on
  black), or copies the window, then runs the recorder's tick; `snap` takes the first layer's size or the window's.
  Each app hands it its `layer.main` layers and its window size at the end of its main draw, and calls its teardown
  in `deallocate`. The two apps' copies go. VIDEO.md names the compositor. Check: guard tests, both apps' tests; C2's
  run draws through it. One commit.

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
  to show Windows grants the pinned blocks and the app runs on while it is held. Check of C1 and B2 in the app; a
  fix found here is one more commit only if a test checks it.

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
- Update all affected call sites in the same change: B1, B2, C1 in deep_flow and low_flow.
- modules never import apps: B2, the compositor takes layer objects, not an app's `Layers`.
- A layer kind by what it owns: B2, a Compositor owns its FBO and draws other layers into it.
- Keep presets and settings schemas in sync: B1, C1.
- Pass collaborators and settings through constructors: B1, B2.
- Components that bind unbind on teardown: B1, B2 (snap), D1 (save, discard, in_out).
- Bound callbacks never write their own field: D1.
- A setting follows the quantity it controls: C1, `ram_reserve` beside `reserve`.
- Use a Protocol for what a consumer needs: C1, `ScreenRing`.
- Minimize CPU-GPU syncs: C1, D2.
- All GL calls on the render thread: D2; the recorder's thread touches only CUDA memory, RAM and the pipe.
- Avoid transient resource churn: D2, the preview made once with the capture.
- Shaders get checked in a GPU harness: D2.
- Name a shader's source after its `.py`: D2.
- A document describes what is built: VIDEO.md in B1, B2, C1, D1, D2.
- Prefer the design with the lower latency, measured in the app: C2.
- One plan step checked by a run or test is one commit: B1, B2, C1, C2, D1, D2.
