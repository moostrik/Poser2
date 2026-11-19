# The video module's plan

## A. A clip's frame count

`StreamReader.frame_count` gives an mp4's stored frame count, but estimates an mkv's from its duration × fps, which
can read one short. The recording package counts each chunk's frames with it (`recording_folder.py`). Looked at when
the apps' plan F reaches the video module.

## C. The ring in RAM

The ring holds 10 to 13 s of 4K in free video memory; this PC has 96 GB of RAM, 74 GB free (measured 2026-11-19),
which holds about 80 s at 60 fps in 60 GB (deduction).

- C2. One -sim run in deep_flow at 4K, 60 Hz with v_sync: tick time idle, recording in HOST, idle, and a 30 s take
  with its encode (encode time, frames, fps, the file plays); then one take that runs until `ram_reserve` stops it,
  to show Windows grants the pinned blocks and the app runs on while it is held. Check of the ring in RAM, and of the drawing
  through `ScreenRecordingCompositor`, in the app; a fix found here is one more commit only if a test checks it.
  The copy's cost over PCIe is a deduction in VIDEO.md; C2 measures it.

## D. In and out before saving

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
- Minimize CPU-GPU syncs: D2.
- All GL calls on the render thread: D2; the recorder's thread touches only CUDA memory, RAM and the pipe.
- Avoid transient resource churn: D2, the preview made once with the capture.
- Shaders get checked in a GPU harness: D2.
- Name a shader's source after its `.py`: D2.
- A document describes what is built: VIDEO.md in D2.
- Prefer the design with the lower latency, measured in the app: C2.
- One plan step checked by a run or test is one commit: C2, D2.
