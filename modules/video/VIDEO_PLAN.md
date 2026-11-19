# The video module's plan

## A. A clip's frame count

`StreamReader.frame_count` gives an mp4's stored frame count, but estimates an mkv's from its duration × fps, which
can read one short. The recording package counts each chunk's frames with it (`recording_folder.py`). Looked at when
the apps' plan F reaches the video module.

## C. The ring in RAM

The ring holds 10 to 13 s of 4K in free video memory; this PC has 96 GB of RAM, 74 GB free (measured 2026-11-19),
which holds about 80 s at 60 fps in 60 GB (deduction).

Recording into RAM jerks about once a second (seen by Matthias in low_flow, 2026-11-19). The code adds a pinned
block of one second (`cudaMallocHost`, about 750 MB at 4K60) each recorded second; pinning that much locks every
page and serializes the driver with the GPU, which stalls the render tick (deduction).

- C2. The RAM ring unpinned. In HOST the ring's blocks are plain RAM (numpy), which allocates without locking
  anything; two pinned staging buffers of one frame each, made with the capture, take the frames. `flush` copies
  last tick's pixel buffer device to host into a staging buffer on the capture's stream, not waited for, with an
  event per buffer; the recorder's thread waits for that event and copies the frame into its slot (about 12 MB per
  frame at UHD, 2 to 3 ms (deduction)), so a staging buffer is free again before the capture writes it two ticks
  later. The render thread does no more than now. The encode and the preview read the RAM slots as they are. VIDEO
  stays as it is. VIDEO.md: the RAM ring and its staging buffers, the reason against pinning it. Check: the recorder's
  unit tests with the fake ring; C3's run. One commit.
- C3. One -sim run in deep_flow at 4K, 60 Hz with v_sync: tick time idle, recording in HOST, idle, and a 30 s take
  with its encode (encode time, frames, fps, the file plays); then one take that runs until `ram_reserve` stops it,
  to show Windows grants the blocks and the app runs on while it is held. The ticks while recording into RAM must
  show no stall once a second. Check of the ring in RAM, C2's copy and the drawing through
  `ScreenRecordingCompositor`, in the app; a fix found here is one more commit only if a test checks it. The copy's
  cost over PCIe is a deduction in VIDEO.md; C3 measures it.

- Question: if a staging buffer is still waiting for the recorder's thread when the capture needs it (the thread
  behind by two frames), should the frame be dropped and counted, or the render tick wait?

## D. In and out before saving

- D2. A preview: `ScreenCapture.preview`, an RGBA Fbo of the recording's size, filled on the render tick from the
  slot under the dragged handle through an unpack pixel buffer in the `Tensor` texture's way (registered
  write-discard, copied device to device; in HOST a plain upload from the RAM slot), then the NV12 planes back to
  RGB by a shader beside `nv12.py`. The recorder says which slot; the preview's GL objects are made and freed with
  the capture's. `ScreenRecordingCompositor` draws `preview` over the main window while a handle is dragged.
  VIDEO.md: the preview. Check: the shader in a GPU harness against a known NV12 frame; one -sim run, a held take
  trimmed by eye and saved, its frame count matching in..out. One commit.

The preview is there so a cut is made on what is seen, not on seconds alone.

## The rules it touches

- Fix a problem where it is made: A; C2, at the allocation that stalls.
- Minimize CPU-GPU syncs: C2 (an event per staging buffer, waited for by the recorder's thread), D2.
- Prefer not to wait on the GPU inside a frame: C2.
- The part that works on the item has its own thread: C2, the copy into the slot on the recorder's thread.
- When that part can't keep up, its queue drops the oldest item and counts it: C2, the open question.
- Create resources at setup, not per frame: C2, the staging buffers made with the capture.
- All GL calls on the render thread: D2; the recorder's thread touches only CUDA memory, RAM and the pipe.
- Avoid transient resource churn: D2, the preview made once with the capture.
- Shaders get checked in a GPU harness: D2.
- Name a shader's source after its `.py`: D2.
- A document describes what is built: VIDEO.md in C2, D2.
- Prefer the design with the lower latency, measured in the app: C3.
- One plan step checked by a run or test is one commit: C2, C3, D2.
