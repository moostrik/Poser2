# The video module's plan

## A. A clip's frame count

`StreamReader.frame_count` gives an mp4's stored frame count, but estimates an mkv's from its duration × fps, which
can read one short. The recording package counts each chunk's frames with it (`recording_folder.py`). Looked at when
the apps' plan F reaches the video module.

## B. low_flow's preset test

`apps/low_flow/tests/test_fluid_preset.py` requires low_flow's fluid values to equal deep_flow's; each app is tuned on its own now, so it fails on 8 values (2026-11-20). Built first, so the steps after it check both apps' tests passing.

- B3. The test checks what is code, not look: low_flow's `studio.json` holds every key of the 2D fluid's settings and loads; the value comparison and its docstring go. FLUID_2D.md: the line that the 2D takes deep_flow's values goes, and its settings test paragraph says keys, not values. Check: low_flow's tests pass. One commit.

## C. The frame handed over, the ring in RAM

The ring lives in RAM only. Video memory held 10 to 13 s of 4K in deep_flow; this PC has 96 GB of RAM, 74 GB free (measured 2026-11-19), about 80 s at 4:2:0 60 fps in 60 GB (deduction). Video memory gave nothing back for its length: the render tick does the same work either way, and the encode copied every frame to RAM anyway.

Recording into RAM jerks about once a second (seen by Matthias in low_flow, 2026-11-19): the code pins a one-second block (`cudaMallocHost`, about 750 MB at 4K60) each recorded second, which stalls the render tick (deduction). The recorder drives the capture through `ScreenRing` because the ring was CUDA memory only the GL side could write; with the ring in plain RAM the frame can be handed over like every other item.

- C2. The capture hands over finished frames. `ScreenCapture` and `ScreenPreview` move from modules/gl to `modules/render/general/screen_recording/` beside the compositor, their shaders in its `shaders/`, so modules/gl stays the framework and imports no video. `ScreenCapture` (render thread) is the producer: it draws and converts as now, then reads back with `glReadPixels` into one of four pack pixel buffers in host memory, mapped once (persistent and coherent) when it is allocated, and sets a fence after it. On a later tick it checks the fence without waiting; once it has passed it calls `add_screen_frame_callback`'s callbacks (a list with its own lock) with a `ScreenFrame`: the mapped buffer as an array, and its `timestamp` (the tick's). `ScreenFrame`, and E1's `ScreenFormat`, are plain dataclasses in modules/video, the recording's item and format; the capture imports them, so the recorder holds no OpenGL and no CUDA and its tests need no GPU. The capture frees or resizes its pixel buffers only once every one is back in the pool, the recorder's queue drained. `ScreenRecorder.submit_screen_frame` enqueues and returns. The recorder's thread wakes on the queue, copies the frame into its ring (plain RAM, numpy blocks of one second) and gives the buffer back to the pool (the render thread reuses it at its next tick). A full queue drops its oldest frame, gives its buffer back and counts it; with no buffer free the capture skips the tick and counts it. The count shows in the panel and in the file's comment. The recorder tells the capture whether to draw this tick, as `capturing` does now. Removed: `ScreenRing`, `ScreenMemory`, the `memory` and `reserve` settings (presets migrated in deep_flow and low_flow), the video-memory blocks, `add_block`, `queue_download` and `wait_download`, the capture's CUDA registration and streams, and the `cudart` calls only the capture used (malloc and free, pinned malloc and free, `mem_get_info`, streams, events, the device-to-host copy); the recording path holds no CUDA. VIDEO.md: the hand-over, the readback with fences, the RAM ring, why it is not pinned, the dropped frames. Check: the recorder's unit tests fed plain arrays (a take, look back, a full queue dropping its oldest); `scripts/screen_capture_check.py` rewritten for the pool and the hand-over, timing one 4K frame's copy out of a mapped buffer (if slow, the buffers get the client-storage flag); both apps' tests; C3's run. One commit.
- C3. One `-sim` run in deep_flow at 4K, 60 Hz with v_sync: tick time idle, recording, idle, and a 30 s take with its encode (encode time, frames, dropped, the file plays); then one take that runs until `ram_reserve` stops it, showing the app runs on while it is held. The ticks while recording show no stall once a second, and 0 dropped. Check of C2 and the drawing through `ScreenRecordingCompositor`, in the app. A fix found here is one more commit only if a test checks it.

## D. In and out before saving

- D2. A preview. `ScreenPreview` (beside the capture): `set_screen_frame` stores the latest frame to show (a held slot, which no one changes); two unpack pixel buffers, mapped once on the render thread (persistent and coherent); the preview's thread copies the frame straight into a free one and marks it written; the render tick uploads a written buffer into an RGBA Fbo, turns the planes back into RGB in a shader, and sets a fence on that pixel buffer; once that fence has passed, checked on a later tick without waiting, the render thread marks the buffer free. No CUDA in the preview. The drag stores the slot under the handle and the preview's thread picks it up, so a drag shows its frame while a save runs. `ScreenRecordingCompositor` draws it over the main window while a handle is dragged. D2's first code (in 49236068, its video-memory branch included, unchecked) is reworked onto this. VIDEO.md: the preview. Check: the shader in the GPU harness against a known frame; one `-sim` run, a held take trimmed by eye and saved, its frame count matching in..out. One commit.

The preview is there so a cut is made on what is seen, not on seconds alone.

## E. Quality, playback and the panel

Quality is traded against length; `max_duration` shows the length the chosen format gives. At 4K60 with about 60 GB of RAM: 4:2:0 8-bit ~80 s, 4:4:4 8-bit ~40 s (the default), 4:2:0 10-bit ~40 s, 4:4:4 10-bit ~20 s, or ~30 s if packed 10-bit RGB encodes as 4:4:4 (deduction, E2 probes it). Every other frame doubles each; HD is a quarter of 4K's bytes.

- E1. Full color. A setting `chroma`: 4:2:0 or 4:4:4 (default), read at Start. The NV12 names become the operation's: `Nv12` → `RgbToYuv` (the capture's shader, RGB to the format's YUV planes), `Nv12ToRgb` → `YuvToRgb` (the preview's, back), `nv12_bytes` → `ScreenFormat.frame_bytes` (the format of chroma and bit depth), and `ClipInput` gets the 4:4:4 input; each renamed everywhere in the same change, tests, harness and VIDEO.md included. HEVC 4:4:4 (Range Extensions) for 4:4:4, preset p7 with `-multipass fullres` and spatial and temporal AQ (`-spatial-aq 1 -temporal-aq 1`) for every format. The clip writer gets the 8 MB pipe the decode already uses (`_big_pipe`); Windows' default pipe takes 4 KB a write. VIDEO.md: the formats, and that 4:4:4 files are masters (edit software plays them, Windows' player often doesn't). Check: both shaders in the GPU harness against a known frame per format; an encode round trip per format; one 30 s 4K encode timed against the 58 fps measured; the recorder's tests for bytes and `max_duration`. One commit.
- E2. 10-bit. First a short encode probe outside the app: does hevc_nvenc take packed 10-bit RGB (4 bytes a pixel) and write 4:4:4 10-bit? A setting `bit_depth`: 8 or 10. At 10 the recording's frame keeps the layers' precision (RGB10_A2 when the probe says yes, else RGBA16F into 16-bit planes: P010, or 16-bit 4:4:4), encoded as HEVC Main 10 or its 4:4:4 profile. Check: the probe; as E1, with a gradient frame showing no 8-bit steps after the round trip; one `-sim` take at the heaviest format with 0 dropped. One commit.
- E3. Playback. A `play` switch on a held take: the recorder's thread sets the take's frames into `ScreenPreview` at their recorded times, looping between in and out, and the compositor draws it over the main window. Play is off while a save runs, as both use the recorder's thread. The secondary windows keep the last show picture, frozen. While it plays the show is held: each app reads `ScreenRecordingCompositor.playing`, a property, not the `play` setting, and skips the flow's work and draw, dropping camera frames as a frozen frame does, and skips the fluid's steps, so the fluid goes on where it stopped. The existing `pause`, which holds the camera frame while the fluid runs on with it, is renamed `freeze_frame` and keeps what it does. In deep_flow and low_flow: settings, renders and every preset. Check: both apps' tests; one `-sim` run, a held take played and looped, ticks steady, the show going on where it stopped. One commit.
- E4. The panel in four groups, in the order the code runs: Frame (main window, size, portrait, width, height, snap, fill), Quality (chroma, bit depth, frame step; max duration as readout), Record (start, stop, look back, duration, RAM reserve; recorded and dropped as readouts), Take (in and out, play, save, discard, name, quality). The size and button binds, and their unbinds, move to the group objects. One meaning per button: in look back Stop holds the take, as in a recording, and `save` only writes. Presets migrated in both apps. VIDEO.md: the four groups, and Stop holding the take in look back. Check: guard tests, both apps' tests, both presets load. One commit.

## The rules it touches

- Fix a problem where it is made: A; C2, at the allocation that stalls
- A producer calls its callbacks itself, a callback only hands over: C2; a callback list has its own lock
- Every other frame type is qualified: C2, `ScreenFrame`
- An item that leaves its package is finished, its GPU work included: C2, the fence checked before the hand-over
- An item handed to a callback is not changed afterwards: C2, a buffer reused only after the recorder gives it back
- The part that works on the item has its own thread: C2 (the recorder's), D2 (the preview's)
- Its queue drops the oldest item and counts it: C2
- A part that only needs the latest value reads it at its own tick: D2, E3
- Prefer not to wait on the GPU inside a frame: C2, D2
- Create resources at setup, not per frame: C2 (the pool), D2 (the preview's buffers)
- All GL calls on the render thread: C2, D2 (buffers mapped and fences set and checked there; the other threads only read or write mapped memory); the recorder's thread touches only RAM and the pipe
- Use a Protocol for what a consumer needs: C2 removes `ScreenRing`, which the hand-over no longer needs
- Remove obsolete pathways, presets in sync: C2, E1, E2, E4
- When a term changes, rename it everywhere in the same change: E1, the NV12 names; E3, `pause` to `freeze_frame`
- Shaders get checked in a GPU harness, named after their .py: C2, D2, E1, E2
- A settings class mirrors the code, in the order it runs: E4
- The show owns only what it does with it: E3, each app holds its own show
- Settings connect the user and a component, not two components: E3, the apps read a property
- A document describes what is built: VIDEO.md in C2, D2, E1–E4; FLUID_2D.md in B3
- A package is one thing: C2, D2, the capture and the preview in the screen recording's render package, not in modules/gl
- Prefer the design with the lower latency, measured in the app: C3, E2
- Unit-test infrastructure; test the look by running the system: B3
- One plan step checked by a run or test is one commit: B3, C2, C3, D2, E1–E4
