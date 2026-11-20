# The video module's plan

## A. A clip's frame count

`StreamReader.frame_count` gives an mp4's stored frame count, but estimates an mkv's from its duration × fps, which
can read one short. The recording package counts each chunk's frames with it (`recording_folder.py`). Looked at when
the apps' plan F reaches the video module.

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

- Fix a problem where it is made: A
- The part that works on the item has its own thread: D2 (the preview's)
- A part that only needs the latest value reads it at its own tick: D2, E3
- Prefer not to wait on the GPU inside a frame: D2
- Create resources at setup, not per frame: D2 (the preview's buffers)
- All GL calls on the render thread: D2 (buffers mapped and fences set and checked there; the other threads only read or write mapped memory); the recorder's thread touches only RAM and the pipe
- Remove obsolete pathways, presets in sync: E1, E2, E4
- When a term changes, rename it everywhere in the same change: E1, the NV12 names; E3, `pause` to `freeze_frame`
- Shaders get checked in a GPU harness, named after their .py: D2, E1, E2
- A settings class mirrors the code, in the order it runs: E4
- The show owns only what it does with it: E3, each app holds its own show
- Settings connect the user and a component, not two components: E3, the apps read a property
- A document describes what is built: VIDEO.md in D2, E1–E4
- A package is one thing: D2, the preview in the screen recording's render package, not in modules/gl
- Prefer the design with the lower latency, measured in the app: E2
- One plan step checked by a run or test is one commit: D2, E1–E4
