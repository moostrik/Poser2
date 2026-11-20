# Video

The video package writes and reads clips through ffmpeg (`ffmpeg.py`), the camera recordings' streams
(`stream_writer.py`, `stream_reader.py`, `depth_frame.py`), and the screen recorder (`screen_recorder.py`,
`screen_frame.py`, `screen_frame_ring.py`). This document holds the screen recorder's design.

## Screen recorder

Each render tick the layers an app records are drawn into the screen frame, an Fbo of the recording's size;
`ScreenCapture` (`modules/render/general/screen_recording/`) converts it to NV12, reads it back into RAM and hands
each finished frame (`ScreenFrame`) to the `ScreenRecorder`, whose thread copies it into a ring in RAM. When the
ring is full, the duration is reached or Stop is pressed, recording stops and the take is held: in and out are set
on it, and each `save` has the recorder's thread encode the frames between them to an HEVC mp4 with NVENC;
`discard` frees the memory. The render thread never waits on the GPU, the copy or the encoder.

### What gets recorded

The recorder's settings are the group `screen_recorder` at the root of an app's settings, so it has a tab
of its own on the panel (deep_flow and low_flow).

`ScreenRecordingCompositor` (`modules/render/general/screen_recording/`) owns the `ScreenCapture` and the
`ScreenRecorder`. Each tick, right after the main window is drawn, the app hands it the layers to record with
their shapes (`ScreenRecordingSource`), the window's size and the tick's time; deep_flow and low_flow hand it
the main panel's layers (`layer.main`). It draws them into `ScreenCapture.frame`, each fit and centered at its
own shape and blended the way the show draws them (additive on black), so a recording is the show. With
`main_window` the window's back buffer is copied into the frame instead (`ScreenCapture.copy_window`), before
the window is shown. A module cannot know an app's layer enum, so the app chooses the layers and the
compositor takes them as layer objects.

A source whose shape differs from the recording's is fit with black bars, or with `fill` fills the frame and
its overhang is cut. It is never stretched.

### Size

`width` and `height` are the recording's size. A preset (`size`) or the `portrait` switch writes them; when
they change, the preset becomes the one that matches in either orientation, else CUSTOM, and `portrait`
becomes height > width. Every write lands on the value already there, so the three never loop. `snap` sets
the size to the first source's shape, or the window's with `main_window`. Sizes are rounded down to even numbers, which
NV12 needs. The size is read at Start and holds for that recording.

| Preset | Landscape   | Portrait    |
|--------|-------------|-------------|
| HD     | 1920 × 1080 | 1080 × 1920 |
| QHD    | 2560 × 1440 | 1440 × 2560 |
| UHD    | 3840 × 2160 | 2160 × 3840 |
| CUSTOM | as set      | as set      |

### How many seconds fit

NV12 is 1.5 bytes per pixel: luma at full resolution, chroma at half width and half height, BT.709 video
range. The render loop is not locked to a rate, so the length follows the window's measured tick rate.

| Recording (60 ticks/s) | Per frame | Per second | Seconds in 8 GB |
|------------------------|-----------|------------|-----------------|
| UHD, NV12              | 12.4 MB   | 746 MB     | 10.7 s          |
| UHD, RGBA8 (not used)  | 33.2 MB   | 1.99 GB    | 4.0 s           |
| QHD, NV12              | 5.5 MB    | 332 MB     | 24.1 s          |
| HD, NV12               | 3.1 MB    | 187 MB     | 42.9 s          |

All four rows are **(deduction)**: computed from the frame sizes, not measured. Per GB the rows scale
alike: UHD NV12 at 60 ticks/s holds about 80 s in 60 GB of RAM **(deduction)**; the PC has 96 GB of RAM, 74 GB
free with the app off, and a 4090 with 24 GB **(measured 2026-11-19)**.

The ring is plain RAM (`ScreenFrameRing`, numpy blocks): a block allocates without locking any page, and its
memory goes back when the ring is cleared. It is not pinned: pinning a one-second block (about 750 MB at UHD60)
locks every page and serializes the driver with the GPU, which stalled the render tick once a second
**(seen in low_flow, 2026-11-19)**. Video memory gave the ring nothing back for its length: the render tick does
the same work either way, the encode needs every frame in RAM, and the fluid and the models use the card's memory.

The ring grows while it records, in blocks of one second. The recorder's thread adds the next block when half of
the last one is left, after reading the RAM available to processes (psutil), so what the rest of the app holds is
counted. When the next block would leave less than `ram_reserve` free (default 8 GB), the ring grows no more and
the recording stops when it is full. The check runs once per recorded second and Start makes no large allocation.

A run in deep_flow (`-sim`, 20261112-130105_show, UHD portrait, 60 Hz with v_sync, `ram_reserve` 10 GB,
commit 51145c73) **(measured 2026-11-20)**:

| Phase                 | Ticks | Mean (ms) | p95 (ms) | Max (ms) | Over 25 ms | Dropped |
|-----------------------|-------|-----------|----------|----------|------------|---------|
| Idle                  | 178   | 16.75     | 26.4     | 38.9     | 8 %        |         |
| Recording 30 s        | 1821  | 16.69     | 25.8     | 42.8     | 7 %        | 0       |
| Encoding the 30 s     | 2609  | 17.00     | 27.9     | 45.2     | 11 %       |         |
| Recording to reserve  | 6171  | 16.74     | 25.8     | 62.3     | 7 %        | 0       |
| Held, 60 GB in RAM    | 300   | 16.72     | 24.8     | 32.7     | 4 %        |         |

The long ticks come as often idle as while recording and at no once-a-second rhythm. The 30 s take held 1800 frames
at 59.93 fps (worst gap 33.3 ms) and encoded in 20.1 s, 90 frames/s (118.6 MB). The take to the reserve held 6151
frames, 103 s, and stopped with 10.3 GB free; the app ran on while it was held, and `discard` gave the RAM back.

`max_duration` is the estimate (free RAM + the ring's own − `ram_reserve`) / (bytes per frame × measured ticks/s /
`frame_step`), refreshed once a second. `duration` 0, or above `max_duration`, records until the memory stops it.

`duration` is counted in time, by the frames' own timestamps: a take ends at the first frame whose time is
`duration` after the first frame's, so it spans `duration` whatever the tick rate does while it records. The
blocks are sized by the rate measured at Start; they are only chunks of memory, and the ring grows by as many
as the take needs.

### Capture, on the render thread

- The Nv12 shader (`shaders/nv12.py`, `nv12.frag` in the screen recording's package) converts the screen frame in
  two passes: Y into a one-channel texture at full size, Cb Cr into a two-channel texture at half size, each chroma
  sample from the average of its 2 × 2 pixels; flipped, so the first row read back is the picture's top. The
  BT.709 matrix is plain code beside the shader (`bt709_video_range`).
- `glReadPixels` reads both planes, Y then UV, into one of four pixel buffers in host memory, each mapped once
  (persistent and coherent) when the capture is made, and a fence is set after it. The GPU writes the frame into
  RAM; the CPU does not wait.
- `poll`, on every tick, checks the oldest fences without waiting; once one has passed, the frame goes to the
  callbacks (`add_screen_frame_callback`, a list with its own lock) as a `ScreenFrame`: the mapped buffer as an
  array, its tick's `timestamp`, and the ticks skipped before it. The buffer is lent until the frame's `release`.
  A frame enters the ring one to two ticks after it was drawn.
- With all four buffers lent the capture leaves the tick out; the next frame carries the count (`skipped`).
- The compositor makes the capture at the recorder's size when a take begins (`ScreenRecorder.capture_size`) and
  frees it after; never while a frame is lent or a read-back is on its way, as its memory would go under a reader.
- The capture sets its own blend and viewport and leaves framebuffer 0 bound. Nothing is drawn or converted unless
  the recorder captures this tick (`ScreenRecorder.capturing`).
- No CUDA: the read-back is GL's own, and a mapped buffer in host memory reads as plain RAM. `scripts/screen_capture_check.py`
  checks the NV12 codes of known colors, the preview, the pool's lending and skipping, and times one UHD frame's copy
  out of a mapped buffer: 0.31 ms over 20 copies **(measured 2026-11-20, 2160 × 3840, partly from the CPU's cache)**,
  which says the mapped memory is cached.

### The hand-over, to the recorder's thread

- `ScreenRecorder.submit_screen_frame` enqueues the frame and returns; a frame no take wants is given back at once.
- The recorder's thread copies each frame into its slot in the ring and gives the buffer back. A full queue (three
  frames) drops its oldest frame and gives its buffer back. Frames left out, by the queue or by the capture with no
  buffer free, are counted for the take (`dropped`), shown on the panel and written in the file's comment.
- Frames still on their way when a take is held, or from before a take began, are given back unused.

### Encode, on the recorder's thread

- One ffmpeg process takes raw NV12 on stdin (`ClipWriter` with `ClipInput.NV12`, `ClipCodec.HEVC`):
  `hevc_nvenc -preset p5 -tune hq -rc vbr -cq <quality> -b:v 0`, tagged BT.709 video range and `hvc1`, so
  Windows and QuickTime play it. HEVC because fine detail at 4K needs fewer bits than h264 for the same
  quality.
- Each slot of the ring is written to the pipe as it is.
- After the last frame the pipe is closed and ffmpeg waited for; the ring stays held.
- The thread makes no GL call; it touches only RAM and the pipe. NVENC is its own unit on the card, so the show
  runs on while it encodes.

Files go to the folder the app gives, named `<date-time>_<name>.mp4`; a second part saved within the same second
gets `_2`, and so on.

### The held take

A held take keeps its ring until `discard`. The `in_out` timeline (the player's `Widget.timeline`, with its
`playhead`, `scrub` and `seek` fields) spans the take's length by its frames' times; `save` writes the frames whose
times lie between in and out, and the take stays held, so a second part can be saved from it. `discard` frees the
ring, or in look back empties it and records on. Start is refused while a take is held. The playhead follows the
dragged marker, or the place on the bar pressed, so it shows where the take is looked at. `scrub` and `seek` are
bound, because the panel writes them back to −1 at once.

While a marker is dragged, the main window shows the held frame nearest to it (`ScreenRecorder.preview_frame`), fit
on black, so a cut is made on what is seen. `ScreenPreview` (beside the capture) draws it without the render thread
waiting: the render tick stores the frame under the marker (`set_screen_frame`, a held slot no one changes), and
the preview's own thread copies it straight into one of two unpack pixel buffers, mapped once on the render thread
(persistent and coherent), and marks it written. The render tick uploads the newest written buffer into a Y and a
Cb Cr texture, turns them back into RGB in `ScreenPreview.fbo` with the `Nv12ToRgb` shader
(`shaders/nv12_to_rgb.py`), upright, and sets a fence on the buffer; once the fence has passed, checked on a later
tick without waiting, the buffer is free again. No CUDA. A drag shows its frames while a save runs, as the recorder's
thread takes no part. In deep_flow (`-sim`, UHD portrait, 60 Hz with v_sync, commit 51145c73 with the preview)
**(measured 2026-11-20)**: dragging over a held 10 s take for 3 s drew 142 pictures in 181 ticks, the ticks 16.66 ms
on average (max 33 ms, 11 % over 25 ms, 8 % held without a drag).

### Look back

With `look_back` on the ring records all the time, the newest frame replacing the oldest. It holds 1.2 ×
`duration` at the rate measured when it is made (`LOOK_BACK_MARGIN`), so ticks faster than that still leave
it the last `duration` seconds; with `duration` 0 it holds what the memory holds. `save` holds the frames whose
times lie in the last `duration` seconds, oldest first (see the held take); a slot being written over at that
moment is left out. The ring pauses while the take is held and records on after `discard`; it is blind for that
time, because recording on in a second ring would need the memory twice. It holds its memory while the switch is
on; a change of size or duration empties it and makes it again.

### Time

One frame per render tick, or per second tick with `frame_step` 2. Each slot keeps its frame's `timestamp`, the
tick's. The file is written at the rate measured over the take, (frames − 1) / (last − first time), so every
stored frame is in it once and the video lasts as long as the take did; uneven ticks play back evened out. The
mean and the worst gap and the frames dropped go in the log line and the file's comment.

### States

| State     | Means                                                        |
|-----------|--------------------------------------------------------------|
| IDLE      | nothing held; Start, or look back on, begins                 |
| RECORDING | frames are captured and stored each tick                     |
| HELD      | the take waits for in, out, `save` and `discard`             |
| ENCODING  | the thread encodes a part, or frees the ring                 |
| ERROR     | a failed capture or encode; only Start retries               |

A failure (no ffmpeg, no NVENC, a capture that cannot be made) is logged once, shown as ERROR, and the show
runs on. At exit a recording in progress is written up to its last stored frame, and a held take none of whose
parts is written yet is written whole; the thread is joined, every frame is given back and the ring freed, then the
render thread frees the capture.

### Limits

- 8 bit: video range keeps 219 of the window's 256 levels, and smooth dark smoke can band after compression.
  10-bit HEVC would halve the duration.
- Sizes that are not multiples of 16 are padded inside HEVC and cropped on playback; any even size is exact
  in the file.
- NVENC refuses frames 128 wide and less (measured, ffmpeg's `hevc_nvenc` on the 4090).

### Sources

- songho, OpenGL Pixel Buffer Object: the asynchronous read-back into a pixel buffer.
- OpenGL 4.4 buffer storage (`glBufferStorage`, persistent and coherent mapping) and sync objects
  (`glFenceSync`, `glClientWaitSync` with a timeout of 0): a mapped buffer read on another thread once its fence
  has passed.
- PyNvVideoCodec would encode straight from memory without ffmpeg. The recorder pipes through ffmpeg instead: the
  encode runs after Stop, ffmpeg is already the encoder here, and it adds no package. It is the swap if the
  encode is too slow.

### Constraints from the alternatives

- The ring is RAM, not GL textures or buffers: a GL ring would be read back on the render thread while the show
  runs, and the encode thread cannot touch GL.
- The ring is not pinned (above).
- The encode runs after the recording, not live: live it would push ~750 MB/s **(deduction)** through a pipe and
  NVENC while the show runs.
- NV12 is made in a fragment shader, not a compute shader: GLSL has no byte arrays, and packing four pixels
  per integer would force widths to multiples of 4.
- The ring holds NV12, not RGBA: 2.7× the seconds, and the encoder makes 4:2:0 anyway.
