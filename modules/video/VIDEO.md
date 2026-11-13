# Video

The video package writes and reads clips through ffmpeg (`ffmpeg.py`), the camera recordings' streams
(`stream_writer.py`, `stream_reader.py`, `depth_frame.py`), and the screen recorder (`screen_recorder.py`).
This document holds the screen recorder's design.

## Screen recorder

Each render tick the app draws the layers it records into the screen frame, an Fbo of the recording's size,
and `ScreenCapture` (`modules/gl/screen_capture.py`) copies it as NV12 into a ring of frames in video memory.
When the ring is full, the duration is reached or Stop is pressed, recording stops and the recorder's thread
encodes the ring to an HEVC mp4 with NVENC, then frees the memory. The render thread never waits on the
encoder.

### What gets recorded

The app chooses the sources and draws them into `ScreenCapture.frame`: in deep_flow the layer list
`layer.record`, each fit and centered at its own shape and blended the way the show draws them (additive on
black), so a recording of the composite is the show. With `main_window` the window's back buffer is copied
into the frame instead (`ScreenCapture.copy_window`), right after the window is drawn and before it is shown.
A module cannot know an app's layers, so the drawing is the app's and the capture is the module's.

A source whose shape differs from the recording's is fit with black bars, or with `fill` fills the frame and
its overhang is cut. It is never stretched.

### Size

`width` and `height` are the recording's size. A preset (`size`) or the `portrait` switch writes them; when
they change, the preset becomes the one that matches in either orientation, else CUSTOM, and `portrait`
becomes height > width. Every write lands on the value already there, so the three never loop. `snap` sets
the size to the first source's texture (the app wires it). Sizes are rounded down to even numbers, which
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

All four rows are **(deduction)**: computed from the frame sizes, not measured.

The ring grows while it records, in blocks of one second. The recorder's thread adds the next block when
half of the last one is left, after reading the free video memory (`cudaMemGetInfo`, so what the rest of the
app holds is counted). When the next block would leave less than `reserve` free, the ring grows no more and
the recording stops when it is full. The check runs once per recorded second, the reserve can be small
(1 GB), and Start makes no large allocation. Windows pages video memory into system RAM when it runs short
(WDDM), which shows as stutter; the reserve keeps clear of that.

`max_duration` is the estimate (free + the ring's own − `reserve`) / (bytes per frame × measured ticks/s /
`frame_step`), refreshed once a second. `duration` 0, or above `max_duration`, records until the memory
stops it.

### Capture, on the render thread

- The Nv12 shader (`modules/gl/shaders/nv12.py`, `nv12.frag`) converts the screen frame in two passes: Y into
  a one-channel texture at full size, Cb Cr into a two-channel texture at half size, each chroma sample from
  the average of its 2 × 2 pixels; flipped, so the first row read back is the picture's top. The BT.709
  matrix is plain code beside the shader (`bt709_video_range`).
- `glReadPixels` reads both planes into one pixel buffer, Y then UV. The read stays on the GPU.
- The pixel buffer is registered read-only with CUDA once (`modules/gl/cudart.py`). The capture maps it,
  queues a device-to-device copy into the ring slot and unmaps it. The ring's blocks are plain `cudaMalloc`
  memory, not torch's, so freeing them gives the memory straight back.
- Two pixel buffers take turns: a tick converts into one while the copy maps the one written the tick before.
  GL has finished that one by then, so the map does not make the CPU wait for the GPU. This is the
  double-buffered pixel buffer readback, applied to a GPU-to-GPU copy. A frame enters the ring one tick after
  it was drawn; Stop and Save copy the last one out (`flush`).
- The capture sets its own blend and viewport and leaves framebuffer 0 bound. Its GL objects are made at
  Start, or when look back is switched on, and freed by the render tick after the encode, and in
  `deallocate`.
- Nothing is drawn or converted unless the recorder is recording (`ScreenRecorder.capturing`).

### Encode, on the recorder's thread

- One ffmpeg process takes raw NV12 on stdin (`ClipWriter` with `ClipInput.NV12`, `ClipCodec.HEVC`):
  `hevc_nvenc -preset p5 -tune hq -rc vbr -cq <quality> -b:v 0`, tagged BT.709 video range and `hvc1`, so
  Windows and QuickTime play it. HEVC because fine detail at 4K needs fewer bits than h264 for the same
  quality.
- Each frame is copied from its slot into one of two pinned host buffers on a stream of its own; the next
  copy runs while the last frame is written to the pipe.
- After the last frame the pipe is closed, ffmpeg waited for and the ring freed (`cudaFree`).
- The thread makes no GL call; it touches only CUDA memory and the pipe. NVENC is its own unit on the card,
  so the show runs on while it encodes. Start is refused until the encode is done.

Files go to the folder the app gives, named `<date-time>_<name>.mp4`.

### Look back

With `look_back` on the ring records all the time, the newest frame replacing the oldest, so it holds the
last `duration` seconds (or what the memory holds). `save` encodes them, oldest first. The ring pauses while
the encode reads it and records on when the file is written; it is blind for that time, because recording on
in a second ring would need the memory twice. It holds its video memory while the switch is on; a change of
size or duration empties it and makes it again.

### Time

One frame per render tick, or per second tick with `frame_step` 2. Each slot keeps its tick's time, the one
the render tick gets. The file is written at the rate measured over the take, (frames − 1) / (last − first
time), so every captured frame is in it once and the video lasts as long as the take did; uneven ticks play
back evened out. The mean and the worst gap go in the log line and the file's comment.

### States

| State     | Means                                                        |
|-----------|--------------------------------------------------------------|
| IDLE      | nothing held; Start, or look back on, begins                 |
| RECORDING | the ring is filled each tick                                 |
| ENCODING  | the thread encodes, or frees the ring                        |
| ERROR     | a failed allocation, capture or encode; only Start retries   |

A failure (no ffmpeg, no NVENC, a failed CUDA registration) is logged once, shown as ERROR, and the show
runs on. At exit a recording in progress is encoded up to its last copied frame, the thread is joined and the
CUDA memory freed; the render thread then frees the GL objects.

### Limits

- 8 bit: video range keeps 219 of the window's 256 levels, and smooth dark smoke can band after compression.
  10-bit HEVC would halve the duration.
- Sizes that are not multiples of 16 are padded inside HEVC and cropped on playback; any even size is exact
  in the file.
- NVENC refuses frames 128 wide and less (measured, ffmpeg's `hevc_nvenc` on the 4090).

### Sources

- songho, OpenGL Pixel Buffer Object: the double-buffered readback.
- CUDA Programming Guide, graphics interoperability: a map makes GL's earlier writes visible to CUDA work on
  that stream; the unmap comes before GL uses the buffer again.
- NVENC Video Encoder API Programming Guide: NVENC takes NV12 from CUDA memory; its OpenGL input exists on
  Linux only.
- PyNvVideoCodec would encode straight from the CUDA ring with no trip through the CPU. The recorder pipes
  through ffmpeg instead: the encode runs after Stop, ffmpeg is already the encoder here, and it adds no
  package beside torch. It is the swap if the encode is too slow.

### Constraints from the alternatives

- The ring is CUDA memory, not GL textures: a GL ring would be read back on the render thread while the show
  runs, and the encode thread cannot touch GL.
- The encode runs after the recording, not live: live it would pull ~750 MB/s **(deduction)** across PCIe
  and through a pipe while the show runs.
- NV12 is made in a fragment shader, not a compute shader: GLSL has no byte arrays, and packing four pixels
  per integer would force widths to multiples of 4.
- The ring holds NV12, not RGBA: 2.7× the seconds, and the encoder makes 4:2:0 anyway.

### Open

- The time of the interop map per tick; the turn-taking pixel buffers are the first remedy.
- A hitch when a one-second block is allocated mid-recording; if ticks jump, allocate the whole ring before
  the first frame and keep the check as the early stop.
- The encode's load on the show (~12 MB per frame down to the CPU and up to NVENC); if the frame rate drops
  while encoding, cap the encode's speed.
- Encode time for 10 s of UHD at 60 ticks/s: 5 to 10 s **(deduction)**.
