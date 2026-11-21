# Screen recording

Render's part of the screen recording: what the compositor draws into the recording's frame, the capture that turns
it into YUV planes in RAM on the main thread, and the preview that draws a held frame back. The recorder that stores
the frames, holds the take and encodes it is `modules/video`'s ([VIDEO.md](../../../video/VIDEO.md#screen-recorder)).

## What the compositor records

`ScreenRecordingCompositor` (`screen_recording_compositor.py`) owns the `ScreenCapture` and the
`ScreenRecorder`. Each tick, right after the main window is drawn, the app hands it the layers to record with
their shapes (`ScreenRecordingSource`), the window's size and the tick's time; deep_flow and low_flow hand it
the main panel's layers (`layer.main`). It draws them into `ScreenCapture.frame`, each fit and centered at its
own shape and blended the way the show draws them (additive on black), so a recording is the show. With
`main_window` the window's back buffer is copied into the frame instead (`ScreenCapture.copy_window`), before
the window is shown. A module cannot know an app's layer enum, so the app chooses the layers and the
compositor takes them as layer objects.

A source whose shape differs from the recording's is fit with black bars, or with `fill` fills the frame and
its overhang is cut. It is never stretched.

## Capture, on the main thread

- The RgbToYuv shader (`shaders/rgb_to_yuv.py`, `rgb_to_yuv.frag`) converts the
  screen frame into its planes, one pass each (`YuvPlane`): Y into a one-channel texture at full size; at 4:2:0 Cb Cr
  into a two-channel texture at half size, each chroma sample the average of its 2 × 2 pixels; at 4:4:4 Cb and Cr
  each into a one-channel texture at full size; at 4:4:4 and 10 bits Cb, Y and Cr into one RGB10_A2 texture (red,
  green, blue), which reads back as `xv30le`. 4:2:0 at 10 bits uses 16-bit textures (P010). It samples through
  `position.glsl`'s `topLeft`, so the first row read back is the picture's top. The BT.709 matrix is plain code
  beside the shader (`bt709_video_range`), scaled to the plane's sample size (255, 1023 or 65535).
- `glReadPixels` reads the planes, Y first, into one of four pixel buffers in host memory, each mapped once
  (persistent and coherent) when the capture is made, and a fence is set after it. The GPU writes the frame into
  RAM; the CPU does not wait.
- `poll`, on every tick, checks the oldest fences without waiting; once one has passed, the frame goes to the
  callbacks (`add_screen_frame_callback`, a list with its own lock) as a `ScreenFrame`: the mapped buffer as an
  array, its tick's `timestamp`, and the ticks skipped before it. The buffer is lent until the frame's `release`.
  A frame enters the recorder's store one to two ticks after it was drawn.
- With all four buffers lent the capture leaves the tick out; the next frame carries the count (`skipped`).
- The compositor makes the capture in the recorder's format when a take begins (`ScreenRecorder.capture_format`) and
  frees it after; never while a frame is lent or a read-back is on its way, as its memory would go under a reader.
- The capture sets its own blend and viewport and leaves framebuffer 0 bound. Nothing is drawn or converted unless
  the recorder captures this tick (`ScreenRecorder.capturing`).
- No CUDA: the read-back is GL's own, and a mapped buffer in host memory reads as plain RAM. `scripts/screen_capture_check.py`
  checks the YUV codes of known colors and the preview in both chromas at both bit depths, a 10-bit gray ramp
  (877 Y codes back, 75 % off the 8-bit steps), the pool's lending and skipping, and times one frame's copy out of
  a mapped buffer. These times mean the mapped memory is cached **(deduction)**.

| Format (2160 × 3840, 20 copies) | Copy out of a mapped buffer (ms) |
|---------------------------------|----------------------------------|
| 4:2:0, 8 bits                   | 0.31                             |
| 4:4:4, 8 bits                   | 0.70–0.90                        |
| 4:4:4, 10 bits                  | 1.27                             |

`scripts/screen_capture_check.py`, partly from the CPU's cache **(measured 2026-11-20)**.

## The preview

While a marker of the held take is dragged, the main window shows the held frame nearest to it
(`ScreenRecorder.preview_frame`), fit on black, so a cut is made on what is seen; `play` shows the take through it
too ([The held take](../../../video/VIDEO.md#the-held-take)). `ScreenPreview` (`screen_preview.py`) draws it without
the main thread waiting: the render tick stores the frame under the marker (`set_screen_frame`, a held slot no one
changes), and the preview's own thread copies it straight into one of two unpack pixel buffers, mapped once on the
main thread (persistent and coherent), and marks it written. The render tick uploads the newest written buffer into
the planes' textures, turns them back into RGB in `ScreenPreview.fbo` with the `YuvToRgb` shader
(`shaders/yuv_to_rgb.py`), upright, and sets a fence on the buffer; once the fence has passed, checked on a later
tick without waiting, the buffer is free again. The preview's thread makes no GL call. No CUDA. A drag shows its
frames while a save runs, as the recorder's thread takes no part.

| Held 10 s take        | Ticks | Pictures drawn | Mean (ms) | Max (ms) | Over 25 ms |
|-----------------------|-------|----------------|-----------|----------|------------|
| Dragged for 3 s       | 181   | 142            | 16.66     | 33       | 11 %       |
| Held, no drag         |       |                |           |          | 8 %        |

deep_flow `-sim`, UHD portrait at 4:2:0, 60 Hz with v_sync, commit 7e32c69a **(measured 2026-11-20)**.

## Sources

- songho, OpenGL Pixel Buffer Object: the asynchronous read-back into a pixel buffer.
- OpenGL 4.4 buffer storage (`glBufferStorage`, persistent and coherent mapping) and sync objects
  (`glFenceSync`, `glClientWaitSync` with a timeout of 0): a mapped buffer read on another thread once its fence
  has passed.

## Constraints from the alternatives

- The planes are made in a fragment shader, not a compute shader: GLSL has no byte arrays, and packing four pixels
  per integer would force widths to multiples of 4.
- 4:4:4 at 10 bits is packed in one plane, not three 16-bit planes: the read-back runs on the GPU's graphics queue
  at about 12 GB/s, so its time follows its bytes, and three 16-bit planes slow the show.

| 4:4:4 read-back at UHD       | Bytes a frame | GPU a frame (ms) | Mean tick (ms) | Dropped in 15 s |
|------------------------------|---------------|------------------|----------------|-----------------|
| 8 bits                       |               | 2.1              |                |                 |
| 10 bits, one packed plane    | 33.2 MB       | 2.6              |                |                 |
| 10 bits, three 16-bit planes | 49.8 MB       | 4.1              | 19.0           | 6               |

GL timer queries **(measured 2026-11-20)**.
- The packed 10-bit RGB formats NVENC takes (`x2rgb10le`) are converted inside it to 4:2:0 with the BT.601 matrix,
  so the shader makes the YUV itself at 10 bits too **(measured 2026-11-20, ffmpeg 2025-08-25)**.
