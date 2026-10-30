# OAK4

For whoever sets up, tunes or changes the OAK4 camera path. This document says what the camera delivers, at what latency, and why it is configured this way. Deep Flow is the app it was measured for. The backend's structure is in the docstrings of [camera.py](camera.py) and [pipeline.py](pipeline.py).

Every number assumes this configuration unless its table says otherwise:

| Item     | Value                                                   |
|----------|---------------------------------------------------------|
| Camera   | OAK-4-PRO-W, serial 2856515419, RVC4                    |
| Sensors  | IMX586 color on CAM_A, OV9282 stereo on CAM_B and CAM_C |
| Software | Luxonis OS 1.45.0, depthai 3.11                         |
| Link     | USB, reading 5 Gbps                                     |
| Frame    | `P720`, 60 fps, 896x720 NV12, undistorted               |
| Boxes    | `pose_chain` `DETECTOR`: yolov8n, 416x416               |
| Depth    | `depth_model` `NEURAL_480`, aligned to 374x300          |
| PC       | i7-13700K with UHD 770, RTX 4090                        |

## Operating

### First setup

1. Build the second venv with `scripts\install.bat oak4` (see the [README](../../../README.md)).
2. Put the model archives in `data/models`. `.venv-oak4\Scripts\python -m modules.camera.oak4.models` fetches the pose models; the detector archive is compiled through HubAI ([definitions.py](definitions.py), `PERSON_DETECTOR_ARCHIVE`).
3. Update the camera to Luxonis OS 1.45.0 with `oakctl device update`. On 1.25.3 depthai could not start a session over USB. The `--info` form of that command needs `-y`, or it waits on a prompt.
4. Connect USB. A camera on USB and Ethernet at once is listed twice under one id, so `link` picks the route; when that route is missing, the camera opens on the other with a warning.

Start an app from the venv: `.venv-oak4\Scripts\python launcher.py -app deep_flow`.

### Reading the panel

The pinned readouts measure capture to arrival on the PC.

| Readout       | Expected    | When it is off                                |
|---------------|-------------|-----------------------------------------------|
| Video         | 60 fps      | The camera or the link is not keeping up      |
| Video latency | 28 to 29 ms | Check the link: Ethernet adds about 6 ms      |
| Depth latency | 42 to 52 ms | Moves between phases of a run; this is normal |

### When the camera is not found

The camera keeps trying by itself, so a reopen right after closing the app needs no action. A problem is logged once, and `delivers frames again` marks its end. While the camera is closed its rates and latencies read 0.

| Situation                              | What the camera does               |
|----------------------------------------|------------------------------------|
| No camera found, or it will not open   | Tries again every 1 s              |
| No frame for 5 s while open            | Closes, then tries again every 1 s |
| No first frame within 15 s of opening  | Closes, then tries again every 1 s |
| The pipeline fails on an opened camera | Logs an error once and stops       |

A quick reopen fails at the search, with `no OAK4 found`. In one reopen, 4.6 s after the camera closed, it stayed out of the search for about 8 s and delivered frames 12.4 s after the close.

The last case is a broken setup, such as a missing model archive. A retry changes nothing there, so the log line `pipeline failed, not trying again` means the setup must be fixed.

### In a dark room

Turn on `sensor.flood_light`. Without IR light there is almost no depth (see [IR light](#ir-light)). The color sensor does not see IR, so the image does not change.

### Tilt

`tilt.enabled` straightens color and depth for the camera's own tilt, read from its IMU. `tilt.offset` adds to that tilt, from -60° to 60°. It trims the correction, and on a level camera it tests the correction: dragging it warps color and depth as if the camera were tilted. A change takes effect within 0.2 s.

### Recording

Deep Flow records chunks of 60 s (`split_seconds`), each one `.mkv` holding color and depth. Play them with `-sim`. Frames are stored mirrored and not straightened, so `tilt.offset` can still be tuned on a recording.

### Measuring

Measure before changing the path. A held frame rate says nothing about latency.

```
.venv-oak4\Scripts\python -m modules.camera.oak4.probe --color --resolution P720 --fps 60 --detector --depth NEURAL_480
```

## Latency

Latency is `dai.Clock.now()` minus the message's timestamp, on depthai's clock, because the message's timestamp is. It is the one duration in the app not on `perf_counter`.

### What arrives

| Stream | Size          | Rate   | Capture to PC        |
|--------|---------------|--------|----------------------|
| Color  | 896x720       | 60 fps | 28 to 29 ms          |
| Boxes  | 416x416 input | 60 fps | 35 ms                |
| Depth  | 374x300       | 60 fps | 42 to 52 ms, mean 46 |

The worst gap between depth frames is 27.5 ms, measured at 900x720 without undistortion. Depth arrives about 18 ms after color, so a moving body's depth trails its image by about a frame **(deduction)**.

### Color path

Capture to PC, mean ms (std), 720x720 at about 59 fps with detector and tracker, without undistortion:

| Path                             | Ethernet   | USB        |
|----------------------------------|------------|------------|
| BGR, then `ImageManip`           | 42.7 (1.0) | 33.9 (0.7) |
| NV12, cropped by the Camera node | 28.8 (0.3) | 24.9 (0.5) |

A color frame takes the NV12 path, with no `ImageManip`. Undistortion adds about 3 ms to it, and [the frame](#the-frame) says why it stays on.

Compare latencies only on one OS version. On OS 1.25.3 NV12 over Ethernet measured 21.9 ms, and the 7 ms difference is unexplained.

### Depth methods

Measured beside color at 25 ms, with the detector, aligned to 360 rows:

| Method               | Input    | Rate     | Depth (ms) | Boxes (ms) | Note     |
|----------------------|----------|----------|------------|------------|----------|
| Neural Depth         | 288x180  | 60 fps   | 26 to 28   | 26 to 28   |          |
| Neural Depth         | 384x240  | 60 fps   | 31 to 41   | 26 to 28   |          |
| Neural Depth         | 480x300  | 60 fps   | 42 to 44   | 35         | Chosen   |
| Neural Depth         | 576x360  | 46 fps   | 139        |            | Too slow |
| Neural Depth         | 768x480  | 26 fps   |            |            | Too slow |
| GPU stereo           |          | 60 fps   | 34         |            | Holes    |
| Stereo FAST_DENSITY  |          | 60 fps   | 36 to 51   |            | Blobby   |
| Stereo FAST_ACCURACY | 640x400  | 60 fps   | 48         |            |          |
| Any stereo           | 1280x800 | Under 60 |            |            | Too slow |

480x300 is the largest size that holds 60 fps. Taking the detector off the camera does not speed depth up. Larger stereo input than 640x400 gained nothing and dropped frames.

Published by Luxonis for RVC4, for comparison:

| Method                 | Size     | Rate   | Latency |
|------------------------|----------|--------|---------|
| Neural Depth           | 480x300  | 60 fps | 50 ms   |
| Neural Depth           | 384x240  | 60 fps | 40 ms   |
| Stereo FAST_ACCURACY   | 640x400  | 60 fps | 35 ms   |
| Stereo FAST_DENSITY    | 640x400  | 60 fps | 39 ms   |
| Stereo FAST_ACCURACY   | 1280x800 | 53 fps | 44 ms   |
| Neural Assisted Stereo | 1280x800 | 54 fps | 71 ms   |

### Work on the PC

The camera thread does no color work: `Uploader` converts and flips on the GPU. On an idle GPU:

| Step        | On the GPU | On the CPU                        |
|-------------|------------|-----------------------------------|
| NV12 to RGB | 0.25 ms    | 0.65 ms, `getCvFrame` plus upload |
| Flip        | 0.02 ms    | 0.5 ms, numpy                     |

The conversion is limited-range BT.601 with nearest chroma, which matches `getCvFrame` to 0.32 of 255. Full range is 8.5 off.

### Link

USB is the link to use. Ethernet carries the whole pipeline at 60 fps, 6 ms later on color, but it stalled in one run of five. Capture to PC in ms, on the probe, with no player in view, in a dark room, with `-sim` running on the PC. Color and boxes are means over the second half of a run; the worst gap covers color and depth.

| Link     | Run  | Color | Boxes | Depth mean | Depth max | Worst gap |
|----------|------|-------|-------|------------|-----------|-----------|
| USB      | 20 s | 28.6  | 34.5  | 42.6       | 48.9      | 32.1      |
| Ethernet | 20 s | 35.1  | 35.9  | 42.6       | 48.2      | 33.0      |
| Ethernet | 20 s | 108.3 | 109.2 | 128.2      | 1676.7    | 4763.0    |
| Ethernet | 30 s | 34.8  | 35.8  | 43.7       | 49.3      | 39.7      |
| Ethernet | 30 s | 34.7  | 35.2  | 47.9       | 52.1      | 35.7      |
| Ethernet | 30 s | 34.8  | 35.4  | 48.4       | 51.3      | 41.9      |

The load on the link:

| Stream                | Rate                      |
|-----------------------|---------------------------|
| Color, 896x720 NV12   | 464 Mbit/s before padding |
| Depth, 374x300 uint16 | 108 Mbit/s                |
| Together              | 572 Mbit/s                |
| Ethernet, expected    | About 800 Mbit/s          |

The stream rates are calculated from the frame sizes **(deduction)**, before the chip's row padding. The Ethernet figure is from the Luxonis docs.

## The frame

The camera delivers one shape, 5:4 at the sensor mode's rows, so the depth model's 8:5 frame is used further across than a square would. There is no setting for it. The width is a multiple of 16 (`FRAME_WIDTH_ALIGNMENT`) because the iGPU's H.264 decoder hangs on any other width: 900 hangs, 896 and 720 play. A frame of another width cannot be replayed.

Undistortion stays on because aligned depth is always undistorted. Without it, depth misses the frame by about 14 of 360 pixels at its edges. The tilt correction needs it too, since a homography only holds on a rectilinear image.

Lens of the delivered frame, as tagged on a recording, in pixels:

| fx    | fy    | cx    | cy    |
|-------|-------|-------|-------|
| 538.1 | 538.0 | 449.3 | 369.8 |

That gives a view of about 79.6° by 67.6° **(deduction)**. The color sensor spans 95° and the stereo pair 127° (Luxonis docs), so the whole frame has depth. Depth has about 291 rows of real detail inside the 720 rows, since the frame spans about 775 of the 800 mono rows **(deduction)**.

`HORIZONTAL_MIRROR` has no effect on this camera: the mirrored frame matches the plain one. So the PC mirrors, and the backend mirrors boxes and keypoints to match. There is no vertical flip for color: `flip_v` logs a warning and does nothing.

## Depth

Neural Depth runs on the camera off 640x400 mono frames, and `ImageAlign` puts it on the delivered frame. Depth is uint16 millimeters, with 0 for no depth. The black shadow bands beside near objects are stereo occlusion, which alignment does not remove.

`DepthSourceLayer` turns it into float32 meters, straightens it, and exposes it as `depth` for use as data; `texture` is its view. The view's colormap matches depthai's `colorizeDepthFrame` defaults: JET on a log scale from 0.3 to 12 m, which puts the detail close to the camera.

### IR light

The camera has two IR lights, `dot_projector` and `flood_light`. Dark room, full brightness, 600 frames, depth read back from a recording:

| Light | Valid pixels | Worst frame | Median step | p95 step | Flicker |
|-------|--------------|-------------|-------------|----------|---------|
| None  | 7.4 %        | 1.1 %       | 51 mm       | 276 mm   | 7.75 %  |
| Dots  | 76.5 %       | 69.6 %      | 36 mm       | 177 mm   | 8.33 %  |
| Flood | 94.4 %       | 90.8 %      | 36 mm       | 252 mm   | 3.20 %  |
| Both  | 95.4 %       | 91.4 %      | 24 mm       | 207 mm   | 2.63 %  |

A step is the change of a pixel's depth from one frame to the next, where both frames have depth. Flicker is the share of pixels that gain or lose depth between two frames. Flood does nearly all the work on valid pixels and flicker; adding the dots lowers the median step.

## Tilt correction

For a camera above a big screen, looking down. A tilt is a pure rotation about the camera, so where a pixel lands does not depend on its distance. One homography therefore serves color, depth and boxes, with no point cloud and no holes.

`H = K' R K⁻¹`, normalized uv to uv (`tilt_homography` in [geometry.py](../geometry.py)). `K` is the delivered lens and `R` the tilt. `K'` refits the picture to the frame, because plain `K` at 30° moves the frame's center by 0.58 focal lengths, about 310 of 720 rows.

Depth pixels move by `H`, and their values change too. Depth is measured along the tilted axis, so a standing player of 1.8 m at 30° reads as leaning by 0.9 m. The corrected depth is `z' = z × w`, with `w` from `H`'s third row, the distance along the level axis.

| Part  | Where it happens                          |
|-------|-------------------------------------------|
| `H`   | `TiltCorrection`, at the IMU's 5 Hz       |
| Color | `Uploader`, `grid_sample`                 |
| Depth | `DepthWarp` shader, nearest sample        |
| Boxes | `corrected_roi` and `corrected_keypoints` |

`H` is rebuilt only when the angle moves by 0.25° or more (`TILT_STEP`), since a picture that followed the noisy IMU would shimmer. The correction runs on the PC for two reasons: the chip warps grey frames only (Luxonis docs), and `ImageManip`'s four-point transform needs the BGR path (see [color path](#color-path)).

The whole picture stays in the frame, which leaves wedges without picture at the top corners. The feet end is enlarged most, so it loses the most detail **(deduction)**. The camera's height above the screen is not corrected in the image, since that would need reprojection and leave holes.

## Recording and replay

The format is in [chunk.py](../chunk.py) and [depth_frame.py](../../video/depth_frame.py). Each chunk is one self-contained `.mkv`: stream 0 color, stream 1 depth, fixed info in the tags and per-frame readings attached. Recording starts at the first depth frame, and each color frame takes the newest depth, so both streams have equal frame counts. This is the pairing the app shows live.

Encoding runs on the iGPU because it does nothing else in the installation **(site fact)**, so recording stays out of the way of the show.

| Stream | Codec       | Encoder    | Format                                    |
|--------|-------------|------------|-------------------------------------------|
| Color  | H.264       | `h264_qsv` | `video_quality` 24, `keyframe_interval` 0 |
| Depth  | 12-bit HEVC | `hevc_qsv` | `p012le`, 3 mm steps, 12 m full scale     |

Depth is lossy. That is acceptable because the fluid spreads and smooths its input and has only 20 to 32 depth layers, so a few millimeters should not show **(deduction)**. On synthetic frames 99 % of pixels are within 11 mm, with a steady offset of about 6 mm. Set depth quality with `-q:v`; `global_quality` breaks at low values, with errors in meters.

### File size

| Source                | `video_quality` | Color    | Depth    | Together |
|-----------------------|-----------------|----------|----------|----------|
| Recorded, 18.1 s      | 18              | 7.4 MB/s | 1.8 MB/s | 9.2 MB/s |
| Same clip, re-encoded | 24              | 1.4 MB/s | 1.8 MB/s | 3.2 MB/s |

The clip is a dark room, and the color rate at 18 is mostly sensor noise **(deduction)**.

### Encoder load

Synthetic frames, nothing written, on the configuration's PC. Speed is a multiple of real time. These are easy test patterns, so real footage runs slower and larger.

| Stream                  | Encoder     | Speed | Load                      |
|-------------------------|-------------|-------|---------------------------|
| Color, 900x720          | H.264, iGPU | 5.4x  | 18 % of the iGPU encoder  |
| Color + depth, 900x1020 | H.264, iGPU | 5.2x  | 19 %                      |
| Depth, 376x300, 10-bit  | HEVC, iGPU  | 5.9x  | 17 % more                 |
| Depth, 376x300, 12-bit  | HEVC, iGPU  | 5.1x  | 20 % more                 |
| Depth, 376x300, 16-bit  | FFV1, CPU   | 30x   | 13 % of one of 24 threads |

FFV1 is the fallback, exact and on the CPU. The second row is one H.264 picture holding both, which limits depth to 8 bits at a range fixed when recording.

### On the real camera

A 10 s recording held 60 fps with the recorder's queues at 1 or less, and gave 600 frames in both streams with 600 metadata lines. The iGPU decodes both streams together at more than 670 fps color and 1700 fps depth.

### Replay

`replay.Camera` emits both streams and rebuilds `H` from the tagged lens and each frame's recorded tilt, plus the live `tilt.offset`. The recorded frames are already mirrored, so the lens's `cx` is mirrored for this and `flip_h` stays off.

## Open

- Video min sometimes reads about 30 fps **(site fact)**. The cause is not known.
- `center_depth` has not been seen with a player in view.
- A standing player reading as one distance from head to feet has not been checked.
- Depth has one outlier of about 153 mm near the top left on synthetic frames, at all widths. The cause is not known.
- `StreamReader.frame_count` can read one short on an `.mkv`, as it estimates from the duration.
- Ethernet stalled once in 130 s of streaming, with a gap of 4.8 s and latency up to 1.7 s. The cause is not known, and a USB run of that length has not been compared.
- A player at show distance has not been compared across depth sizes or IR lights.
