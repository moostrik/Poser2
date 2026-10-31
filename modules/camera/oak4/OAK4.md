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
| Boxes    | `model_chain` `DETECTOR`: yolov8n, 640x512              |
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
| Link          | USB         | Fell back to Ethernet; see [Open](#open)      |

The camera's `readings` show its own `cpu`, `memory` and `temperature` on the line under `video_latency`, once a second (see [camera load](#camera-load)).

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

### Sensor limits

Measured on the color sensor at P720 and 60 fps, by asking for manual values and reading back what the frames report:

| Control         | Low   | High     | Note                         |
|-----------------|-------|----------|------------------------------|
| Manual exposure | 86 µs | 33000 µs | High end from the docs       |
| Manual ISO      | 65    | 16585    | Higher values clamp to 16585 |
| Auto exposure   |       | 16277 µs | At 60 fps                    |
| Auto ISO, seen  | 65    | 5832     | Holds 5832 in a dark room    |

Whether manual exposure really goes past the frame time is in [Open](#open).

### Auto exposure

Auto exposure is on by default, and these settings in `sensor` shape it, live. Measured in a dark room, where it sits at 16277 µs and ISO 5832:

| Setting                     | ISO   | Mean luma |
|-----------------------------|-------|-----------|
| Defaults                    | 5832  | 24.0      |
| `auto_iso_limit` 1000       | 998   | 10.8      |
| `exposure_compensation` -9  | 2183  | 15.4      |
| `exposure_compensation` +9  | 16585 | 42.4      |

`exposure_region` is in fractions of the picture as shown, mirrored or not. The other new settings in `sensor` are `auto_exposure_lock`, `auto_balance_lock`, `anti_banding` and `chroma_denoise`; what is known about them is in [Open](#open).

### Tilt

`tilt.enabled` straightens color and depth for the camera's own tilt, read from its IMU. `tilt.offset` adds to that tilt, from -60° to 60°. It trims the correction, and on a level camera it tests the correction: dragging it warps color and depth as if the camera were tilted. A change takes effect within 0.2 s.

### Recording

Deep Flow records chunks of 60 s (`split_seconds`), each one `.mkv` holding color and depth. Play them with `-sim`. Frames are stored mirrored and not straightened, so `tilt.offset` can still be tuned on a recording.

### Measuring

Measure before changing the path. A held frame rate says nothing about latency. With no flags the probe runs the setup at the top of this document.

```
.venv-oak4\Scripts\python -m modules.camera.oak4.probe
```

## Latency

Latency is `dai.Clock.now()` minus the message's timestamp, on depthai's clock, because the message's timestamp is. It is the one duration in the app not on `perf_counter`.

### What arrives

| Stream | Size          | Rate   | Capture to PC        |
|--------|---------------|--------|----------------------|
| Color  | 896x720       | 60 fps | 28 to 29 ms          |
| Boxes  | 640x512 input | 60 fps | 35 ms                |
| Depth  | 374x300       | 60 fps | 42 to 52 ms, mean 46 |

The worst gap between depth frames is 27.5 ms, measured at 900x720 without undistortion. Depth arrives about 18 ms after color, so a moving body's depth trails its image by about a frame **(deduction)**.

### Detector size

The detector is 640x512 so a frame goes in at its own 5:4 shape and a player keeps the most rows. The sizes cost the same. Beside color and depth, about 4 s each, nobody in view, dark room with the flood light on:

| Detector | Color   | Boxes   | Depth   | Rate   |
|----------|---------|---------|---------|--------|
| 416x416  | 28.8 ms | 34.7 ms | 44.5 ms | 60 fps |
| 480x384  | 28.2 ms | 35.4 ms | 41.7 ms | 60 fps |
| 512x416  | 28.4 ms | 34.8 ms | 41.6 ms | 60 fps |
| 640x512  | 28.7 ms | 35.0 ms | 40.2 ms | 60 fps |

### Frame size

`resolution` sets the delivered frame on USB and `ethernet_resolution` the one on Ethernet. The backend reads the link the camera actually opened on, so a fallback to the other link gets that link's frame. Deep Flow runs `P1080` on USB and `P800` on Ethernet.

With the 640x512 detector and depth, on the probe, about 3 s each on USB and 10 s on Ethernet:

| `resolution` | Link     | Frame     | Color   | Boxes   | Depth   | Rate   | NV12 on the link |
|--------------|----------|-----------|---------|---------|---------|--------|------------------|
| `P720`       | USB      | 896x720   | 28.9 ms | 35.5 ms | 43.4 ms | 60 fps | 464 Mbit/s       |
| `P800`       | USB      | 992x800   | 31.1 ms | 36.1 ms | 39.4 ms | 60 fps | 571 Mbit/s       |
| `P800`       | Ethernet | 992x800   | 37.0 ms | 37.2 ms | 46.0 ms | 60 fps | 571 Mbit/s       |
| `P1080`      | USB      | 1344x1072 | 37.7 ms | 39.2 ms | 43.5 ms | 60 fps | 1037 Mbit/s      |

The link rates are calculated from the frame sizes **(deduction)**, before the chip's row padding. With depth, `P800` comes to 679 Mbit/s and `P1080` to 1145 Mbit/s, so `P1080` does not fit Ethernet's [expected 800 Mbit/s](#link).

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

## Camera load

One sample each of `oakctl device stats`: idle, and with Deep Flow running on the camera over USB.

| Reading            | Idle        | Deep Flow   |
|--------------------|-------------|-------------|
| CPU, all 6 cores   | 15.7 %      | 48.3 %      |
| Memory, of 7.55 GB | 0.81 GB     | 1.36 GB     |
| Power, camera      | 3.4 W       | 14.8 W      |
| Power, chip        | 3.0 W       | 7.8 W       |
| NPU temperature    | 28 to 31 °C | 51 to 57 °C |
| Hottest zone       | 37.5 °C     | 62.2 °C     |

The panel's `cpu`, `memory` and `temperature` come from depthai's `SystemLogger` node, which runs in the pipeline. It reports no power, no NPU load, and no NPU temperature; those need `oakctl`.

## Player distance

A player's distance can come from the camera's `SpatialDetectionNetwork` or from the PC, as the median of the depth in the middle half of the box. The camera's node costs almost nothing, and the two agree: over 1803 tracklets the camera's `z` minus the PC's median was -1 mm at the median, and 38 mm at p95.

Capture to PC in ms, 30 s each over USB, in a dark room with `flood_light` at full and no player in view. The rows with boxes set the detector's confidence to 0.02, so it reports about one spurious box a frame and the node has work to do. Worst gap covers color, boxes and depth.

| Detector                  | Boxes in frame | Color | Boxes | Boxes max | Depth | Worst gap | Camera CPU |
|---------------------------|----------------|-------|-------|-----------|-------|-----------|------------|
| `DetectionNetwork`        | None           | 29.8  | 36.2  | 53.1      | 45.6  | 36.3      | 48.8 %     |
| `SpatialDetectionNetwork` | None           | 28.5  | 36.3  | 59.7      | 43.3  | 37.2      | 50.3 %     |
| `DetectionNetwork`        | About 1        | 29.1  | 36.4  | 56.6      | 42.9  | 34.7      | 49.5 %     |
| `SpatialDetectionNetwork` | About 1        | 29.2  | 36.7  | 63.0      | 43.9  | 43.3      | 50.4 %     |

The node logs a warning on every frame, because the detector's stretched frame and the depth have different transforms and it remaps points. Its `z` is along the tilted axis. Recordings hold no camera `z`, so `-sim` needs the PC's reading either way.

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

### Depth filters

`depth` in the camera's settings holds Neural Depth's filters, live. A pixel that fails one reads 0. Dark room, flood light at full, a still scene, about 2 s per row, read at the backend's hand-over:

| Setting              | Rate   | Latency | Valid  | Median step | p95 step | Flicker |
|----------------------|--------|---------|--------|-------------|----------|---------|
| Defaults             | 60 fps | 44 ms   | 93.8 % | 38 mm       | 255 mm   | 3.0 %   |
| `confidence` 200     | 60 fps | 44 ms   | 61.5 % | 25 mm       | 134 mm   | 10.7 %  |
| `confidence` 50      | 60 fps | 44 ms   | 99.1 % | 43 mm       | 389 mm   | 0.3 %   |
| `edge` 100           | 60 fps | 44 ms   | 94.3 % | 38 mm       | 265 mm   | 3.4 %   |
| `temporal_alpha` 0.4 | 60 fps | 44 ms   | 95.6 % | 12 mm       | 86 mm    | 1.3 %   |
| `temporal_alpha` 0.1 | 60 fps | 45 ms   | 95.7 % | 3 mm        | 20 mm    | 1.3 %   |

The two temporal rows have `temporal_filter` on. `near` and `far` cut exactly: at 2 m the depth read 2000 mm and over, or 2000 mm and under. `far` at its top sets no limit.

The filter's latency is the latency of a message: a moving body's depth lags by more, since each pixel is an average of past frames **(deduction)**.

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

| Stream | Codec       | Encoder    | Format                                |
|--------|-------------|------------|---------------------------------------|
| Color  | H.264       | `h264_qsv` | `video_quality` 24                    |
| Depth  | 12-bit HEVC | `hevc_qsv` | `p012le`, 3 mm steps, 12 m full scale |

`keyframe_interval` is 60 on both.

Both streams share `keyframe_interval`, and depth has no B-frames, so a seek finds both streams whole. A seek in a chunk goes to a color keyframe, and depth without a keyframe there comes back about 100 mm off. With B-frames depth was also off after a shared keyframe; why is not settled. At 60, 15 of 15 seeks on a real recording matched reading from the start, in both streams, at about 185 ms a frame. A recording made with `keyframe_interval` 0 keeps the fault in its depth.

Depth is lossy. That is acceptable because the fluid spreads and smooths its input and has only 20 to 32 depth layers, so a few millimeters should not show **(deduction)**. On synthetic frames 99 % of pixels are within 11 mm, with a steady offset of about 6 mm. Set depth quality with `-q:v`; `global_quality` breaks at low values, with errors in meters.

### File size

| Source                | `video_quality` | `keyframe_interval` | Color    | Depth    | Together |
|-----------------------|-----------------|---------------------|----------|----------|----------|
| Recorded, 18.1 s      | 18              | 0                   | 7.4 MB/s | 1.8 MB/s | 9.2 MB/s |
| Same clip, re-encoded | 24              | 0                   | 1.4 MB/s | 1.8 MB/s | 3.2 MB/s |
| Same clip, re-encoded | 24              | 60                  | 1.8 MB/s | 2.1 MB/s | 3.9 MB/s |
| `P1080`, recorded     | 24              | 60                  |          |          | 10 MB/s  |

The first three rows are `P720`, and the third is the current setup at that size. The last row is the current setup at `P1080`, and a 60 s chunk comes to about 600 MB. All rows are a dark room, and the color rate at 18 is mostly sensor noise **(deduction)**.

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

A 10 s recording at `P720` and one at `P1080` each held 60 fps with the recorder's queues at 1 or less, and gave 600 frames in both streams with 600 metadata lines. At `P1080`, color was 36.8 ms and depth 46.9 ms while recording.

The iGPU decodes both `P720` streams together at more than 670 fps color and 1700 fps depth. It plays both `P1080` streams together at real time, 600 frames each, and a seek to frame 240 matched the full read exactly in both.

### Replay

`replay.Camera` emits both streams and rebuilds `H` from the tagged lens and each frame's recorded tilt, plus the live `tilt.offset`. The recorded frames are already mirrored, so the lens's `cx` is mirrored for this and `flip_h` stays off.

Dragging a marker on the player's bar pauses playback and shows the frame under it, the newest position as fast as a frame can be fetched. Letting go plays from the in marker. Each recording keeps its markers in `in_out.json` in its folder.

### Detector in replay

Replay finds players on the PC with [`PersonDetector`](../../inference/detection/person_detector.py). Its `resolution` is set to match `PERSON_DETECTOR_ARCHIVE` on the camera, and its `model_type` picks ONNX Runtime or TensorRT. Time per frame on an empty `P1080` frame, 500 runs each, batch 1, FP32, GPU otherwise idle:

| `resolution` | ONNX, back to back | TensorRT, back to back | ONNX, at 60 Hz | TensorRT, at 60 Hz |
|--------------|--------------------|------------------------|----------------|--------------------|
| `R640X512`   | 5.6 ms             | 4.6 ms                 | 10.1 ms        | 9.0 ms             |
| `R512X416`   | 4.6 ms             | 4.0 ms                 | 8.4 ms         | 8.1 ms             |
| `R480X384`   | 4.3 ms             | 3.6 ms                 | 8.2 ms         | 7.4 ms             |

The times cover preprocessing on the CPU, inference and decoding. At 60 Hz the GPU drops to idle clocks between runs, which nearly doubles them, and the slowest frame at `R640X512` was 20.1 ms on both runtimes. Back to back, the slowest frame at any size was about 8 ms. Both runtimes find the same players on a photo of people, with scores within 0.001 and boxes within 0.03 px. ONNX Runtime stays the default, as TensorRT saves about 1 ms and takes turns with the other TensorRT models on their shared execution lock.

## Open

- NPU load is not read. depthai does not report it, and `oakctl device stats` gives `dsp` null with a pipeline running. `sysMonApp` on the camera reads the NPU through `/dev/fastrpc-cdsp`, which only the `system` user can open, and the ADB shell runs as `adb`, so it fails with a FastRPC error. It needs a root shell through `oakctl device unlock`, which has not been tried.
- depthai once stopped listing the camera on USB after an app closed, and kept listing it on Ethernet, while ADB and Windows still saw it on USB. Every open fell back to Ethernet until `oakctl device reboot` brought USB back. The cause is not known.
- `SpatialDetectionNetwork` has not been measured with a real player, whose larger box holds more depth pixels, or with several players. Which depth frame its `z` comes from is not known: boxes reach the PC before that frame's depth does.
- Video min sometimes reads about 30 fps **(site fact)**. The cause is not known.
- `center_depth` has not been seen with a player in view.
- A standing player reading as one distance from head to feet has not been checked.
- Depth has one outlier of about 153 mm near the top left on synthetic frames, at all widths. The cause is not known.
- `StreamReader.frame_count` can read one short on an `.mkv`, as it estimates from the duration.
- Ethernet stalled once in 130 s of streaming, with a gap of 4.8 s and latency up to 1.7 s. The cause is not known, and a USB run of that length has not been compared. That was at `P720`; `P800` loads Ethernet more and has only run for 10 s.
- A player at show distance has not been compared across depth sizes or IR lights.
- Detection at 640x512 has not been compared with 416x416 on a player, and its latency has not been measured with a player in view.
- `P800` and `P1080` have not been run in the app, and `P800` has not been recorded and replayed.
- `auto_exposure_lock` did not hold: with it on, `exposure_compensation` +9 still moved ISO to 16585. Whether it holds against a change of light has not been tested.
- `exposure_region`, `auto_balance_lock` and `anti_banding` changed nothing in a dark room, where exposure was already at its ceiling and nothing flickered. They need a test with light. `chroma_denoise` 4 lowered mean luma from 24.0 to 22.1, so it does reach the sensor.
- The temporal filter has only been measured on a still scene. How far depth trails a moving player has not been measured.
- A manual exposure past the frame time is reported as asked and the rate stays at 60 fps, up to 100000 µs, which cannot be real. In a dark room the picture's brightness did not follow exposure at all, so the real top needs a test with light.
- `test_a_split_keeps_the_streams_in_step_across_chunks` fails about 2 runs in 5, because a split lands 1 to 3 frames early. No frames are lost: 50 of 50 in all three streams.
