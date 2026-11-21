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
| Depth    | `depth_model` `NEURAL_480`, 480x300, aligned on the GPU |
| PC       | i7-13700K with UHD 770, RTX 4090                        |

## Operating

### First setup

1. Build the second venv with `scripts\install.bat oak4` (see the [README](../../../README.md)).
2. Put the model archives in `data/models`. `.venv-oak4\Scripts\python scripts/oak4_models.py` fetches the pose models; the detector archive is compiled through HubAI ([models.py](models.py), `PERSON_DETECTOR_ARCHIVE`).
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
| Link          | USB         | Fell back to Ethernet                         |

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

### Auto exposure

Auto exposure is on by default, and these settings in `sensor` shape it, live. Measured in a dark room, where it sits at 16277 µs and ISO 5832:

| Setting                     | ISO   | Mean luma |
|-----------------------------|-------|-----------|
| Defaults                    | 5832  | 24.0      |
| `auto_iso_limit` 1000       | 998   | 10.8      |
| `exposure_compensation` -9  | 2183  | 15.4      |
| `exposure_compensation` +9  | 16585 | 42.4      |

`exposure_region` is in fractions of the picture as shown, after the frame's `rotate90` and flips (`sensor_region` takes it back through them). The other new settings in `sensor` are `auto_exposure_lock`, `auto_balance_lock`, `anti_banding` and `chroma_denoise`.

### Tilt

`tilt.enabled` straightens color and depth for the camera's own tilt, read from its IMU. `tilt.offset` adds to that tilt, from -60° to 60°. It trims the correction, and on a level camera it tests the correction: dragging it warps color and depth as if the camera were tilted. A change takes effect within 0.2 s.

### Recording

Deep Flow records chunks of 60 s (`split_seconds`), each one `.mkv` holding color and depth. Play them with `-sim`. Frames are stored mirrored and not straightened, so `tilt.offset` can still be tuned on a recording.

### Measuring

Measure before changing the path. A held frame rate says nothing about latency. With no flags the probe runs the setup at the top of this document.

```
.venv-oak4\Scripts\python scripts/oak4_probe.py
```

## Latency

Latency is `dai.Clock.now()` minus the message's timestamp, on depthai's clock, because the message's timestamp is.

### Capture time

Every frame and tracklet message is stamped with its moment of capture on `perf_counter` (`capture_clock.capture_time`): now less the message's latency. The message's timestamp is the end of the exposure. Color and depth of one capture carry the same time, so time pairs them (`CAPTURE_TOLERANCE`, 2 ms). Sequence numbers do not: depth's skip while every frame arrives.

Measured over 90 s, deep_flow's setup (P720 color NV12 at 60 fps, the detector, NEURAL_480, flood light 1.0), USB:

| Measure                                  | Result                                        |
|------------------------------------------|-----------------------------------------------|
| Depth against the nearest color capture  | median 0.11 ms, p95 0.12 ms                   |
| Depth without a color frame within 1 ms  | 4 of 5407, all in the first 30 ms after start |
| Tracklets against their color frame      | same time and sequence number, 5103 of 5103   |
| `perf_counter` minus `dai.Clock`         | −3 µs, std 6 µs, no drift                     |
| Frame interval, stamped at capture       | 16.17 to 16.78 ms                             |
| Frame interval, stamped at arrival       | color 0.5 to 33 ms, depth 2.6 to 28 ms        |
| Exposure, color (auto) / mono            | 0.3 to 16.3 ms / 3 ms                         |

The color and mono sensors are synchronized: they end their exposures together. The middle of the exposure lies half the exposure before the stamp, so a color frame's motion is centered up to 8 ms earlier than depth's at long exposures **(deduction)**. On Windows `dai.Clock` and `perf_counter` read one counter; `capture_time` does not depend on it.

depthai maps the device's clock onto the host's again about every 8 s, in steps of 0.01 to 0.24 ms, measured over three opens of 25 s from the first message. Once, in a run started seconds after another process had closed the camera, more than half the messages carried a capture time after now. That did not happen in five other opens and is not explained. A capture time after now cannot be, so `capture_time` stamps such a message with its arrival and logs a warning once.

### What arrives

| Stream    | Size          | Rate   | Capture to PC, median (p95) |
|-----------|---------------|--------|-----------------------------|
| Color     | 896x720       | 60 fps | 28.8 (31.1) ms              |
| Tracklets | 640x512 input | 60 fps | 35.4 (38.6) ms              |
| Depth     | 480x300       | 60 fps | 36.9 (46.8) ms, no detector |

Measured with the capture time above. The worst gap between depth frames is 27.5 ms, measured at 900x720 without undistortion. Depth arrives about 13 ms after the color frame captured with it and carries that frame's time.

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

`P2160` runs the color sensor's 4K mode and is for USB only; Ethernet takes `ethernet_resolution`. Color alone at 30 fps, no model and no depth, on the probe, 10 s each on USB, auto exposure at 31.9 ms in both:

| `resolution` | Link | Frame     | Color   | Rate   | Worst gap | Chip CPU | NV12 on the link |
|--------------|------|-----------|---------|--------|-----------|----------|------------------|
| `P1080`      | USB  | 1344x1072 | 36.6 ms | 30 fps | 66.7 ms   | 15 %     | 519 Mbit/s       |
| `P2160`      | USB  | 2704x2160 | 66.6 ms | 30 fps | 66.7 ms   | 44 %     | 2102 Mbit/s      |

`P2160` holds 30 fps and comes about 30 ms later than `P1080`; the cause of the extra time is not measured.

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

The camera thread does no color work: the `SerialUploader` converts and flips on the GPU. On an idle GPU:

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
| Depth, 480x300 uint16 | 138 Mbit/s                |
| Confidence, uint8     | 69 Mbit/s                 |
| Together              | 671 Mbit/s                |
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

## The backend

Each camera opens its own device and builds its own pipeline on it, as a depthai 3 pipeline is bound to one device.
It reads the frames, the tracklets and the IMU through queues on the node outputs, on depthai's callback thread,
where they become the contract's and get their capture time. The device runs the chain `model_chain` names:
`DETECTOR` gives boxes only; with a pose chain a tracklet carries its keypoints, from its own detection in the
one-model chain, or in the two-model chain from its crop's pose, put on it per frame by `KeypointJoin`. With
`depth_model` it delivers depth as DEPTH frames on the left camera's view, each with its confidence and the `DepthAlignment` that puts
it on the frames' view on the GPU. An NV12 frame goes on packed, so the callback thread does no color work; the
boxes and keypoints go on as the model found them, and the uploader straightens and mirrors them with their frame.

## The frame

The camera delivers one shape, 5:4 at the sensor mode's rows, so the depth model's 8:5 frame is used further across than a square would. There is no setting for it. The width is a multiple of 16 (`FRAME_WIDTH_ALIGNMENT`) because the iGPU's H.264 decoder hangs on any other width: 900 hangs, 896 and 720 play. A frame of another width cannot be replayed.

Undistortion stays on because aligned depth is always undistorted. Without it, depth misses the frame by about 14 of 360 pixels at its edges. The tilt correction needs it too, since a homography only holds on a rectilinear image.

Lens of the delivered frame, as tagged on a recording, in pixels:

| fx    | fy    | cx    | cy    |
|-------|-------|-------|-------|
| 538.1 | 538.0 | 449.3 | 369.8 |

That gives a view of about 79.6° by 67.6° **(deduction)**, and `readings.fov_factory` reports the horizontal figure from the first frame's lens (`rectilinear_field`). The color frame takes its lens from the camera; `fov` and the `lens_*` settings shape only the grey frame's warp. The color sensor spans 95° and the stereo pair 127° (Luxonis docs), so the whole frame has depth. Depth has about 291 rows of real detail inside the 720 rows, since the frame spans about 775 of the 800 mono rows **(deduction)**.

`HORIZONTAL_MIRROR` has no effect on this camera: the mirrored frame matches the plain one. So a color frame leaves its flips to the PC: `flip_h` and `flip_v` go on the frame as its flags, and the uploader flips the image, boxes and keypoints to match. The settings' `rotate90`, a camera mounted on its side, goes on every frame as its flag the same way; with `rotate90`, the tilt correction is not applied (a warning when `tilt.enabled` is on), as on a camera on its side the IMU's tilt is a rotation about the sensor's other axis.

## Depth

Neural Depth runs on the camera off 640x400 mono frames and arrives as 480x300 uint16 millimeters on the left mono camera's view, with 0 for no depth. Its confidence comes with it, 480x300 uint8, higher surer, on the 0–255 scale of the `confidence` filter: a `Sync` node joins the two outputs of Neural Depth into one message, as they share one sequence number and timestamp, and the DEPTH `CameraFrame` carries it as `confidence`. Probed with `NEURAL_480`, 100 frames, no IR light, it ran 73 to 255, median 230; the 2.2 % of pixels with no depth at every cut read lower (median 158 against 231) but overlap the valid range, so no one value parts them. The `Sync` node adds no wait: Deep Flow live over USB at 50 fps, 30 s after 10 s, as the panel reads it:

| Depth            | Rate     | Capture to PC, mean | p95     | Max     | Worst gap |
|------------------|----------|---------------------|---------|---------|-----------|
| With confidence  | 50.0 fps | 43.5 ms             | 49.3 ms | 51.0 ms | 20.0 ms   |

which matches the depth alone in [Link](#link) and [Depth filters](#depth-filters). Each depth frame carries its `DepthAlignment`: both lenses and the move from the left camera to the color camera, 3.76 cm sideways, read once from the calibration the frames carry. The uploader puts depth on the color view on the GPU (`DepthAligner` in `modules/inference/preprocessing/depth_align.py`), together with the tilt correction and the mirror: each depth pixel goes to its point in space and into the color frame, the nearest point wins where two land on one pixel, and a pixel nothing lands on stays 0. The output is 374x300, the depth's rows at the color frame's aspect (`aligned_depth_size`).

Next to a near object the color camera sees background the left camera could not, so that band has no depth. It is about `538 px × 3.76 cm × (1/Z_near − 1/Z_far)` wide, 12 px for a player at 1 m before a wall at 2.5 m. The black shadow bands beside near objects are this band and the stereo occlusion.

`ImageAlign`, the camera's own alignment and its only one on RVC4, fills part of that band with the near object's depth, so a player grows 10 to 20 color pixels on one side at about 1 m **(measured)**. It also delays depth: capture to PC at 60 fps, beside color, without the detector:

| Depth           | Median  | p95     |
|-----------------|---------|---------|
| Neural Depth    | 36.9 ms | 46.8 ms |
| `ImageAlign`    | 45.0 ms | 51.5 ms |

The alignment takes 0.07 ms of GPU a frame with its copy, as two CUDA kernels compiled at runtime; it matches the GL `DepthAlign` shader (0.13 ms), which matches a float64 reference on captured frames (the same empty pixels, values within 6 mm), but for about one border pixel a frame (`scripts/depth_align_check.py`). Recordings store the depth on its own view with its alignment in the chunk info; recordings made before hold depth already on the color view and no alignment, and play through the same pass.

The uploader hands it on as float32 meters with the confidence on the color view (`GpuDepth`, [INFERENCE.md](../../inference/INFERENCE.md)); deep_flow's view stage reads it into the flow's view ([INPUT_THREADS.md](../../flow/INPUT_THREADS.md)), and `ViewDepth` draws it. The colormap matches depthai's `colorizeDepthFrame` defaults: JET on a log scale from 0.3 to 12 m, which puts the detail close to the camera.

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

`H = K' R K⁻¹`, normalized uv to uv (`tilt_homography` in [tilt_correction.py](../geometry/tilt_correction.py)). `K` is the delivered lens and `R` the tilt. The frame's rows run opposite to the lens's and the IMU's **(site fact)**, so `TiltCorrection` makes `H` in their rows and turns it over to the frame's (`FLIP_V`). That is the opposite tilt on the lens with its center row mirrored. `K'` refits the picture to the frame, because plain `K` at 30° moves the frame's center by 0.58 focal lengths, about 310 of 720 rows.

Depth pixels move by `H`, and their values change too. Depth is measured along the tilted axis, so a standing player of 1.8 m at 30° reads as leaning by 0.9 m. The corrected depth is `z' = z × w`, with `w` from `H`'s third row, the distance along the level axis.

| Part  | Where it happens                          |
|-------|-------------------------------------------|
| `H`   | `TiltCorrection`, at the IMU's 5 Hz       |
| Color | `ImageUpload`, one CUDA kernel            |
| Depth | `DepthAlign` compute shader               |
| Boxes | `corrected_roi` and `corrected_keypoints` |

`H` is rebuilt only when the angle moves by 0.25° or more (`TILT_STEP`), since a picture that followed the noisy IMU would shimmer. The correction runs on the PC for two reasons: the chip warps grey frames only (Luxonis docs), and `ImageManip`'s four-point transform needs the BGR path (see [color path](#color-path)).

The picture's height fills the frame's, so every row has picture: the wide end is cropped at the sides and the narrow end leaves wedges without picture beside it. On the 992x800 frame at 30° the narrow end is 518 px wide, so the flow's centered 448x800 window has no black; from about 34° the wedges reach it, and past about 56° a frame corner passes the horizon (`MIN_CORNER_SCALE`) **(deduction, from the lens)**. The angle is not capped, so a mount that is too steep shows as wedges. The feet end is enlarged most and the far end squeezed most, so the far end loses the most detail **(deduction)**. The camera's height above the screen is not corrected in the image, since that would need reprojection and leave holes.

## Recording and replay

How recording and playback work is in [RECORDING.md](../../recording/RECORDING.md); this section holds what is the OAK4's. The format is in [chunk.py](../../recording/container/chunk.py) and [depth_frame.py](../../video/stream/depth_frame.py). Each chunk is one self-contained `.mkv`: stream 0 color, stream 1 depth, stream 2 the depth's confidence, fixed info in the tags and per-frame readings attached. Recording starts at the first depth frame, and each depth frame is written with the color frame captured with it ([Capture time](#capture-time)), so all streams have equal frame counts. A color frame whose depth never comes is left out. Recordings made before capture times paired each color frame with the newest depth, which was captured one or two frames earlier **(deduction)**. Replay plays a recorded frame's color and depth with one time.

Encoding runs on the iGPU because it does nothing else in the installation **(site fact)**, so recording stays out of the way of the show.

| Stream     | Codec       | Encoder    | Format                                |
|------------|-------------|------------|---------------------------------------|
| Color      | HEVC        | `hevc_qsv` | `video_quality` 24                    |
| Depth      | 12-bit HEVC | `hevc_qsv` | `p012le`, 3 mm steps, 12 m full scale |
| Confidence | 8-bit HEVC  | `hevc_qsv` | grey                                  |

`keyframe_interval` is 50 on all three. Recordings made before 2026-11-21 hold H.264 color and no confidence, and play as they are.

The streams share `keyframe_interval`, and depth and its confidence have no B-frames, so a seek finds every stream whole. A seek in a chunk goes to a color keyframe, and depth without a keyframe there comes back about 100 mm off. With B-frames depth was also off after a shared keyframe; why is not settled. A recording made with `keyframe_interval` 0 keeps the fault in its depth. Seeks on real recordings, each frame against the same frame read from the start:

| Recording                | Color | `keyframe_interval` | Seeks matched                        | A seek        |
|--------------------------|-------|---------------------|--------------------------------------|---------------|
| A real recording, `P720` | H.264 | 60                  | 15 of 15, color and depth            | 185 ms        |
| `probe_A5_hevc`, `P800`  | HEVC  | 50                  | 15 of 15 in color, depth, confidence | 241 to 310 ms |

The second row is 2026-11-21, a 28.4 s chunk of 1421 frames, read on the iGPU decoder, the median seek per stream.

Depth is lossy. That is acceptable because the fluid spreads and smooths its input and has only 20 to 32 depth layers, so a few millimeters should not show **(deduction)**. On synthetic frames 99 % of pixels are within 11 mm, with a steady offset of about 6 mm. Set depth quality with `-q:v`; `global_quality` breaks at low values, with errors in meters.

### File size

| Source                | `video_quality` | `keyframe_interval` | Color    | Depth    | Together |
|-----------------------|-----------------|---------------------|----------|----------|----------|
| Recorded, 18.1 s      | 18              | 0                   | 7.4 MB/s | 1.8 MB/s | 9.2 MB/s |
| Same clip, re-encoded | 24              | 0                   | 1.4 MB/s | 1.8 MB/s | 3.2 MB/s |
| Same clip, re-encoded | 24              | 60                  | 1.8 MB/s | 2.1 MB/s | 3.9 MB/s |
| `P1080`, recorded     | 24              | 60                  |          |          | 10 MB/s  |

All rows have H.264 color and no confidence. The first three rows are `P720`, and the third is the H.264 setup at that size. The last row is the H.264 setup at `P1080`, and a 60 s chunk comes to about 600 MB; HEVC color makes a `P800` chunk about 17 % smaller ([Encoder load](#encoder-load)). All rows are a dark room, and the color rate at 18 is mostly sensor noise **(deduction)**.

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

On the real camera, recording: Deep Flow live over USB at 50 fps, `P800` color 992x800, `video_quality` 24, `keyframe_interval` 50, color, depth and confidence encoding at once on the iGPU, 28.4 s each, started 10 s after the first frame, the two back to back on 2026-11-21. Load is the UHD 770's engines in Windows' GPU Engine counters, 17 samples, mean; QSV's encode shows on its 3D engine and VDBOX (VideoDecode):

| Color | 3D engine | VideoDecode | Queues, max                 | Dropped | Frames, 3 streams | Chunk   |
|-------|-----------|-------------|-----------------------------|---------|-------------------|---------|
| H.264 | 29.7 %    | 5.0 %       | writers 1, recorder 2       | 0       | 1418              | 87.1 MB |
| HEVC  | 36.0 %    | 4.1 %       | writers 1, recorder 1       | 0       | 1421              | 72.4 MB |

HEVC adds about 6 points and leaves the 3D engine about 63 % free, so color records in HEVC; the chunk comes out 17 % smaller.

### On the real camera

A 10 s recording at `P720` and one at `P1080` each held 60 fps with the recorder's queues at 1 or less, and gave 600 frames in both streams with 600 metadata lines. At `P1080`, color was 36.8 ms and depth 46.9 ms while recording.

The iGPU decodes both `P720` streams together at more than 670 fps color and 1700 fps depth. It plays both `P1080` streams together at real time, 600 frames each, and a seek to frame 240 matched the full read exactly in both.

### Detector in replay

Replay finds players on the PC with [`PersonDetector`](../../inference/person/person_detector.py). Its `resolution` is set to match `PERSON_DETECTOR_ARCHIVE` on the camera, and its `model_type` picks ONNX Runtime or TensorRT. It takes the uploaded images (`GpuImage`), so a frame goes to the GPU once: both runtimes stretch it into the model on the GPU and hand back only the box and person rows. The uploaded image is straightened, so the boxes are found in the straightened frame and need no `corrected_roi`, where the camera's detector looks at the frame before the correction.

Time per frame on an empty `P1080` frame, 500 runs each, batch 1, FP32, GPU otherwise idle, with the frame resized and converted on the CPU:

| `resolution` | ONNX, back to back | TensorRT, back to back | ONNX, at 60 Hz | TensorRT, at 60 Hz |
|--------------|--------------------|------------------------|----------------|--------------------|
| `R640X512`   | 5.6 ms             | 4.6 ms                 | 10.1 ms        | 9.0 ms             |
| `R512X416`   | 4.6 ms             | 4.0 ms                 | 8.4 ms         | 8.1 ms             |
| `R480X384`   | 4.3 ms             | 3.6 ms                 | 8.2 ms         | 7.4 ms             |

The times cover the preprocessing, inference and decoding; with the preprocessing on the GPU they are not measured. At 60 Hz the GPU drops to idle clocks between runs, which nearly doubles them, and the slowest frame at `R640X512` was 20.1 ms on both runtimes. Back to back, the slowest frame at any size was about 8 ms. Both runtimes find the same players on a photo of people, with scores within 0.001 and boxes within 0.03 px. ONNX Runtime stays the default, as TensorRT saves about 1 ms and shares the GPU with the other TensorRT models.
