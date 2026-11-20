# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F). Who uses what, from the code: the contract
(`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture match) is read by
every app and by inference, recording, tracker and render. The backends are each one kind of camera, used now by
the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, `vimba` and `webcam` by smile. That is
general code with one user, which stays.

## A. Depth confidence to the motion

Neural Depth also gives a confidence per depth pixel (`NeuralDepth.confidence`). Today only the depth is taken, and
the confidence is used once, as the camera's cut (`DepthSettings.confidence`, 125 by default, 51 in deep_flow's settings files).
Probed on the OAK4 (2026-11-18, `NEURAL_480`, 100 frames, no IR light): the confidence is 480 × 300 RAW8 like the depth,
with the depth's sequence number and timestamp; higher is surer, on the cut's 0–255 scale (Luxonis docs). It ran 73 to
255, median 230, so deep_flow's 51 cuts nothing. 2.2% of pixels have no depth at every cut, 0 included; theirs is lower
(median 158 against 231) but overlaps the valid pixels' range, so no one value parts them.

The motion is to weight each depth sample by it (Flow Motion's line, [FLOW_PLAN.md](../flow/FLOW_PLAN.md#c-the-motion)),
so it goes the depth's whole way: camera, alignment, recording, view, motion. It rides on the depth at each step, never
as a frame of its own, so it is never paired with another frame's depth. A step in another thread's package is told to
that thread when it is built.

- **A1. The camera.** The oak4 pipeline links `neural.depth` and `neural.confidence` into one `Sync` node, so one
  message holds a frame's depth and its confidence and the backend needs no join (`pipeline.py`, `camera.py`). The DEPTH
  `CameraFrame` gets `confidence: np.ndarray | None`, uint8 (H, W) on the depth's view, a copy as the depth is; None
  from every other backend. It adds 7 MB/s on the link at 50 fps **(deduction: 480 × 300 bytes a frame)**. Check, live:
  the panel's depth fps and latency against OAK4.md's table, the Sync node adding no wait **(deduction: both outputs
  come off one node with one timestamp)**; `camera/oak4` tests. OAK4.md's Depth section, with the probe's numbers.
- **A2. The alignment.** The uploader stages the confidence beside the depth, in a pinned buffer of its own key, and
  `DepthAligner` moves it with the depth that wins a cell: the atomic minimum's key is the depth's float bits with
  255 − the confidence in the lowest 8, so the nearest point brings its own confidence in the one pass and, of two
  equally near, the surer wins; 15 bits of depth precision, under 0.1 mm at 3 m, against the recording's 3 mm steps.
  The result is one tensor (H, W, 2), the layout GL takes and the kernel writes: meters, and the confidence 0 to 1;
  0 and 0 where nothing lands; a confidence of 1 where the frame has none (an older recording), so nothing after needs
  a case for it. `GpuDepth.depth` says so; `InferenceFrameDepthLayer` takes it as it is, an RG32F texture: `Tensor`
  infers the format, and its two uploads (`_update_with_pbo`, `_update_with_cpu`), which take only (C, H, W) now, take
  (H, W, C) as `update` does. All three paths: the kernels, the torch ops (`scatter_reduce` on the key's int32 bits),
  and the CUDA graph, captured with the confidence as a second input (255 where the frame has none). `depth_metres`
  becomes `depth_meters`, and the file's comments say meters. Check: the preprocessing tests hold the kernels to the
  torch ops, the confidence included; `depth_align_check.py` holds the meters to the GL shader as before. INFERENCE.md's
  upload.
- **A3. Recording it.** A chunk's `.mkv` holds the confidence as a third stream after the depth, written by
  `StreamWriter` as grey 8-bit HEVC on the iGPU (`ClipWriter`, built), with the depth's keyframes. It is not in the depth
  stream's empty chroma: the iGPU's `hevc_qsv` encodes 12 bits only as 4:2:0, which halves the confidence to 240 × 150,
  and 4:4:4 only up to 10 bits, which would take the depth from 3 mm to 12 mm steps (checked 2026-11-18). The recording
  names its streams itself, `ChunkStream` (`VIDEO`, `DEPTH`, `CONFIDENCE`, in `container`), since `FrameType` has no
  confidence: the recorder's writers and temp files, `merge_chunk` and `finish_unmerged` go by it and take the third
  part, `ChunkInfo` gets `confidence` beside `depth`, and `mirrored_capture` mirrors the confidence with its depth. The
  player reads streams 0 and 1 as before, so the repository plays every recording after this step. Check:
  `test_round_trip` records three streams (equal counts, the confidence back within a few levels by ffmpeg), `test_chunk`
  merges and finishes three parts. RECORDING.md's chunk table.
- **A4. Playing it.** The player goes by `ChunkStream` where it goes by `FrameType` now (its readers, its hand-over and
  the frames waiting to be paired), and reads stream 2 grey where `ChunkInfo.confidence` says it is there. Its
  hand-over hands the depth frame out with its confidence (`ClipFrameCallback`), which the replay camera puts on the
  DEPTH `CameraFrame` and `RecordedPoses.set_clip_frame` takes as well (deep_flow, low_flow); a chunk without it plays
  with None. Check: `test_round_trip` plays three streams back (the confidence on its depth frame, a seek whole in all
  three), the replay camera's tests. RECORDING.md's playback.
- **A5. The color in HEVC.** deep_flow's `recorder.video_codec` in its settings files from H264 to HEVC, so a chunk is one codec,
  smaller at the same quality **(deduction: HEVC is usually 30–50% smaller)**; `MP4` stays H.264, for any player; older
  recordings play as they are. The writer side is built (1670244c). Check, first: the iGPU encoder's load on the real
  camera at 50 fps, color in HEVC with the depth and the confidence encoding at once, against color in H.264; color has
  only been measured in H.264 (18 % at 900 × 720, OAK4.md's Encoder load), and the setting changes only if HEVC leaves
  margin. Then OAK4.md's seek check on a new recording. After A3, for the three streams.
- **A6. The view.** `ViewDepth`'s textures become RG32F, the confidence in green: the upscale blends green with the
  depth's own weights, or takes the one pixel's where it takes the one depth; the fill's last pass keeps the read's
  green where the read has a depth and gives 0 where it filled. The crop's depth feeds only the mask and stays as it
  is. `DepthView` and `LucasKanadeFlow3D.update` carry it with the texture; the motion's shaders read red, so the motion
  runs as now until Flow Motion uses green. `TextureKind.DEPTH` says green is the confidence, and the monitor's depth
  drawing gets a second view of a DEPTH port, its green as grey from 0 to 1, beside the colormap of its red. Check: `view_check.py` reads green (the camera's where kept, 0 where filled); red as before.
  VIEW.md's depth.

The recordings measured so far have no confidence, so the motion's A/B needs a new recording.

## Open

- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.

## Rules it touches

- Stamp a frame once, where the backend hands it over: A1, the confidence on the depth's frame.
- An item handed to a callback is not changed afterwards: A1, the confidence a copy.
- Fix a problem where it is made: A2, a missing confidence is 1 at the alignment, and `Tensor` takes (H, W, C) in its uploads.
- Avoid unnecessary copies: A2, one tensor of two channels.
- Keep settings files in sync with code changes: A5, deep_flow's `video_codec`.
- A step checked by a run or a test is one commit: A1–A6, A3 and A4 apart.
- Name a concept after what it is: `ChunkStream`, the recording's streams, beside the camera's `FrameType`.
- Separate what the code or a measurement shows from what is inferred: the probe, A1, A5.
