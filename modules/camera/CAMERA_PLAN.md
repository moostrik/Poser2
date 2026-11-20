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

- A step checked by a run or a test is one commit: A6.
- Separate what the code or a measurement shows from what is inferred: the probe.
