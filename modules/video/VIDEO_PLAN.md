# The video module's plan

## A. One ffmpeg system

One place starts ffmpeg and ffprobe, with one set of names ([VIDEO.md](VIDEO.md)). Left: A6, after the camera plan's
A1.

- A6. deep_flow's color in HEVC (CAMERA_PLAN.md A4's writer side, after the camera plan's A1, the confidence field):
  deep_flow's presets from H.264 to HEVC on the iGPU, `MP4` staying H.264. The confidence stream's writer is
  `StreamWriter` with gray input, HEVC on the iGPU, the depth's keyframe interval and no B-frames; the camera plan
  wires it. OAK4.md's recording section. Check: the seek check and the encoder load run again.

## The rules it touches

- Keep presets in sync with code changes: A6
- A document describes what is built: A6, OAK4.md
