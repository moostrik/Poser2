# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

## A. Open findings

- A2. The player's decode on the CPU? A 60 s chunk (992 × 800 H.264) decoded in 1.2 s on the CPU against 6.3 s on the
  iGPU (d3d12va), measured once outside the app. Question: is the CPU the better default for playback? Check:
  deep_flow `-sim` with the player's decoder on the CPU and on the iGPU: the frame interval, late frames, CPU load,
  ticks.
- A4. The preview checked by eye (Matthias): a held take, the in and out markers dragged, the preview showing the
  frame under the marker.

## The rules it touches

- Prefer the design with the lower latency, measured in the app: A2
