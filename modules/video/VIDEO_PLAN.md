# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

## A. Open findings

- A2. The player's decode on the CPU? A 60 s chunk (992 × 800 H.264) decoded in 1.2 s on the CPU against 6.3 s on the
  iGPU (d3d12va), measured once outside the app. Question: is the CPU the better default for playback? Check:
  deep_flow `-sim` with the player's decoder on the CPU and on the iGPU: the frame interval, late frames, CPU load,
  ticks.
- A3. Type errors in smile's tests: about 180 across 20 test files, predating today: `Optional` results used
  without narrowing, and `SimpleNamespace` or fakes passed where real classes are typed (recording's were fixed in
  their own commit). Fixed at the cause: a narrowing assert where a test expects a value, a `Protocol` for what the
  code needs, or the fake typed as what it stands for; no suppressions. Check: pyright clean on smile, its tests pass.
- A4. The preview checked by eye (Matthias): a held take, the in and out markers dragged, the preview showing the
  frame under the marker.

## The rules it touches

- Fix a problem where it is made: A3
- Prefer the design with the lower latency, measured in the app: A2
- Fix a type error by correcting the annotation; suppress only as a last resort: A3
- A thread that finds a minor problem fixes it and asks the owner thread for feedback: A3
