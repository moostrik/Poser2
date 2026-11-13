---
description: The SMILE app
paths:
  - "apps/smile/**"
---
# SMILE

- Say things with `logging`, never `print`; the app runs without a console
- A worker thread's loop catches what its work raises, logs it and goes on
- Keep `README.md` current: setup, scripts, on-site checks, known limits
- Keep ComfyUI core and the node packs unmodified; our own nodes sit in `data/comfy/aiia_grin/`
- The panel is the source of truth; the workflow only provides the structure
- Keep one workflow per visitor
- Regenerate `data/comfy/smile_test.json` with `testflow`; never edit it by hand
- A shader is a `.frag` file named after its `.py`, beside it
- `sphere.py` is the one place that knows the sphere and its wiring
- Start `ffmpeg` and `ffprobe` only through `modules/video`
- Build an output file in `output/temp/` and rename it into place when complete
- The online token is a file, never a setting in a preset
- Send OSC bools as ints
- SMILE reaches its walls by a call, never by OSC
- `cam` is the app's image source (a `modules/camera` backend, an image folder, a phone), not a camera backend
