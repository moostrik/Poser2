---
description: The SMILE app
paths:
  - "apps/smile/**"
---
# SMILE

- The running app says things with `logging`, never `print`; it has no console
- A worker thread's loop catches what its work raises, logs it and goes on
- Keep `README.md` current: setup, scripts, on-site checks, known limits
- Keep ComfyUI core and the node packs unmodified; our own nodes sit in `data/comfy/aiia_grin/`
- The panel is the source of truth; the workflow only provides the structure
- Keep one workflow per visitor
- Regenerate `data/comfy/smile_test.json` with `testflow`; never edit it by hand
- `sphere.py` is the one place that knows the sphere and its wiring
- Start `ffmpeg` and `ffprobe` only through `modules/video`
- Build an output file in `output/temp/` and rename it into place when complete
- The online token is a file, never a setting in a preset
- Send OSC bools as ints
- SMILE reaches its walls by a call, never by OSC
- The relay is a standalone web server; it uses async/await
- `cam` is the app's image source (a `modules/camera` backend, an image folder, a phone), not a camera backend

## Names

- The person in SMILE is a "visitor"; a player plays clips
- One of windows 1 and 2, which drive 24 screens of the sphere: `wall` (`sphere.WALLS`, `WallCompositor`, `walls`, "On walls"); not `player`, `PLAYERS`, `PlayerRender`, "player walls"
- The held grin player: `grin_player`; the clip player: `clip_player`
- The image generator: `generator` (`GeneratorMessage`, `GeneratorState`, settings group `generator`); not `diffuse`, `diffusion_callback`, `pipeline`, `GenMessage`. `DIFFUSION` stays as a state name
- The sequence's end effect: bloom / fade (`bloom_*`, `fade_*`, `BloomLayer`); not `destroy`, `destruct`
- The music track: `music_id`; the visitor's style: `style_id`; not `audio_id`, `movement_id`
- The laughter test: `LaughterTest`; not `ScreenTest`
