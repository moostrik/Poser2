# SMILE

Art installation: a camera captures a face, an image generator turns it into a portrait, and the
result is shown with music (Max patch via OSC) and lights (Art-Net). Entry points: `smile.py`
(main app, `modules/Forgettable.py`) and `pl_player.py` (image players). Python 3.12, venv in
`.venv`, installed with `scripts/install.bat`. Tests: `.venv\Scripts\python -m pytest` from the repo root.

## Camera

`modules/cam/Camera.py` is the one source the app reads: it wraps the device (`VimbaCam` default,
`WebCam` with `-wc`) and an image source from `input/` that the panel's `use_image` switch puts in
place of the device frames at any time. A missing device is a warning and `device: none`, never an
automatic switch to the image. The camera settings (`Settings.camera`) also hold the face settings
(`camera.face`).

## Face

`modules/face/`: `FaceFinder` (own thread) detects every face of a camera frame (SCRFD,
`files/scrfd_10g_bnkps.onnx`) and judges it in `quality.py`: pose from the 5 keypoints (the pitch's
zero is `pitch_zero`, set on site with the "Set pitch zero" button: it depends on the camera height), eye
openness from the 106 landmarks (`FaceLandmarks`, `files/2d106det.onnx`, only run for faces that
passed the pose checks), sharpness from the pixels; the first failing check is the face's
`Rejection`. The subject of a frame is its largest qualifying face; a face that qualifies for
`stable_time` without interruption starts a sequence. The capture is the last frame of the
recording in which the subject qualified (the last frame at all when none did), cropped by
`portrait_crop` (eyes levelled, at the workflow's portrait size). Every frame's faces go to the render for the
operator overlay (`render.show_faces`: `FaceOverlayLayer`, marks built in `face_marks.py`).

## Image generator

ComfyUI server (installed in `ComfyUI/` from the `comfy/` bundle, both git-ignored, API on
127.0.0.1:8188). `modules/comfy/`: `Generator` sends each capture to ComfyUI and turns its
websocket events into `GenMessage`s for the render (CAM_DEPTH → DIFFUSION → DIFF_LOOP → IDLE);
`workflow.py` adapts the API export in `data/comfy/` (face crop removed — SMILE crops and rotates
the face itself — depth outputs added) and fills in every value from `PipelineSettings` per run: the
panel is the source of truth, the workflow only provides the structure. Keep one workflow per
visitor: a second workflow in between drops the first from ComfyUI's cache and reloads the models.
Our own ComfyUI nodes live in `comfy/aiia_grin/` (copied into `ComfyUI/custom_nodes/aiia_grin`).

## Render

OpenGL engine copied from poser2 (`modules/gl/`: glfw, GL 4.6 core, render thread owned by
`WindowManager`); keep it identical to poser2 so it can be synced with a diff, and fix shared bugs in
both. `modules/render/SmileRender.py` runs the `Sequencer` once per frame (it is the app clock:
Art-Net, OSC and saving hang off its messages), then the stage for the current `SequenceType`
(`stages.py`) renders the layers (`layers/`) and picks the one to show. Images from other threads go
through `inputs.py` (`Image.set_image`, uploaded on the render thread); the grin frames go
through `GrinPlayer.py`, a buffer the generator fills and the render plays on its own clock, with
its timings in the sequence settings. Shaders are
`classname.frag` next to their `.py` and hot-reload while running. `pl_player.py` uses the same engine
(`player/Render.py`).

## Recordings

`modules/save/ClipSaver.py` saves each finished visitor for the players under `recordings/` (git-ignored):
the grin as `grins/<id>.mp4` (h264 at the full portrait size, at `grin_fps`), the capture and the
portrait as `portraits/<id>_capture.jpg` and `<id>_portrait.jpg`. Files are built in `recordings/temp/`
and renamed into place when the grin is complete, so the players never see a partial file.
Encoding and decoding run `ffmpeg`/`ffprobe` subprocesses through `modules/video/ffmpeg.py` (ffmpeg
must be on the PATH: the gyan.dev full build); the encoder is picked at startup by a trial encode
(Intel Quick Sync on the iGPU, then Nvidia, then libx264; `sequence.encoder` forces one). Never start
those processes any other way: the apps run under pythonw, so they need `CREATE_NO_WINDOW` and no
inherited handles. The players (`player/`) decode each of the newest clips once into RAM
(`ClipPool`, 512x640, about 200 MB per clip) and play from memory; `pl_player.py -n` sets the number
of cells, `--pool` the clips kept, `--fps` the playback speed (default 15, 0 = the clip's own rate).
The cells start one by one, each in a random place on the next clip that is loaded (going round the
loaded clips again when there are fewer than cells). `scripts/start_players.bat` (re)starts both players alone
(`console` first for logs, other arguments pass through); `scripts/start.bat` calls it.

## Settings

Settings system copied from poser2 (`modules/settings/`); root tree in `modules/smile_settings.py`,
presets in `data/settings/`, web panel at `http://localhost:667`.

- `BaseSettings` subclasses are pure data containers with no runtime side effects
- Use `BaseSettings` for reactive configuration and `@dataclass` for plain value objects
- Components take their settings object (or a `Group`/`Child` of it) in the constructor, never as a global, and read fields when they use them
- Never unpack setting values into constructor parameters; a copied value goes stale when the panel changes it
- Cache only `Field.INIT` values, since they cannot change after `initialize()`
- Use `bind()` only to react to configuration changes, never to relay runtime data between components
- Keep bound callbacks thread-safe, and never write the field that triggered the callback
- Components that `bind()` must `unbind()` the same callbacks on teardown
- Treat `Field.READ` values as snapshots; copy arrays before handing them to another thread
- Keep `Field` descriptions to one short line of panel text: what the value is and its unit
