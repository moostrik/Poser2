# SMILE

Art installation: a camera captures a face, an image generator turns it into a portrait, and the
result is shown with music (Max patch via OSC) and lights (Art-Net). Entry points: `smile.py`
(main app, `modules/Forgettable.py`) and `pl_player.py` (image players). Python 3.12, venv in
`.venv`, installed with `scripts/install.bat`. Tests: `.venv\Scripts\python -m pytest` from the repo root.
All Python code lives in `modules/`; `data/` is what the installation is given (presets, models,
workflow, audio, test photos), `output/` what it produces (git-ignored, safe to delete).

## Camera

`modules/cam/Camera.py` is the one source the app reads: it wraps the device (`VimbaCam` default,
`WebCam` with `-wc`) and an image source from `data/input/` that the panel's `use_image` switch puts in
place of the device frames at any time. A missing device is a warning and `device: none`, never an
automatic switch to the image. The camera settings (`Settings.camera`) also hold the face settings
(`camera.face`). Gain, white balance and colour (saturation, hue, gamma) are panel settings that
`FeatureWriter` writes to the Vimba camera when they change, over what the camera's `UserSet2`
loaded; a feature the camera model lacks is a warning. "Set white balance" lets the camera measure
once and takes the ratios it found into the panel.

## Face

`modules/face/`: `FaceFinder` (own thread) detects every face of a camera frame (SCRFD,
`data/models/scrfd_10g_bnkps.onnx`) and judges it in `quality.py`: detector confidence, size (the
distance between the eyes, min and max), position (horizontal distance from the image centre),
pose from the 5 keypoints (the pitch's zero is `pitch_zero`, set on site with the "Set pitch zero"
button: it depends on the camera height), eye openness from the 106 landmarks (`FaceLandmarks`,
`data/models/2d106det.onnx`, only run for faces that passed the other checks); the first failing check
is the face's `Rejection`, and `distance()` says how far the face is from qualifying (0 good, a floor
just past a limit, 1 when its worst check is `falloff` times the limit past it). Gaze is not checked: the "looking at viewer" LoRA corrects it. The subject of a frame is its largest qualifying face.
The flash is earned: the `Sequencer` keeps a **charge** that rises while the newest detector frame
has a qualifying subject (`sequence.charge` seconds to fill) and drains while it has none
(`sequence.drain` seconds, slower); CAMERA_SHOW is an empty charge, CAMERA_DETECT a partial one
(its progress is the charge, also for OSC and Art-Net), and a full one starts the flash. A manual
start charges without a face. The capture is the last frame in which the subject qualified (the
last frame at all when none did), cropped by
`portrait_crop` (eyes levelled, at the workflow's portrait size). Every frame's faces go to the render
(`Inputs.faces`): `CameraLayer` draws the camera in black and white (`facecolor.frag`). Every face's
box (the portrait crop's, upright: `FaceObservation.crop`) shows the face's distance from good
(`render.face_effect`: dark, pixels, blur or fog, at `render.face_amount`), so a visitor sees
whether they get closer; the subject's box turns to colour as the charge fills (`render.face_fill`:
fade, rise, iris, trace or scan). `render.face_marks` switches all marks on or off: the subject's outline (from
grey to `render.face_color` with the charge) and grey keypoints, the other faces in white, a
rejected one in grey with its reject reason (the panel has the readout). They are part of the
camera layer, so the flash fades them out with it. Marks are built in `face_marks.py`, drawn by `MarkDrawer`.

## Image generator

ComfyUI server (installed in `ComfyUI/` from the `comfy/` bundle; `ComfyUI/` and `comfy/models/` are git-ignored, API on
127.0.0.1:8188), started by `scripts/start_comfy.bat`, with a log per start in `output/logs/comfy_*.log`. Every model is
local and a run makes no internet calls; a pack only downloads a model that is missing. Keep ComfyUI core and the node packs unmodified; depth comes from
ComfyUI's own Depth Anything 3 nodes. `modules/comfy/`: `Generator` sends each capture to ComfyUI and turns its
websocket events into `GenMessage`s for the render (CAM_DEPTH → DIFFUSION → DIFF_LOOP → IDLE);
`workflow.py` adapts the API export in `data/comfy/` (face crop removed — SMILE crops and rotates
the face itself — depth outputs added) and fills in every value from `PipelineSettings` per run: the
panel is the source of truth, the workflow only provides the structure. Each visitor gets one of two
styles (`style_a`, `style_b`, cycled by `styles.py` every `style_visitors` visitors, or one of them for
everyone with `style_mode` A or B; the style id is the movement id of the music): a style holds the describe settings (instruction, max tokens, repetition
penalty, prefix and suffix around the description), the style prompt, the negative and three LoRA
slots (file and strength, filled into the workflow's three LoRA nodes; `none` or 0 is off). Keep one workflow per
visitor: a second workflow in between drops the first from ComfyUI's cache and reloads the models.
After every connect, a warm-up run on `data/comfy/warmup.jpg` (`build_warmup`: the same workflow
with one sampler step and one grin frame) loads the models and proves the chain works (own limit
`WARMUP_TIMEOUT`; a visitor has `connection.job_timeout`). Until `Generator.ready` no sequence starts (`Sequencer.blocked`,
set in `Forgettable`) and the render shows the generator's status as a line (`StatusLayer`).
For debugging the rest of the cycle, `pipeline.skip_generation` sends the capture for the
description and its depth only (`build_describe`: no portrait, no grin); the flash out then goes
to the timeout (`Sequencer.finish`). `pipeline.skip_animation` only leaves out the grin
(`build_portrait`): the portrait holds for the timeout.
ComfyUI sends the events of all its clients from one queue, so a client that stops reading stalls
them for SMILE too: a ComfyUI browser tab that Edge put to sleep did this (ComfyUI's log says
"Prompt executed", SMILE gets nothing). The generator sees it (`NoEvents`: a queued workflow left the
queue without one event), shows `no events from ComfyUI: close or reload its browser tab` and is
not ready until a warm-up gets its events again; a warm-up that fails is retried, never counted as
ready. Keep no ComfyUI tab open on the installation, or exclude `127.0.0.1:8188` from Edge's sleeping tabs.
For tests in the ComfyUI page, `python -m modules.comfy.testflow` writes `data/comfy/smile_test.json`
(drag it into the page): what the app sends with a preset's values, plus an input switch (the
app's last capture, or any photo through the face crop), previews and a video recorder. Regenerate
it after changing the workflow or the presets instead of editing a copy by hand.
Our own ComfyUI nodes live in `comfy/aiia_grin/` (copied into `ComfyUI/custom_nodes/aiia_grin`).

## Render

OpenGL engine copied from poser2 (`modules/gl/`: glfw, GL 4.6 core, render thread owned by
`WindowManager`); keep it identical to poser2 so it can be synced with a diff, and fix shared bugs in
both. `modules/render/SmileRender.py` runs the `Sequencer` once per frame (it is the app clock:
Art-Net, OSC and saving hang off its messages), then the stage for the current `SequenceType`
(`stages.py`) renders the layers (`layers/`) and picks the one to show. Images from other threads go
through `inputs.py` (`Image.set_image`, uploaded on the render thread); the grin frames go
through `GrinPlayer.py`, a buffer the generator fills and the render plays on its own clock, with
its timings in the sequence settings. `render.show_camera` draws the raw camera on top and starts
no new sequence while it is on (`Sequencer.blocked`). Shaders are
`classname.frag` next to their `.py` and hot-reload while running. `pl_player.py` uses the same engine
(`modules/player/Render.py`).

## Recordings

`modules/save/ClipSaver.py` saves each finished visitor for the players under `output/` (git-ignored):
the grin as `grins/<id>.mp4` (h264 at the full portrait size, at `grin_fps`), the capture and the
portrait as `portraits/<id>_capture.jpg` and `<id>_portrait.jpg`. Files are built in `output/temp/`
and renamed into place when the grin is complete, so the players never see a partial file.
Encoding and decoding run `ffmpeg`/`ffprobe` subprocesses through `modules/video/ffmpeg.py` (ffmpeg
must be on the PATH: the gyan.dev full build); the encoder is picked at startup by a trial encode
(Intel Quick Sync on the iGPU, then Nvidia, then libx264; `sequence.encoder` forces one). Never start
those processes any other way: the apps run under pythonw, so they need `CREATE_NO_WINDOW` and no
inherited handles. The players (`modules/player/`) decode each of the newest clips once into RAM
(`ClipPool`, 512x640, about 200 MB per clip) and play from memory; `pl_player.py -n` sets the number
of cells, `--pool` the clips kept, `--fps` the playback speed (default 20, 0 = the clip's own rate), `--spread` a random speed difference per play so the cells drift apart (default 0.1), `--trail` a blend with the previous output (default 0: none).
The cells start one by one, each in a random place on the next clip that is loaded (going round the
loaded clips again when there are fewer than cells). `scripts/start_players.bat` (re)starts both players alone,
each fullscreen on its own monitor (a player whose monitor is not connected takes a window on the
primary one); leading `window` puts both in a window for testing, `console` gives them a console
for logs, other arguments pass through. `scripts/startSMILE.bat` is the main startup: it closes and
restarts everything (ComfyUI, Max, the players, SMILE).

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
