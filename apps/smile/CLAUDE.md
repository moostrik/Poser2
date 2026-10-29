# SMILE

Art installation: a camera captures a face, an image generator turns it into a portrait, and the
result is shown with music (Max patch via OSC) and lights (Art-Net), and the grins of the visitors before
play on two walls of screens. SMILE is `apps/smile/` of Poser2: `launcher.py -app smile` from the Poser2 root starts it
(`SmileMain` in `main.py`; its own arguments `-fs`, `-wc`, `-lm`, `-dw`, `-dh` follow the launcher's; the walls are two
secondary windows of its render). Paths in this file are relative to `apps/smile/`; commands run from the Poser2 root with
its `.venv` (`scripts/install.bat` there makes it, `apps/smile/scripts/install.bat` adds ComfyUI). Tests:
`python -m unittest discover -s apps/smile/<package>/tests -t .` (`tests`, `face`, `comfy`, ..., `relay`).
The app's packages are its own (`cam`, `face`, `comfy`, `render`, `player`, ...); `modules/gl`, `modules/settings`,
`modules/utils` and `modules/log_config.py` are Poser2's, shared with the other apps. `data/` is what the installation
is given (presets, models, workflow, audio, test photos), `output/` what it produces (git-ignored, safe to delete).
`README.md` is for whoever installs or runs the installation: setup, scripts, the on-site checklist
with the date each check passed, and the known limits; keep it current when any of that changes.
The app runs under pythonw, without a console: say things with `logging`, never `print` (the log file
in Poser2's `logs/` is the only place they show; unhandled exceptions of the threads and of the main
thread go there too, `log_config.install_excepthooks`, and each one is a line in
`logs/crashes.log`, which every start reports as its first log line, `crash_summary`). ComfyUI's logs are in `output/logs/`. A worker thread's loop catches what its work
raises, logs it (not at every round) and goes on: a thread that ends leaves an installation that looks
alive and does nothing (the face finder: nobody is ever chosen; the generator: the flash stays).

## Camera

`cam/Camera.py` is the one source the app reads: it wraps the device, a camera backend of `modules/camera/` on the
camera contract (`modules.camera.vimba.Camera` default, `modules.camera.webcam.Camera` with `-wc`; both take
`rotate90` for the portrait mode at construction), and an image source from `data/input/` that the panel's `use_image` switch puts in
place of the device frames at any time. `Camera` fits every frame to the render size with `cover()`
(scaled to fill it, the middle kept, no distortion); the backends only rotate and flip. A missing device is a warning and `device: none`, never an
automatic switch to the image. The Vimba backend's thread keeps the camera: `start()` makes the first try, and
the thread searches again every 5 s while there is none and opens it again (user set, packet size, the
panel's values) when no complete frame arrived for 5 s; the `device` readout follows its `connected`. Without the Vimba SDK
the backend logs an error at start and stays unconnected. Every frame
has the time it arrived (`Camera.frame()`, on `time.perf_counter()` like every stamp and timer of the sequence; a device frame the time its backend stamped): the face finder gets new frames only and its faces carry
that time, so a camera that stopped never charges on its last frame, and the render shows
`no camera frames` as its status line after 3 s without one (`Camera.stalled`). The camera settings (`Settings.camera`, `cam/settings.py`)
are the Vimba backend's `VimbaCameraSettings` with the app's source fields and the face settings
(`camera.face`). Exposure, gain, white balance and colour (saturation, hue, gamma) are panel settings that
the backend's `SensorControl` writes to the Vimba camera (Mako G-125C) when they change, over what the camera's `UserSet2`
loaded; a feature the camera model lacks is a warning, and so is a write the camera refuses, which is
tried again (it does not switch the feature off). Exposure and gain each have auto, a manual
value (sent while auto is off), a limit for auto and a readout of what the camera uses now; both
autos aim for `target_brightness`, the exposure going up first and the gain after it. "Set WB" (white balance) lets the camera measure
once and takes the ratios it found into the panel.
The camera is a GigE camera on the PC's Ethernet adapter. Its `UserSet2` holds the region of interest, 1200x960 of the
1292x964 sensor, centred (OffsetX 46, OffsetY 2): the render's 4:5 turned, which `Camera.cover()` scales by exactly 16/15 to
1024x1280 with nothing cropped; the log line at open shows the region the set loaded. It sends BGR, 3.5 MB per frame
(104 MB/s at 30 fps, the panel's `fps`, also written by `SensorControl`; `UserSet2` holds 20; `readings.video_fps` is the rate of the frames
that arrive complete; one that lost network packets arrives `Incomplete`, with part of an older image still in its buffer,
and is skipped and counted in `dropped_frames`). At every open `setup_camera` runs
`GVSPAdjustPacketSize` (after loading `UserSet2`, which holds a packet size of its own), which takes the largest packet the path accepts (`GevSCPSPacketSize`, up to
9973). For that the adapter's Jumbo Packet must be on (Device Manager → the Ethernet adapter →
Advanced → Jumbo Packet → 9014 Bytes; it is off by default on a new PC or after a driver
reinstall), and so must any switch between camera and PC; otherwise the size stays 1500 and frames
drop more easily, and the log warns at open. The size is only negotiated at open: restart SMILE after changing the adapter.

## Face

`modules/face` (Poser2's, promoted from SMILE): `FaceFinder` (own thread) detects every face of a camera frame (SCRFD,
`data/models/scrfd_10g_bnkps.onnx`) and judges it in `quality.py`: detector confidence, size (the
distance between the eyes, a minimum), the portrait crop's box at least `crop_margin` crop widths inside the image edge
(`crop_edge`: the one check for too close and too near an edge; negative lets it reach past),
pose from the 5 keypoints (the pitch's zero is `pitch_zero`, set on site with the "Set pitch zero"
button: it depends on the camera height), eye openness from the 106 landmarks (`FaceLandmarks`,
`data/models/2d106det.onnx`, only run for faces that passed the other checks); the first failing check
is the face's `Rejection`, and `distance()` says how far the face is from qualifying (0 good, a floor
just past a limit, 1 when its worst check is `falloff` times the limit past it). Gaze is not checked: the "looking at viewer" LoRA corrects it. The subject of a frame is its largest qualifying face.
The flash is earned: the `Sequencer` keeps a **charge** that rises while the newest detector frame
has a qualifying subject (`sequence.charge` seconds to fill) and drains while it has none
(`sequence.drain` seconds, slower); a detector frame is as old as its camera frame (`FaceFrame.time`),
and one older than half a second counts as nobody. CAMERA_SHOW is an empty charge, CAMERA_DETECT a partial one
(its progress is the charge, also for OSC and Art-Net), and a full one starts the flash. A manual
start charges without a face; one that gets blocked before it is full is dropped, it does not wait. The capture is the last frame in which the subject qualified (the
last frame at all when none did), cropped by
`portrait_crop` (eyes levelled, at the workflow's portrait size). Every frame's faces go to the render
(`Inputs.faces`): `CameraLayer` draws the camera in black and white (`facecolor.frag`). Every face's
box (the portrait crop's, upright: `FaceObservation.crop`) shows the face's distance from good
(`render.face_effect`: dark, pixels, blur or fog, at `render.face_amount`), so a visitor sees
whether they get closer; the subject's box turns to colour as the charge fills (`render.face_fill`:
fade, rise, iris, trace or scan). `render.face_marks` switches all marks on or off: the subject's outline (from
grey to `render.face_color` with the charge) and grey keypoints, the other faces in white, a
rejected one in grey with its reject reason (the panel has the readout). They are part of the
camera layer, so the flash fades them out with it. Marks are built in `face_marks.py`, drawn by `MarkDrawer`; a box that
reaches past the image is drawn clamped to `BOX_PAD_PX` inside it (`fit_box`), so the subject's outline and fill stay whole.

## Image generator

ComfyUI server (installed in `ComfyUI/` by `scripts/install.bat` through `scripts/install_comfy.ps1`: the core commit and the pack versions
pinned in `data/comfy/comfy.lock`, the packs from their registry archives kept in `data/comfy/packs` (git LFS), the venv from `data/comfy/requirements.lock`;
no pack is patched, what SMILE needs from a pack that the pack does wrong is done in `data/comfy/aiia_grin`; `install_comfy.ps1 -Update` builds the
newest ComfyUI and packs beside the install, copies the models over, smoke-tests them with a fixed-seed picture before and after, and `-Accept`
writes the locks, `-Rollback` restores: an update may change what a node expects, which the smoke test reports by node and `workflow.py` then
provides; the models live in `ComfyUI/models` as ComfyUI has it, listed with size, sha256 and source in `data/comfy/models.json` and fetched by
`comfy/models.py` from a local folder when one is given, else from the source: the public URL, or for the three LoRAs the copy in
`data/comfy/loras` (git LFS); a backup is a copy of the SMILE folder; `ComfyUI/` is git-ignored, API on
127.0.0.1:8188), started by `scripts/start_comfy.bat`, with a log per start in `output/logs/comfy_*.log`. Every model is
local and a run makes no internet calls; a model that is missing is an error (`HF_HUB_OFFLINE=1` in `start_comfy.bat`) and a WARNING in the SMILE log when the panel names it. Keep ComfyUI core and the node packs unmodified; depth comes from
ComfyUI's own Depth Anything 3 nodes. `comfy/`: `Generator` sends each capture to ComfyUI and turns its
websocket events into `GenMessage`s for the render (CAM_DEPTH → DIFFUSION → DIFF_LOOP → IDLE);
`workflow.py` adapts the API export in `data/comfy/` (SMILE crops and rotates the face itself, so the export
starts from the uploaded image; an old export's crop subgraph and video recorder are stripped; depth outputs added) and fills in every value from `PipelineSettings` per run: the
panel is the source of truth, the workflow only provides the structure. Each visitor gets one of two
styles (`style_a`, `style_b`, cycled by `styles.py`: `style_visitors_a` visitors in A, then `style_visitors_b` in B, or one of them for
everyone with `style_mode` A or B; the style id is the movement id of the music): a style holds the describe settings (instruction, max tokens, repetition
penalty, prefix and suffix around the description), the style prompt, the negative and three LoRA
slots (file and strength, filled into the workflow's three LoRA nodes; `none` or 0 is off). The model is
shared by both styles (`pipeline.portrait`): the `checkpoint`, and a `distill_lora` (e.g. DMD2, on the
model only, a node `build` adds in front of the style LoRAs) whose strength sets how far the checkpoint
is distilled; steps, CFG, sampler and scheduler have to be set to match by hand. Keep one workflow per
visitor: a second workflow in between drops the first from ComfyUI's cache and reloads the models.
After every connect, a warm-up run on `data/comfy/warmup.jpg` (`build_warmup`: the same workflow
with one sampler step and one grin frame) loads the models and proves the chain works (own limit
`WARMUP_TIMEOUT`; a visitor has `connection.job_timeout`). Until `Generator.ready` no sequence starts (`Sequencer.blocked`,
set in `Main`) and the render shows the generator's status as a line (`StatusLayer`) and the
whole camera image in pixels without any face box or mark (`CameraLayer.warmup`, from `Generator.warmup`: the step
the warm-up reached, None when ready): the largest block that tiles the image whole (`warmup_block`, 256 pixels at
1024 x 1280) without ComfyUI, halved with every step the warm-up reaches, sharp when ready.
The sequence waits for the generator (the flash for the capture's depth, the result for IDLE), so every
end of a job says IDLE, and three things keep that wait from lasting: a reset (`Generator.interrupt`,
sent at every RESET) ends a visitor's job only, never a warm-up, which would otherwise count as passed;
a capture taken while the generator is not ready is not handed over and stands as the result at once;
and `Main._limit_visitor` ends a visitor the generator did not end 30 s after its `job_timeout`
(not while the grin loops), which only happens when the generator's thread is stuck.
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
For tests in the ComfyUI page, `python -m apps.smile.comfy.testflow` writes `data/comfy/smile_test.json`
(drag it into the page): what the app sends with a preset's values, plus an input switch (the
app's last capture; any other photo is cropped first with `python -m modules.face.crop`), previews and a video
recorder from ComfyUI's own video nodes. Regenerate
it after changing the workflow or the presets instead of editing a copy by hand.
Our own ComfyUI nodes live in `data/comfy/aiia_grin/` (`ComfyUI/custom_nodes/aiia_grin` is a junction to it: restart ComfyUI after a change).
The grin's random picks are SMILE's (`Workflow._fill_grin`, from the visitor's seed): each value's peak between its min and max, and
the moment of that peak within midpoint ± deviation, sent to Grin Frames as `<name>_peak` (without them the node draws its own moments).

## Render

OpenGL engine: Poser2's `modules/gl/` (glfw, GL 4.6 core, render thread owned by `WindowManager`). The render loop goes on when `update`, `draw`, an event callback (key, mouse, window, monitor) or
a deferred settings call raises: it is logged, and a monitor that is unplugged does not close the app
(`modules/render/tests/test_window.py`). The render is the display's 1024 x 1280 (`main.py` defaults, swapped by the
default portrait mode): the same 4:5 as the workflow's 896 x 1120 portrait, which stays at the size SDXL is best at and fills
the render as a texture. `render/SmileRender.py` runs the `Sequencer` once per frame (it is the app clock:
Art-Net, OSC and saving hang off its messages), then the stage for the current `SequenceType`
(`stages.py`) renders the layers (`layers/`) and picks the one to show. Images from other threads go
through `inputs.py` (`Image.set_image`, uploaded on the render thread); the grin frames go
through `GrinPlayer.py`, a buffer the generator fills and the render plays on its own clock, with
its timings in the sequence settings. At the capture the diffusion layer starts over from it
(`FlashStage` seeds it, and the capture is the diffusion input until the generator sends an image), so
a generation that fails shows the capture as its result and nothing of the previous visitor.
`render.show_camera` draws the raw camera on top and starts
no new sequence while it is on (`Sequencer.blocked`). Shaders are
`classname.frag` next to their `.py` and hot-reload while running.
The two **walls** of the sphere are secondary windows of the same `WindowManager` (`render.window.secondary_list`, read at
the start: the first entry is wall 0, screens 1-24, the second wall 1; `secondary_fullscreen` borderless on that monitor,
forced on by `-fs` like the main window; a monitor that is missing gets a small window on the primary one, and back when
it is plugged in). `SmileRender.walls` are two `PlayerRender`s (`player/Render.py`, no window of their own)
composited in `update()` after the main screen, and `draw_secondary` blits the wall of the monitor's index in the list
(poser2's pattern, hd_trio). The secondary windows share the main context (textures yes, FBOs and VAOs no), so the
render loop makes the main context current at the start of every frame and before `deallocate` (`WindowManager._render_loop`;
drawing a secondary leaves its context current; the same fix is in poser2).

## Credit

`credit.py` holds the artist's credit (name, URL, the lines file `data/credit_lines.txt`: `CreditLines` reads it again when
it changes and `pick`s a random line, never the last one again, `LINE` when it has none) and makes the QR code (OpenCV's encoder, whole pixels per
module, with its quiet zone). `CreditLayer` draws the code in the bottom right corner of everything the main screen shows, and the caption, the name with the
visitor's line under it (drawn at the capture, `CreditLayer.new_visitor`, into the `credit_line` readout), as a meme caption over
the top (white with a black edge, in the comic font `data/Bangers-Regular.ttf`: the credit is a joke) over the generation, the grin and the result (DIFFUSION, DIFF_LOOP and TIMEOUT, fading in and out), and with `credit_camera` over the camera view too (its
own line, drawn when that view begins: `new_line_due`), from the Render tab's `credit` fields; it is not in the saved clips. The
same pass without the QR code goes into `CreditLayer.clean`, the phone's and the online cell's copy. The walls show the code and the name under it (scaled to the code's width) on one cell instead of its clip,
for one play: it takes a cell when a clip starts there and leaves at that cell's next start (`next_credit_cell`, never the online cell), so no
clip start flashes over it; `credit_qr_players` switches it (the walls read the Render tab's fields every frame), off takes the card away at once. With `credit_players`
on, every cell shows a caption over its clip, the name over the top and a random line from the same file along the bottom, new at every play (`Player._on_play`, not tied
to the clip), drawn into a transparent portrait texture per cell (`PlayerRender._draw_caption`).
`sphere.py` is the one place that knows the sphere and its wiring: 12 ribs holding 5 and 3 screens by turn, heights 0 (top) to 4
(heights 1 to 3 have 12 screens, 0 and 4 have 6), screens numbered down each rib (1-5, 6-8, 9-13 …, the ribs counting to the left, so rib r is at -30r degrees), and screen s on Datapath unit
⌈s/4⌉ output ((s-1) mod 4)+1, unit n = grid column n-1, output k = grid row k-1 (`screen_of_cell`, `cell_of_screen`, `row_screens`,
`cell_angles`): the laughter test (it steps the screens in the sphere's order, 1 to 48 down the ribs, lighting the cell `cell_of_screen` gives), the scrolling name's rows and `data/audio/screen_angles.txt` (generated: `screen_angles_text`,
a test keeps the file equal to it) all come from it; a cabling that differs is changed there.
The **scrolling name** (`credit_scroll.py`, no GL): each word of `credit_name` goes round its own ring of the sphere without end, the first word on the
first of `credit_scroll_rows` (`plan_loops`, `RingPath`: all the ring's screens from the rib at the camera down the rib numbers, since the ribs count to the camera's left,
the visitor's right, and a ticker moves to the visitor's left; `reverse` the other way round; `row_screens`, `screen_cell`, screens counted from 0 there). `CreditScroller`
(stepped from `Main.sequence_callback` with `time.time()`, not during the laughter test) hands each wall its own cells with their place round the ring every `SEND_EVERY`
second (`send_loops` to `Player.scroll_loop`: wall, word, px/s, gap, size, the ring's screens, `started` the time the loop began, the cells with their places),
each word at its own speed (`credit_scroll_speed_1`, `_2`; a speed changed in the panel moves the start
so the word goes on from where it is, `word_start`); `credit_scroll_last` is the readout. The walls
(`PlayerRender.scroll_loop`, `_draw_scrolls`) take the position from that and the clock (a renewal moves nothing), carried once onto `time.perf_counter()` for the frames: on Windows
`time.time()` moves in 15.6 ms steps on Python 3.12, coarser than a frame, which made the word stand still one frame and jump the next and drop a
loop no message renewed for `TIMEOUT`. They draw a word as a ticker reading left to right and moving left: `name_strip` renders it once with freetype into one long BGRA image,
640 high, the ink `credit_scroll_size` tall in the vertical middle, white with a black edge (not the font atlas: too big at that size); each cell on the ring blits the strip
shifted by `scroll_offset` (its place times screen width plus `credit_scroll_gap`, minus the distance travelled round the lap, and the next lap's copy) into its own portrait
texture, over the clips and the credit card, under the lit test cells. Nothing of it is in the recordings.

## Recordings

`save/ClipSaver.py` saves each finished visitor for the players under `output/` (git-ignored):
the grin as `grins/<style>/<id>.mp4` (silent h264 at the full portrait size, at `grin_fps`), the capture and the
portrait as `portraits/<style>/<id>_capture.jpg` and `<id>_portrait.jpg`, and after the grin landed the phone's copy
`shares/<style>/<id>.mp4` (`save/share.py`: the same grin re-encoded with the watermark, "SMILE", a circled "ai" and "by" the credit name, in the bottom
right, no caption, built before the players' clip lands, and the visitor's track as its sound: `ClipSaver.set_share` from `Main` at DIFF_LOOP's start with the seconds
since the flash, read with `video.read_audio`, played on the tape of `audio/tape.py`, the Max patch's spin-up over
the first half second and a wind-down over the last, muxed by `video.make_share`; a share clip that fails is a warning and
the players' clip goes to the phone instead). `<style>` is the visitor's
style, `A` (colourful and happy) or `B` (monochrome and dark); the players read it from the folder, the id is the time. Files are built in `output/temp/`
and renamed into place when the grin is complete, so the players never see a partial file
(a visitor that failed, or was reset, is closed and deleted there).
The clip (its `comment` tag) and the portrait (JPEG comment segments) carry the visitor's **record** as JSON: id, style, description,
the visitor's one seed (portrait, description and grin share it), the settings of its style (not the other's) and of the portrait
as the workflow was filled with them, and the grin as it was drawn: each value's peak and the moment of it (`peak`, `peak_at`), not
the ranges (`Generator.record`, handed to the saver with the final portrait). `save/record.py` reads it:
`python -m apps.smile.save.record <file>` prints it, `--preset <name>` writes it as a preset to `data/settings/<name>.json` (the grin's
ranges closed on the drawn values, so loading it repeats the pose); dropping a clip or portrait on `scripts/clip_info.bat` opens it.
Encoding and decoding run `ffmpeg`/`ffprobe` subprocesses through `modules/video` (ffmpeg
must be on the PATH: the gyan.dev full build); the encoder is picked at startup by a trial encode
(Intel Quick Sync on the iGPU, then Nvidia, then libx264; `sequence.encoder` forces one). Never start
those processes any other way: the apps run under pythonw, so they need `CREATE_NO_WINDOW` and no
inherited handles. The clip player (`player/`: `Player`, one for the sphere, started and stopped by `Main`) decodes each of the newest clips once into RAM
(`ClipPool`, 512x640, about 200 MB per clip) and plays from memory on the 48 cells of the two walls (`sphere.CELLS` each, cell ids
0-47, wall = id // 24; the loaders feed `PlayerRender.set_image` and bang Max on 4115 for wall 0, 4116 for wall 1); the Players tab
(`player/settings.py`) has `pool` the clips kept, `recent`, `recycle_every` and `hwdecode` (read at the start), `fps` the playback
speed (default 20, 0 = the clip's own rate), `spread` a random speed difference per play so the cells drift apart (default 0.1), `trail` a blend
with the previous output (default 0: none), read at every play or frame.
The pool (`pool`, default 30) is two lists. `recent`: the last `recent` recordings of the style (default 4), always playing on every
third of the sphere (`sphere.SECTORS`, by rib: 0-3, 4-7, 8-11, `sector_of_cell`; a cell's loader passes its sector to the pool, `Clip.on` counts
the cells per sector), so a visitor sees their video wherever they stand. `archive`: the other clips, in the order they joined. A new recording joins `recent`; when that holds too many,
its oldest moves to `archive` as the newest there. A clip from disk joins `archive` as its newest. Over its size, the pool
drops the oldest of `archive` (a cell playing it finishes its play), so a recording that leaves `recent` gets a full turn.
The walls show one style, that of the newest recording (`FolderScanner`): a recording is a file newer than every clip
seen, and one in the other style's folder is the switch. The pool then gets that clip and the newest others of its style
as `recent` and a random selection of that style from disk as `archive`, and each cell changes over at the end of its play;
an older file that appears (clips sorted into a folder by hand) is archive and waits on disk.
Recycling keeps the walls changing: whenever no clip has joined for `recycle_every` seconds (default 20, 0 = never), one
clip of the style that is on disk and not in the pool joins `archive`, from a shuffled deck, so all come round before any
repeats; a recording counts as a clip joining, and with every clip of the style already in the pool nothing is loaded.
A cell that finishes first takes a `recent` clip that is on no cell of its third of the sphere; else a clip that is on the fewest cells, of those the
least played, at random among equals and never the one it just played (`ClipPool.choose`), so the pool is spread evenly
over the sphere. Joining `archive` levels a clip's play count, or a recording that was always on a cell would never be drawn there.
The cells start one by one, each in a random place on the next clip that is loaded (going round the
loaded clips again when there are fewer than cells). `scripts/startSMILE.bat` is the main startup: it closes and
restarts everything (ComfyUI, Max, SMILE with its walls).

## Music

`audio/DB.py` picks the track of a visitor's movement from `data/audio/music_map.json` into
`audio.music_id`. `modules.inout.OscSender` (`Settings.audio.osc`; the clip player builds an `OscSenderSettings` of its
own per wall for the `/bang`) sends what `Main.sequence_callback` queues every frame, the newest value per address:
`/music_id`, `/state`, `/progress` and the Audio tab's controls `/sound`, `/music_volume`, `/laughter`,
`/laughter_volume` (both levels in dB; the laughter gets a `dbtoa` in front of its linear multipliers). The panel is
their source: the patch's own controls follow it, and a control moved
by hand in Max snaps back (the speaker button holds until the panel's `sound` changes: Max gets it through
`change -1`, so the DSP is not started every frame). Bools go as ints: python-osc sends a bool as OSC True/False,
which Max's toggles do not take. The patch `data/audio/Smile-Music-Player.maxpat` routes them off `udpreceive 4114`
into its controls; the laughter samples are triggered by the players' `/bang` (4115, 4116) through a gate the
laughter switch controls.
Three speakers on the Realtek jacks (Windows at Quadraphonic, 4 channels, so the black jack's left is 3; Max on
`ad_portaudio Windows WASAPI` at 48000; the README has the steps, Max's `ad_mme` is stereo only): the music in
stereo on channels 1 and 2 (`ezdac~`, `p musicMixer` filters both channels), the laughter panned over 1, 2 and 3
(`dac~ 1 2 3`). A `/bang` carries the cell its clip starts
in; `p laughterSampler` adds 24 for port 4116, looks the screen's angle up in `data/audio/screen_angles.txt`
(`coll screen_angles.txt`, found through the patch's folder; generated from `sphere.py`, `tests/test_sphere.py` checks
the file against it), adds the panel's `laughter_offset` (`/laughter_offset`, degrees) and turns the angle into a gain per
speaker (three `expr`s: speakers at -60, 60 and 180 degrees from the camera, each laugh over its two nearest at
constant power). The gains travel with the note into the `poly~` voice (`SmileSamplepoly.maxpat`), which
applies them only on a note-on, so the note-off from `makenote` does not move a fading laugh. The panel's
`laughter_test` (`audio/ScreenTest.py`, stepped from `Main.sequence_callback`) lights the cell on its wall
(`Player.light`, a call: no OSC between SMILE and its walls): the wall shows the cell white with the screen's number (`Text` into a portrait FBO, blitted with `BlitRotated`
like a clip, so it stands upright on the turned screen). SMILE bangs Max itself with `/test_bang <index> <sample>`
(`Sender` sends a tuple as separate arguments), which `p laughterSampler` takes past the gates (`r testBang`) with
the panel's `laughter_test_sample` (a file of the laughter folder; `ScreenTest.list_samples` numbers the folder as
`polybuffer~ readfolder` does, alphabetical with case ignored) at velocity 100; `/laughter_test` closes the clips' gate meanwhile (an `expr` on
`laughterOn` and `laughterTest`) and the sequencer is blocked, so only the test laugh sounds. The patch is JSON, edited by hand in Max's own formatting (tabs, LF) so the diff stays
readable; Max rewrites the whole file when the patch is saved from Max.

## Online

SMILE Online (branch `online`, a proof of concept): someone outside gets the eye when nobody inside has it. Three parts.
`relay/` is a small aiohttp server that runs on the VPS or, for a test, on the PC (`scripts/start_relay.bat`): it serves
the phone page (`relay/web/`: one camera frame with the terms over it and the bar under it, a field of smileys behind; it
draws the judging itself from the face data), keeps the waiting line (`relay/line.py`, pure logic) and passes messages (`relay/server.py` has the protocol: JSON
text, images as binary with one leading byte); it decides nothing, SMILE starts and ends every turn. `online/`
is SMILE's side: `Link` (a websocket client thread that reconnects, plus an upload thread), `Online` (the turn, stepped
from `Main.sequence_callback`, the only place its phase changes: OFF, IDLE, ASKING, JUDGING, TRANSFORMING; the files
after a turn are a `Delivery` record beside the phase) and
`fake.py`, a stand-in for SMILE at the relay. A turn puts the phone's frames into `Camera` as a third source
(`set_online`, `push_online`; the device and the image are muted meanwhile and back once the capture is taken) and the face finder,
the charge, the flash and the generation run unchanged; `Online.on_face` sends the judging, and from the flash to the
end of the timeout the phone gets the main screen as rendered (`ScreenGrab` in `render/grab.py`, taken in
`SmileRender._render_credit` from a credit pass without the QR code and before the status line, `stream_fps` x `stream_width`,
encoded on the link's thread as `SHOW` frames, paced by the relay to one in flight per phone (`seen` from the page); the phone's own judging is the whole image in pixels while the face is not right and the
page's white as the charge). The phone plays the visitor's track from the flash (`music` message; the relay keeps the
tracks in its own `data/tracks` as AAC, `Link._track_aac` transcodes one the first time it is used, `DB.track_path`) on the tape: SMILE picks the style
and track at the turn's start (`Main.pick_style` from `Online._begin_turn`) and names the track to the phone, which fetches it
during the judging; `Online._send_stage`
sends the sequence's stage and progress (`STAGE_NAMES`, every 0.25 s) and the page follows the Max patch's curves (speed
and low-pass filter, `tapeStage` in `app.js`, floored to what a phone speaker can do; `?debug` shows the sound state); the picture stops at the end of the timeout with a last frame
taken without the caption (`ScreenGrab.last`, sent as `LAST`: the page fades the caption out itself and keeps the frame; the
bloom would brighten it), the turn runs on through the bloom and the fade (sound only) and ends back at the camera. The phone also plays the room's laughter samples (`laugh_files`, uploaded to the relay's `laughs/` at
the first connect, `/api/library`) at random, following `audio.laughter` through the `eye` message. The turn ends
with the sequence; `ClipSaver`'s saved callback gives the files (the recording closes at the last grin frame, `Main.diffusion_callback`, while the
grin still plays), which are uploaded at once (`Delivery`)
and the `result` goes to the phone by its id, so it has the video when its turn ends and the next turn can start meanwhile. The phone sends its camera mirrored, as it
shows it. The token is a file (`data/online_token.txt`, git-ignored), never a setting in a preset.
The **online cell**: the screen `online.screen` (1-48, 0 = none; pick one off the scroll rows) of the walls shows a white card with the
QR code to the phone page (`online.page`, empty = the relay's host, `Link.http_base`) and a line under it while SMILE Online is on
and the relay connected, and the main screen as the phone sees it (`CreditLayer.clean`: the caption, no QR code, no status line)
from the judging to the phone's last frame (`Online.showing`), then the card again; off or disconnected it is a clip cell like any
(`online_cell_mode` in `Player.py`). `Main._online_cell` sets it every frame: `Player.online_cell` (no bang, no caption: its clip plays
unseen) and `SmileRender.set_online_cell`, which the walls take in `update()` (`PlayerRender.set_online`).
Tests: `relay/tests` (the line, the server on aiohttp's test server, the page's files: `test_web.py`) and `online/tests`
(a fake link and camera).

## Settings

Poser2's settings system (`modules/settings/`, its rules in Poser2's CLAUDE.md); root tree in `settings.py`,
presets in `data/settings/` (`Smile` at startup unless `.ui_preset.json` names another), web panel at `http://localhost:667`.
