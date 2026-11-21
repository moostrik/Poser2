# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

Each step is one commit. Every step runs the guard tests, the tests of the smile packages it touches and pyright.
A step that moves or rewires code also starts smile once: ComfyUI first (`scripts/start_comfy.bat`), then
`launcher.py -app smile -sim` on a recording; it reaches the camera screen with the walls playing,
takes a recorded visitor through (flash, portrait, grin, bloom, the clip saved and on the walls) and leaves no error
in the log. A rename or a text change needs no start. The whole of smile's tests run after the last step.

## C. Wiring and time

- C1 The view: the fit to the output size becomes a plain geometry module, which the render's camera layer draws on
  the GPU; the face finder maps the detector's faces (found on the full frame) into it, counting a face only inside
  and measuring `OUTSIDE` against the view's edge; `cover` runs once, at the capture. `cam` hands the full
  `CameraFrame`s on, and the uploader takes them through `cam`'s `CameraFrameCallback`, not from `Main` polling
  `camera.frame()`.
  Check: the start check; `cam/tests`, `face/tests` (a frame larger than the view), `render/tests`, `save/tests`,
  `tests`; a test of the view geometry.
- C2 A board, `apps/smile/board/`, in place of `SmileRender`'s setters and its `Inputs` and `Faces`: `modules/board`'s
  `CameraImageStoreMixin` holds the full uploaded frame, so the render no longer uploads it a second time, and smile's
  own slices hold the faces, the capture and its depth, the generator's images, the status and warm-up, and the online
  cell. The generator's `CAM_DEPTH` still starts the reveal through `Main`. Check: the start check; `render/tests`,
  `tests`, a test per slice.
- C3 The five callback lists (`Camera`, `FaceFinder`, `Generator`, `Sequencer`, `ClipSaver`) become `utils.Broadcast`:
  callbacks run outside the lock, and one that fails no longer stops the others; `ClipLoader`'s go with C6.
  Tests: `cam/tests`, `face/tests`, `comfy/tests`, `sequence/tests`, `save/tests`; a test that a failing callback
  leaves the next one called.
- C4 `ClipSaver` makes its folders, picks its encoder and starts its thread in `start()`, not in its constructor;
  `Main` starts it with the other workers after the wiring. It reads the credit name at each share, through a getter
  as `Online` has, and writes `watermark.png` again only when the name differs from the file's. The frame's tick is
  `SmileRender.update`: it reads `perf_counter()` once and hands that time to `Sequencer.update(now)`,
  `GrinPlayer.update(now)`, the credit and the screen grab; `SequenceMessage` carries it as `timestamp`, and `Main`
  takes its time from the message. Tests: `save/tests` (a `ClipSaver` built without a thread; a share after a name
  change carries the new name), `sequence/tests` stepping time without sleeping, `render/tests`, `tests`.
- C5 The online cell's mode (`'clip'`, `'qr'`, `'screen'`) becomes an `IntEnum`, and cells are ints, not `str(i)` keys.
  Tests: `player/tests` (`test_online_cell`), `online/tests`, `render/tests`.
- C6 The 48 clip cells play from one thread: a heap of the cells by their next frame's deadline and one `Waker` for
  the earliest; at each wake the due cells hand their frame to their wall and take their next deadline. Each cell keeps
  its own clock, speed and spread (`fps`, `spread`), its clip order and its bang as now; the 48 `ClipLoader` threads go.
  Check: the start check, the 48 cells playing and drifting apart as before; `player/tests`.

## D. Names

- D1 Renames, settings files migrated in the same commit:
  - `online.active` to `online.enabled`, `render.trail_active` to `render.trail_enabled` (`Smile.json`)
  - `FrameCallback` in `online.link` (the phone's JPEG bytes from the relay) qualified as `PhoneFrameCallback`
  - `music_map.json`: each style's `movement` (its label, which no code reads) to `name`, its `audio_ids` to
    `music_ids`, and the read in `music_map.py` with it
  - in `Main`: `self.face` to `self.face_finder`, `self.save` to `self.clip_saver`

  Tests: `tests`, `audio/tests`, `cam/tests`, `online/tests`.

## E. One place for each fact

- E1 The caption and the QR card are drawn once (now in the credit layer and again in the wall compositor);
  `credit_scroll.py`'s zero-based screen numbering (`row_screens`, `screen_cell`) and its repeated constants move
  into `sphere.py`; the data, output and recording paths (`apps/smile/data/…` in six places, `apps/smile/output/` in
  two, `apps/smile/data/recordings/`) are written in one place.
  Tests: `tests` (`test_sphere`, `test_credit_scroll`), `player/tests`, `render/tests`, `save/tests`.

## F. Documents

- F1 SMILE.md and README.md: the settings paths (`Settings.setup.camera`, `Settings.setup.audio.osc`), `decoder` for
  `hwaccel`, SMILE Online no longer "branch `online`, a proof of concept", American spelling, a link to `smile.md`'s
  Names for "visitor" (the exception to the glossary's "player"), `smile.md`'s rules linked instead of restated;
  "the players" reading `output/grins/` becomes the clip player, README's "player 1/2" for the walls becomes "wall",
  and "movement" for a style becomes "style" (SMILE.md twice, README.md once); the docstrings that name
  `modules/online/`, `modules/save/…` and `modules/audio/tape.py` name their real place in `apps/smile/`.
  Tests: none (text only).

## Open questions

- Recordings of real visitors: a few recorded with the Vimba camera (or a webcam) at the PC or on site, the camera
  in portrait. The PC the plan was built on has neither; its one recording (`data/recordings/20261121-110540`, 1280 × 960) is a
  photo handed by a stand-in webcam, turned and flipped as the camera delivers.

## The rules it touches

- `glossary.md`, a settings class is named after the class it configures: B2
- `CLAUDE.md` Imports, a package's `__init__.py` is its public boundary: B1, B2, B3
- `CLAUDE.md` Project shape and Packages, packages independent, one job per folder: B1, B2
- `rendering.md` and the glossary, a `Compositor` owns an FBO and draws other things into it: B1
- `CLAUDE.md` API evolution and Naming, code that does the same is written once, one term everywhere: B4, E1
- `CLAUDE.md` Composition and wiring, share runtime data through `modules/board/`; `board.md`: C2
- `CLAUDE.md` Concurrency, a callback list is a `utils.Broadcast`: C3
- `app-lifecycle.md`, start workers only after the wiring: C4
- `CLAUDE.md` Settings, read fields when they are used, never unpack them into constructor parameters: C4
- `CLAUDE.md` Time, take time from the frame or the tick downstream: C1, C4
- `CLAUDE.md` Types, `IntEnum` over string keys: C5
- `glossary.md`, `enabled` not `active`, `FrameCallback` for the pose frame and every other frame type qualified,
  `timestamp`, American spelling; `smile.md`, `music_id` / `style_id`: B2, C4, D1, F1
- `camera.md`, the camera's thread runs only the camera, and `modules/camera`'s `CameraFrame`, a frame as delivered: C1
- `CLAUDE.md` Testing, geometry in plain modules, drawing in renderers: C1
- `smile.md`, `sphere.py` is the one place that knows the sphere: B1, E1
- `docs.md`, each fact in one place: F1
- `CLAUDE.md` Concurrency, the part that works on the item has its own thread: C6
- `CLAUDE.md` Time, a thread that waits for a moment waits on a `utils.Waker`: C6
- `CLAUDE.md` Plans and commits, a plan is named after the package, app or folder it is for, and sits in it: this
  plan, in `apps/smile`
