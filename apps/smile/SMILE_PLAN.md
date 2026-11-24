# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

## G. The OAK4 as a camera

- G4 At `P2160` and 30 fps with a visitor in front of the OAK4, the face finding and the upload keep up: the camera's
  `video_min_fps` and the uploader's dropped count read during the camera states, one live `-oak` start.
  Done: live `-oak` starts with a visitor through (19:26 and 19:42, a clip saved each); a `P2160` recording
  (`20261121-194424_Oak-D Test`, 2704 × 2160, unturned) played with `-sim` on the iGPU decoder, a visitor through,
  the screen and capture upright. The OAK4 turned on its side is the camera's orientation plan's (`CAMERA_PLAN.md`).

## H. The new smile on location

The Mako, the lights and the sphere are on site; the site PC runs the old smile. After `SCRIPTS_PLAN.md` A1, smile
runs in `.venv` (depthai 3). Each check says whether it is fixed (a log line, a readout) or needs Matthias's eye.

- H1 Before leaving:
  - A committed state that passes smile's tests, the guard tests and pyright; no other thread's uncommitted work in
    it.
  - The site's `Smile.json` brought here, committed as it is, then migrated to this version's settings: `online.active`
    to `enabled`, `render.trail_active` to `trail_enabled`, the grin and saver groups under `sequence`; a `-sim` start
    on it, which loads it without a missing-setting warning (fixed).
  - A clean install beside the working checkout, as the site makes it, in `C:\Developer\SmileClean`; the checkout
    and its venvs are not touched:
    1. `git clone` from GitHub of the branch the site pulls (pushed on Matthias's word), `git lfs pull`.
    2. `scripts\install.bat` (`.venv`, depthai 3; after `SCRIPTS_PLAN.md` A1).
    3. `apps\smile\scripts\install.bat --mirror C:\Developer\Poser2\apps\smile\ComfyUI\models`: ffmpeg, ComfyUI,
       its models and the smoke test (fixed). The site mirrors from its old install the same way.
    4. `scripts\makemodels.bat smile`: the face detector's engines, built from the clone's own ONNX (fixed).
    5. By hand, not in git by design: the online token, `Smile.json`, one recording for `-sim`; ffmpeg and the
       Vimba SDK are the PC's.
    6. smile's tests in the clean `.venv`, one `-sim` start with a visitor through (fixed).
    Anything missing beyond 5 is a defect, fixed in the install or the README; the list in 5 goes into README.md's
    setup.
- H2 Install on site:
  - No app runs. `git pull`; `scripts\install.bat` (`.venv`, depthai 3); `apps\smile\scripts\install.bat`: ffmpeg,
    ComfyUI, its models and the smoke test pass (fixed).
  - The face detector's TensorRT engine is built for the site's GPU: `scripts\makemodels.bat smile`; the log says
    `Face detector: … ready (TRT)` (fixed).
- H3 Start and camera:
  - `startSMILE.bat`; the log says `Camera device vimba connected` and no `gone` (fixed); the packet size line 8228 or
    more (ON SITE CHECKS 11, fixed).
  - The Mako in portrait: the camera screen upright and the right way round, the face marks on the face, the captured
    portrait upright (eye). The Mako's turn is `camera.rotate90` (on, a mount setting in the settings file); its flips
    are `camera.flip_h` / `flip_v` in the panel.
  - The flash against the capture: the captured frame is the one the flash lit (eye).
- H4 Lights and sound:
  - Art-Net: the lights flash at start and when `artnet.enabled` comes on, follow the show, and go black when it is
    switched off and at stop (eye); the senders are new (`modules/inout`).
  - Max: the music on `/music_id`, `/state` and the controls (OSC 4114), the laughs on `/bang` from both walls (4115,
    4116) (eye); the laughter test steps 1 to 48 and each laugh comes from its lit screen (ON SITE CHECKS 8, eye).
- H5 The walls on the sphere:
  - The log says `48 cells playing, 30 clips`, and no `frames/s per cell` warning (fixed).
  - The screen order (the laughter test's numbers on the right screens), the credit card, the captions and the
    scrolling name (ON SITE CHECKS 15, 16; eye).
- H6 A visitor and the rest:
  - One full visitor: flash, portrait, grin, bloom, the clip saved and on every third of the sphere within seconds
    (ON SITE CHECKS 17); its share clip in `output\shares\` with the track and the watermark (eye).
  - SMILE Online, if it is used: `online.enabled` on, the QR card on its screen, one phone turn (ON SITE CHECKS 18, eye).
  - After a few hours: the newest `logs\smile_*.log` has no unexplained `WARNING`, `ERROR` or `CRITICAL`, a few dozen
    lines plus about one per visitor (ON SITE CHECKS 6, fixed). The checks that passed get their date in README.md.

## Open questions

- Recordings of real visitors: a few recorded with the Vimba camera (or a webcam) at the PC or on site, the camera
  in portrait. The PC the plan was built on has neither; its one recording (`data/recordings/20261121-110540`, 1280 × 960) is a
  photo handed by a stand-in webcam, turned and flipped as the camera delivers.
- README.md's SMILE Online section keeps a **not yet** list: unbuilt work in a document, which `docs.md` puts in
  the plan. Move its items here as steps?

## The rules it touches

- `CLAUDE.md` Running code, a run only on the user's go, what it measures and how long said first: G4
