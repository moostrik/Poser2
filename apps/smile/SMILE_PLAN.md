# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

## G. The OAK4 as a camera

- G1 `modules/camera/oak4`:
  - a `rotate90` argument, carried on every frame as the flag, as the Vimba and webcam backends do; no pixel work on
    the camera's thread. `flip_v` is carried as a flag for color too, beside `flip_h` (now ignored for color in
    `pipeline.py`): turn and `flip_h` make one quarter turn, turn and `flip_v` the other.
  - tilt correction and `rotate90` together: a warning, and no homography. On a camera on its side the IMU's tilt is a
    rotation about the sensor's other axis, which the correction does not model.
  - a mode `P2160` in `geometry/sensor_modes.py`, color only, on the sensor's 4K mode: 2160 rows (already on the
    warp's alignment of 16), delivered by `direct_size` as 2704 × 2160 (5:4, the width on 16). USB only: on Ethernet
    the camera takes `ethernet_resolution`, as now. `CameraResolution` is shared with the OAK-D backend:
    `modules/camera/oak/sensor.py`'s `COLOR_MODES` gets `P2160` (`THE_4_K`).
  - CAMERA.md's mode table and OAK4.md's frame size section say so; OAK4.md says the exposure region is in the
    delivered frame's coordinates, before the turn.

  Check: the oak4 backend's tests in `.venv-oak4` and the OAK-D backend's in `.venv`, with `P2160` cases in
  `test_warp_mesh` and `test_frame_size`; on the connected OAK4 over USB 3 at `P2160` and 30 fps, the rate held, the
  color latency and the link's Mbit/s (about 2.1 Gbit/s of NV12, calculated), a row in OAK4.md's frame size table. If
  `P2160` does not hold 30 fps on USB, the step ends with those numbers and takes `P1440` (1440 rows) instead.
- G2 `modules/camera`: `bgr_frame(frame)`, the frame with its NV12 image converted to BGR (`nv12` off), any other
  frame as it is. `upright_image` keeps its one job and its assert. The OAK4's color frames are NV12, so smile's
  portrait crop (`FaceFinder.best`) and capture fallback (`Main`) call `upright_image(bgr_frame(frame))`. NV12 is
  converted before the turn because a quarter turn would split its interleaved color pairs.
  Tests: `modules/camera/tests`.
- G3 smile runs in `.venv-oak4` for good: `scripts/startSMILE.bat`, `scripts/install.bat` (`VENV_DIR`),
  `scripts/clip_info.bat`, `scripts/make_test_workflow.bat` and README.md's commands. First check that the vendored
  Vimba code and the rest of smile's dependencies import there (the venvs differ only in depthai, `pyproject.toml`).
  An `-oak` start beside `-wc` picks the OAK4 (`modules/camera/oak4`, the one OAK backend smile imports), with an
  `oak4` group (`Oak4CameraSettings`) under smile's camera settings: `model_chain` NONE, `depth_model` OFF,
  `tilt.enabled` off (smile needs no tilt), `resolution` G1's mode, 30 fps. The OAK4 takes its flips and fps from
  that group, as the replay camera does from `replay`; smile's own camera fields stay the Vimba camera's. Portrait
  mode hands `rotate90` as it does to the other two. The settings files get the group; SMILE.md says how to start it.
  Resolution: the portrait is cut from the full frame around the face (`FaceFinder.best`, `portrait_crop`), so the
  face's pixels count, and a visitor further away is sharper on a larger frame. At `P2160` the frame is 2704 × 2160,
  turned 2160 × 2704: twice the pixels per side of P1080 (1344 × 1072) and of the Mako (960 × 1200, turned).
  Check: smile's tests, the guard tests and pyright; on the connected OAK4 (now unturned) a live `-oak` start in
  landscape and one with `rotate90` on, the link readout USB 3, the screen, the face marks and the capture upright as
  the frames come, one visitor through; at `P2160` and 30 fps the face finding and the upload keep up, no frames
  dropped; a recording at `P2160` with the OAK4 and its `-sim`. The iGPU decoder stalled on a 3088 × 2316 recording
  in A3 (`VIDEO_PLAN.md` A1): if a `P2160` recording does not play with it, `-sim` plays it on the CPU decoder
  (the simulator's `decoder`) and the step says so.

## Open questions

- Recordings of real visitors: a few recorded with the Vimba camera (or a webcam) at the PC or on site, the camera
  in portrait. The PC the plan was built on has neither; its one recording (`data/recordings/20261121-110540`, 1280 × 960) is a
  photo handed by a stand-in webcam, turned and flipped as the camera delivers.
- README.md's SMILE Online section keeps a **not yet** list: unbuilt work in a document, which `docs.md` puts in
  the plan. Move its items here as steps?
- `CaptionRenderer` (`render/layers/caption_renderer.py`) holds its font and title textures, where `rendering.md` has a
  `Renderer` own no texture: a name that fits better, or the textures handed in?

## The rules it touches

- `camera.md`, the camera's thread runs only the camera: G1
- `CLAUDE.md`, fix a problem where it is made: G1 (tilt off in smile's settings, the backend only warns), G2
- `CLAUDE.md` API evolution, all call sites in the same change (`CameraResolution` in both OAK backends): G1
- `docs.md`, a measurement in a table, with the configuration it assumes: G1
- `docs.md`, each fact in one place (the OAK4's flips and fps in its group): G3
- `camera.md`, an app imports one OAK backend, `oak4` in `.venv-oak4`: G3
- `app-lifecycle.md`, the settings schema in sync with the settings files: G3
- `smile.md`, `cam` is the app's image source, a `modules/camera` backend among others: G3
