# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F). Who uses what, from the code: the contract
(`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture match) is read by
every app and by inference, recording, tracker and render. The backends are each one kind of camera, used now by
the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, `vimba` and `webcam` by smile. That is
general code with one user, which stays. One thing in a backend is one app's: the Vimba camera's user set. One step
per commit, each checked by the tests named and every app's start (-sim, smile on its image input, 10 s).

Built: 1, Vimba's user set is a setting, `VimbaCameraSettings.user_set` (empty loads none), `UserSet2` in smile's
four presets; 2, the four backends' callback lists have their lock, registered under it, copied under it, called
outside it.

## 3. Names and form

- A camera's number is `cam_id` (the glossary), not `id`: `CameraSource.id` and every backend's `self.id`, with
  their callers.
- Vimba's `camera_id` is the camera's serial (a string, None for the first found), not a camera number: `serial`.
- American spelling: "centre" and "colour" in log lines, docstrings, comments, two panel descriptions
  (`OakCameraSettings.resolution`, "P1080 is colour only"; `VimbaCameraSettings.saturation`, "Colour saturation")
  and test names.
- The two tests that time their waits with `time.time()` use `perf_counter` (`vimba/tests/test_camera.py`,
  `webcam/tests/test_camera.py`).
- OAK4.md's Open section (unbuilt and unmeasured work) moves to this plan's Open, as the documents rule puts it; its
  measurements move with it whole.

Checked by: the camera tests in both venvs and the guard tests.

## The rules file

`camera.md`, line by line:
- "The camera's thread runs only the camera": true of every app's wiring (the cameras' callbacks only queue); about
  this module; checkable in the apps' `main.py`.
- The two-venv line: true (`oak` and `oak4` call `require_backend` on import, which raises in the wrong venv); about
  this module. Its naming of `oak_backend()` went to the rules thread and is fixed (212331ae).
- Missing: the registration before `start()` that CAMERA.md's contract states clashes with CLAUDE.md's callback-list
  lock; with step 2 the contract line goes and no rule is needed.

## Rules it touches

- What one app decides sits in that app (`modules/` holds what is general): 1.
- A callback list has its own lock: 2.
- Use the glossary's terms (`cam_id`, no `camera_id`, American spelling): 3.
- Read `perf_counter` for every stamp and duration: 3.
- Open questions and unbuilt work sit in the plan, not the document: 3.
