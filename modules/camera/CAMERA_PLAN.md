# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F). Who uses what, from the code: the contract
(`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture match) is read by
every app and by inference, recording, tracker and render. The backends are each one kind of camera, used now by
the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, `vimba` and `webcam` by smile. That is
general code with one user, which stays.

Built: 1, Vimba's user set is a setting, `VimbaCameraSettings.user_set` (empty loads none), `UserSet2` in smile's
four presets; 2, the four backends' callback lists have their lock, registered under it, copied under it, called
outside it; 3, a camera's number is `cam_id` on the contract and every backend (the replay camera's too), Vimba's
serial is `serial`, the package spells American, its tests time with `perf_counter`.

## Open

- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.

## The rules file

`camera.md`, line by line:
- "The camera's thread runs only the camera": true of every app's wiring (the cameras' callbacks only queue); about
  this module; checkable in the apps' `main.py`.
- The two-venv line: true (`oak` and `oak4` call `require_backend` on import, which raises in the wrong venv); about
  this module. Its naming of `oak_backend()` went to the rules thread and is fixed (212331ae).
- Missing: nothing; the contract's registration before `start()`, which clashed with CLAUDE.md's callback-list lock,
  went with step 2.

## Rules it touches

- What one app decides sits in that app (`modules/` holds what is general): 1.
- A callback list has its own lock: 2.
- Use the glossary's terms (`cam_id`, no `camera_id`, American spelling): 3.
- Read `perf_counter` for every stamp and duration: 3.
- Open questions and unbuilt work sit in the plan, not the document: 3.
