# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F). Who uses what, from the code: the contract
(`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture match) is read by
every app and by inference, recording, tracker and render. The backends are each one kind of camera, used now by
the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, `vimba` and `webcam` by smile. That is
general code with one user, which stays. One thing in a backend is one app's: the Vimba camera's user set. One step
per commit, each checked by the tests named and every app's start (-sim, smile on its image input, 10 s).

Built: 1, Vimba's user set is a setting, `VimbaCameraSettings.user_set` (empty loads none), `UserSet2` in smile's
four presets; 2, the four backends' callback lists have their lock, registered under it, copied under it, called
outside it; 3, a camera's number is `cam_id` on the contract and every backend (the replay camera's too), Vimba's
serial is `serial`, the package spells American, its tests time with `perf_counter`, and OAK4.md's open work is
below.

## Open

The OAK4's unbuilt and unmeasured work, from [OAK4.md](oak4/OAK4.md), with what was measured of it.

- NPU load is not read. depthai does not report it, and `oakctl device stats` gives `dsp` null with a pipeline running. `sysMonApp` on the camera reads the NPU through `/dev/fastrpc-cdsp`, which only the `system` user can open, and the ADB shell runs as `adb`, so it fails with a FastRPC error. It needs a root shell through `oakctl device unlock`, which has not been tried.
- depthai once stopped listing the camera on USB after an app closed, and kept listing it on Ethernet, while ADB and Windows still saw it on USB. Every open fell back to Ethernet until `oakctl device reboot` brought USB back. The cause is not known.
- `SpatialDetectionNetwork` has not been measured with a real player, whose larger box holds more depth pixels, or with several players. Which depth frame its `z` comes from is not known: boxes reach the PC before that frame's depth does.
- Video min sometimes reads about 30 fps **(site fact)**. The cause is not known.
- `view_depth` has not been seen with a player in view live.
- A standing player reading as one distance from head to feet has not been checked.
- Depth has one outlier of about 153 mm near the top left on synthetic frames, at all widths. The cause is not known.
- `StreamReader.frame_count` can read one short on an `.mkv`, as it estimates from the duration.
- Ethernet stalled once in 130 s of streaming, with a gap of 4.8 s and latency up to 1.7 s. The cause is not known, and a USB run of that length has not been compared. That was at `P720`; `P800` loads Ethernet more and has only run for 10 s.
- A player at show distance has not been compared across depth sizes or IR lights.
- Detection at 640x512 has not been compared with 416x416 on a player, and its latency has not been measured with a player in view.
- `P800` and `P1080` have not been run in the app, and `P800` has not been recorded and replayed.
- `auto_exposure_lock` did not hold: with it on, `exposure_compensation` +9 still moved ISO to 16585. Whether it holds against a change of light has not been tested.
- `exposure_region`, `auto_balance_lock` and `anti_banding` changed nothing in a dark room, where exposure was already at its ceiling and nothing flickered. They need a test with light. `chroma_denoise` 4 lowered mean luma from 24.0 to 22.1, so it does reach the sensor.
- The temporal filter has only been measured on a still scene. How far depth trails a moving player has not been measured.
- Why one run's capture times lay after now ([Capture time](oak4/OAK4.md#capture-time)): perhaps a camera reopened seconds after another process closed it.
- A manual exposure past the frame time is reported as asked and the rate stays at 60 fps, up to 100000 µs, which cannot be real. In a dark room the picture's brightness did not follow exposure at all, so the real top needs a test with light.
- `test_a_split_keeps_the_streams_in_step_across_chunks` fails about 2 runs in 5, because a split lands 1 to 3 frames early. No frames are lost: 50 of 50 in all three streams.
- The camera is planned to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been measured. A Script node is too slow for it **(deduction)**.

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
