# smile's plan

smile (`apps/smile`, not the modules it uses) held up to the rules (2026-11-13): what does not meet them, one step
each, in order, each one commit, checked by the tests named. After the last step smile runs once on the image input
(`use_image`, 20 s, as before the plan: 25 camera frames a second, a face in every one; with `credit_scroll` on, to
see the name scroll on the walls) and, with ComfyUI started
(`scripts/start_comfy.bat`), through one whole visitor. Its command-line tools (`comfy/models.py`, `comfy/smoke.py`,
`online/fake.py`: their folder, logger names and `print`) need no change now (settled in the rules thread, 06fd4666).

## 1. The camera's frames fitted on the camera source's thread

`Camera._on_device_frame` fits every device frame to the render size (`cover`, a cv2 resize) on the device's thread
before it hands it on (`cam/camera.py`); the phone's frames are fitted the same way on the online link's thread
(`process_online_frame`, from `Online._on_frame`). The device and the phone hand their frame over and return (the
newest kept, a wake: `submit_online_frame` for the phone's, named by its timing); the camera source's own thread
(`_run`, which already emits the image input) fits the newest and hands it to the callbacks. One thread fits every
source. The thread waits on that wake, with the image input's pace (`IMAGE_FPS`) as its timeout, not on its
`sleep(0.01)` poll, which would hold a device frame up to 10 ms; its loop catches what the fit or a callback raises,
logs it and goes on (`_emit` calls the callbacks with no `try` today).
Checked by: `cam/tests/test_camera.py`: a device frame and a phone frame in, each fitted and out on the source's
thread; `test_relays_device_frames_and_drops_them_with_the_image_on` and `test_a_frame_has_the_time_it_arrived`,
which expect the frame on the device's own call, wait for the source's thread.

## 2. smile's own times on `perf_counter()`

smile times on `time.time()` (15.6 ms steps) and `monotonic()` (the same) where the rules ask `perf_counter()`:

- `main.py`'s app clock, which steps the laughter test and the scrolling name; `credit_scroll.py` hands its loops'
  start times to the walls, and `player/wall_render.py` turns them onto `perf_counter` (`started_perf`), carrying
  the 15.6 ms step into the scroll's position; its lit cells (`lit_cells`) are timed on `time.time()` too. All of
  them take `perf_counter()` and the conversion goes.
- `comfy/generator.py`: the job's duration, its deadlines and its silence checks.
- `lights.py`: the fade's clock.
- `player/player.py` (the gaps between clip starts) and `player/clip_loader.py` (its pacing).
- `online/link.py`: the interval between two warnings.
- `comfy/smoke.py`: its deadlines (`monotonic()`).

`relay/server.py` keeps the wall clock: it runs on the online server, not in the app. No app time leaves the
machine: the online turn sends the phone durations only (`_music_offset`, the seconds since the flash) and times its
messages by intervals (`_send_eye`, `_send_stage`).
Checked by: smile's tests (`apps/smile/tests`), the scroller's and the laughter test's on the new clock.

## Rules it touches

- The camera's thread runs only the camera; a callback only hands the item over and returns: 1.
- A worker thread's loop catches what its work raises, logs it and goes on (smile): 1.
- Read `time.perf_counter()` for every stamp and duration: 2.
