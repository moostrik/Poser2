# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

## C. One way to run ffmpeg

C1. `process.py` gets the three things the package now writes several times, and every caller uses them:
   - `run(args, stdin_bytes, timeout, command)`: start, feed, wait, kill on timeout, raise with ffmpeg's error text.
     It replaces the copies in `audio.transcode_aac`, `audio.read_audio`, `share.make_share`, `encoder.encoder_works`
     and `probe._ffprobe` **(seen)**. `_ffprobe` today lets `TimeoutExpired` out without killing ffprobe, so a hung
     probe leaves a process behind **(seen)**.
   - `read_exactly(file, view)`: fill a buffer from the pipe, short only at its end. It replaces
     `clip_decode._read_exactly`, the fill loop in `decode_clip` and `StreamReader._read_frame` **(seen)**.
   - the stderr reader thread, which `decode_clip`, `ClipWriter._start` and `StreamReader._load` each start their own
     way **(seen)**; StreamReader keeps logging line by line.
   Why: CLAUDE.md Packages, "Inside a package, code that does the same is written once"; one place gets timeouts and
   kills right. Check: the video unit tests, and smile's `save` tests (share, audio).

## D. The stream threads

D1. `StreamReader._play` ends cleanly whatever happens: the frame callback is called in a `try` that logs and goes
   on, and `_close` runs in a `finally`. Today an exception in the callback ends the thread and leaves ffmpeg running
   **(seen)**. `_close` closes the pipe before it waits for ffmpeg, with a timeout: on Linux `terminate` is a SIGTERM
   that ffmpeg handles, and it stays blocked writing into the full pipe while `_close` waits forever; four
   `test_stream_reader` tests hang and fail on Linux for this **(seen in a run, Python 3.13 on Linux; on Windows
   `terminate` kills at once)**. Why: Error handling, "A thread's loop catches an exception per item, logs it and goes
   on"; Concurrency, "A failing callback does not stop the others"; keep Linux possible. A new test has a frame
   callback that raises: the reader logs it, plays on, and leaves no ffmpeg running. Check: `test_stream_reader` on
   Windows, and on Linux (a cloud copy), where the hang shows.

D2. `StreamWriter` waits on its work instead of polling: a `None` put on the queue ends `_record` (no
   `get(timeout=.1)`), `_drain` waits on `Queue.join` with `task_done` (no `sleep(0.01)` loop, which steps 15.6 ms on
   Windows), and `is_recording` / `is_receiving` become `Event`s, as they are read on the camera's thread and written on
   another **(seen)**. When the encoder dies, `_clear` marks every frame it drops done, and the end marker too, so
   `stop` and `split` never hang on `Queue.join`. Why: Time, the Waker line and its 15.6 ms tick; Concurrency, "Use
   `Lock` and `Event` for shared mutable state and lifecycle signals"; a `split` then returns as soon as the queue is
   written, not up to 100 ms later per stream **(deduction)**. Check: the video and recording recorder unit tests.

D3. `ScreenRecorder._run` waits for work or for the next estimate (`ESTIMATE_INTERVAL` left), not every `POLL`
   (50 ms); `POLL` goes **(seen)**. Why: it wakes 20 times a second with nothing to do. Check: `test_screen_recorder`.

## E. The tree and what the package exports

E1. `share.py` (`make_share`) moves to smile's `apps/smile/save/share.py`, its only caller, with `_SHARE_QUALITY`
   **(seen: no other caller)**, beside `ClipSaver`. It then imports `modules.video` only through its `__init__.py`:
   `encoder_name` stays exported, and `H264_OPTIONS` and C1's `run` are exported for it. Smile's
   `save/tests` and SMILE.md, where it names the share clip's place, change in the same commit. Why: the share clip's
   overlays and its quality for a phone are smile's decisions, and the package is ffmpeg for any app; smile's thread
   agrees. Check: smile's `save` tests, and smile `-sim` with one visitor, the share clip and `watermark.png`
   written.

E2. The loose files are grouped by job, the root keeping ffmpeg itself (`process`, `hardware`, `encoder`, `probe`,
   `clip_writer`, `clip_decode`, `audio`): `stream/` holds `stream_reader`, `stream_writer`, `depth_frame`; `screen/`
   holds `screen_format`, `screen_frame`, `screen_frame_store`, `screen_recorder`, and `screen_size.py` for
   `ScreenSize` and its size rules (`even`, `preset_size`, `matching_preset`), split out of the 39 KB
   `screen_recorder.py`. The tests' imports from moved files (`test_depth_frame` imports `modules.video.depth_frame`)
   and the paths VIDEO.md names (`modules/video/screen_format.py`) follow. Why: Packages, "A package with many loose
   files groups them in folders by job" (16 files at the root today) and "Its files each hold one class or one
   subject". Check: the video, recording and render unit tests and the guard tests.

E3. `__init__.py` exports only what other packages use. Out, used by no other package on the PC's deep-flow
   **(seen, grep)**: `FFMPEG`, `hwaccel_args`, `HEVC_QUALITY`, `ClipInput`, `master_encode`, `stream_encode`,
   `DEPTH_PIX_FMT`, `DEPTH_QUALITY`, `depth_frame_bytes`, `depth_to_p012`, `p012_to_depth`, `ScreenFrameStore`,
   `ClipSink`, `even`, `preset_size`, `matching_preset`, `frames_for`, `ScreenRecorderState`, and the recorder's
   `quality`, `record` and `take` settings classes. `FFPROBE`, `ScreenRecorderFrameSettings` and `ScreenSize` stay:
   recording's and render's tests and render's compositor use them. In: C1's `run` and `H264_OPTIONS`, for smile after E1. The package's
   tests import the rest from their files. `HEVC_QUALITY` and `master_encode`'s default go: the take's `quality`
   setting is the one value, and `test_clip_hevc` passes a quality. Why: Packages, "Its `__init__.py` holds what
   other packages use"; API evolution, remove the stale default. Check: the video unit tests and the guard tests.

## F. Docstrings and VIDEO.md

F1. Reasons and measurements leave the docstrings for VIDEO.md, which gets a short depth section (3 mm steps, 12 m
   full scale, the 24 mm error measured, no `global_quality`): `depth_frame.py`, `process.py` (pythonw), and the
   reasons in `master_encode`, `encoder_works`, `transcode_aac`, `StreamWriter`'s module docstring and
   `StreamReader.__init__` **(seen)**. American spelling: "colour" (`StreamReader.__init__`) and "Millimetres"
   (`depth_frame.py`) **(seen)**. Why: Docstrings, "A docstring says what the thing is"; "Reasons sit in the
   package's document"; glossary, American spelling. `StreamWriter._open`'s comment ("with B-frames a seek in a
   merged chunk gave depth about 100 mm off, why is not settled") becomes a VIDEO.md constraint: depth is written
   without B-frames, as with them its frames came out in the wrong order **(site fact)**. Check: none, text only.

F2. VIDEO.md matches the code: "Every file is written by `ClipWriter`" is not so for `audio` (and `share` until E1)
   **(seen)**; Limits still treats 10 bits as unbuilt ("10-bit HEVC would halve the duration"); they are built, and
   cost a third more bytes at 4:4:4 and twice at 4:2:0 (`ScreenFormat.frame_bytes`) **(seen)**; the line "The 8 MB
   pipe left the first row as it was through the default pipe" says nothing a reader can use and goes; the file list
   follows E1 and E2. Why: docs, a document describes what is built. Check: none, text only.

F3. VIDEO.md's parts on render's screen recording move to a new `SCREEN_RECORDING.md` in
   `modules/render/general/screen_recording/`, and VIDEO.md links to it: what the compositor records and how
   (`ScreenRecordingCompositor`), "Capture, on the main thread" (`ScreenCapture`, the shaders), the preview in "The
   held take" (`ScreenPreview`), and the capture's constraints (fragment shader, packed 10 bits, `x2rgb10le`).
   VIDEO.md keeps the recorder: ring, hand-over, encode, take, look back, time, states. Why: docs, "A component is
   described in its own document; other documents link to it". RENDER.md's line that names `screen_recording/`
   links to it too. Render's thread checks the result after it is built. Check: none, text only.

## Rules

- Log exceptions with context; never fail silently
- Name a concept after what it is in the domain, and use that one term everywhere it appears
- Inside a package, code that does the same is written once
- A thread's loop catches an exception per item, logs it and goes on with the next
- A failing callback does not stop the others
- A thread that waits for a moment waits on a `utils.Waker`; a wait's timeout (`Event.wait(t)`, `Queue.get(timeout=t)`) ends on Windows' 15.6 ms tick
- Use `Lock` and `Event` for shared mutable state and lifecycle signals
- A package with many loose files groups them in folders by job
- Its `__init__.py` holds what other packages use
- A docstring says what the thing is
- Reasons sit in the package's document, history in the commit
- Its files each hold one class or one subject, named after it
- A document describes what is built; what is not built is in the plan
- A component is described in its own document; other documents link to it
