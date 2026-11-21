# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

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
