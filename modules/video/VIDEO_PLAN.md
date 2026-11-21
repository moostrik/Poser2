# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

## E. The tree and what the package exports

E1. Check (built, its unit tests pass): smile `-sim` with one visitor, the share clip and `watermark.png` written.

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
