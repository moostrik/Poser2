# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

## A. The screen recorder's queue

A1. VIDEO.md's hand-over says why a full queue drops its oldest frame (`QUEUE_LIMIT`, `submit_screen_frame`):
   waiting for the recorder's thread would stall the render tick and the show, and a drop needs the thread to fall
   three ticks behind; no measured take dropped a frame, and the count shows on the panel (`dropped`) and in the
   file's comment **(seen)**. Why: his choice (2026-11-23); "Reasons sit in the package's document". Check: none,
   text only.

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
