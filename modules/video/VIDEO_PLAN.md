# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

## E. The tree and what the package exports

E1. Check (built, its unit tests pass): smile `-sim` with one visitor, the share clip and `watermark.png` written.

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
