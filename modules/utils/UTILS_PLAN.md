# Utils plan

Proposed by the utils rescan of 2026-11-24 (deep-flow at 38929e26); not built. Each step names its finding,
**(seen)** in the code or a run, **(deduction)** when inferred. A step checked by a test or pyright is one commit.

## A. Tests

A1. `logging/tests/` moves into `utils/tests/` (`test_crash_log.py`, `test_ring_buffer.py`), beside the geometry,
smoothing and time tests.
Why: CLAUDE.md Packages "Its tests sit in its `tests/` folder"; today `discover -s modules/utils/tests` runs 53 tests and
misses logging's 9 **(seen)**. Check: the utils tests, 62.

## B. Docstrings

B1. Two docstrings that say the wrong thing: `geometry/point.py` `Point2f.is_zero` says "either coordinate", the code
checks both; `logging/crash_log.py` `crash_summary` says "the first line of every log", it returns the count, the first
date and the last line **(seen)**.
Why: CLAUDE.md Docstrings "A docstring says what the thing is". Check: pyright.

## C. Written once

C1. `Point2f` and `Rect` lose their hand-written `__eq__`; the dataclass makes the same one. `Rect.position` and
`Rect.top_left` are one property: `top_left` stays and the callers of `position` move (grepped at the build) **(seen**,
`point.py:60`, `rect.py:38,53,63`**)**.
Why: CLAUDE.md Packages "code that does the same is written once"; Naming "Use no synonyms for one concept". Check:
test_geometry.

## D. Time

D1. `time/timing.py`: `git_commit` reads the repo root as `parents[3]`, where it reads `parents[2]`, `modules/`, since
the move into `time/` **(seen)**; `run_session`'s `stop.wait(warmup)` and `stop.wait(duration)` wait on a `Waker`
**(seen**, `timing.py:56,60`**)**. The file stays in utils: the launcher and `scripts/model_benchmark.py` share it
(chosen 2026-11-24).
Why: the move's stale path is the move's to fix; CLAUDE.md Time "A thread that waits for a moment waits on a
`utils.Waker`". Check: pyright; one launcher timing session whose report holds the commit.

## E. Names

E1. `lfo`'s `time` becomes `timestamp` **(seen)**, before F3.
Why: glossary "A time stamp: `timestamp`; not `time`". Check: pyright.

E2. `Rect.affine_transform` becomes `Rect.denormalized(target)`, with its callers (grepped at the build).
Why: it maps a normalized rect into a target, not a general affine map **(seen)**; CLAUDE.md Naming "Name a concept
after what it is". Chosen 2026-11-24. Check: pyright, test_geometry.

## F. What utils holds

F1. `object_pool.py` goes, with `ObjectPool` from `__init__.py`.
Why: written for the multipose prototype (2025-03-20), whose use went; unused since 2026-02-03 **(seen**, git log**)**;
the repo's pools (gl's `PixelBufferPool`, the screen capture's slots, inference's `StagingRing`, smile's `ClipPool`)
each need their own shape and could not use it **(seen)**. Chosen 2026-11-24. Check: pyright, utils tests.

F2. `thread_priority.py` goes, with `ThreadPriority`, `set_current_thread_priority` and `get_current_thread_priority`
from `__init__.py`.
Why: added 2026-09-09 to raise white_space's light threads, removed from every caller 2026-10-08 after profiling showed
an elevated conductor caused playback stutter (GIL) **(seen**, e8f3e12f5**)**. Chosen 2026-11-24. Check: pyright, utils
tests.

F3. smile's lights use `utils.lfo`, and `lights.py`'s own `lfo` goes; `lfor` scales `utils.lfo`, with the frequency
halved and the phase turned to radians, so the breath looks as before. *Owner: smile*, a minor fix with smile's thread
asked for feedback.
Why: `apps/smile/lights.py`'s own `lfo` is the same 0..1 sine at half its `Htz` (`π·Htz` where `utils.lfo` has
`2π·f`), with its phase in cycles plus 0.75 **(seen)**; written once. Chosen 2026-11-24. Check: smile's tests; the
breath's value at a few times equals the old one.

## Unused but working, which stays

`AngleFilter`, `SpringFilter`, `FPS`'s counts (`count_processed`, `count_dropped`, `dropped()`, their getters, `reset`)
and `PerformanceTimer`'s `get_average`, `get_minimum`, `get_maximum` and `reset`, read only by its own report, have no
user outside utils **(seen)**.

## Rules this plan touches

- Packages: Its tests sit in its `tests/` folder
- Packages: Inside a package, code that does the same is written once
- Docstrings: A docstring says what the thing is
- Time: A thread that waits for a moment waits on a `utils.Waker`
- API evolution: Remove obsolete pathways instead of leaving parallel legacy ones (F1, F2)
- Naming: Name a concept after what it is; use no synonyms for one concept (C1, E2)
- Glossary: A time stamp: `timestamp`
