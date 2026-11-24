# Utils plan

Proposed by the utils scan of 2026-11-22 (deep-flow at 3c7a387d), go 2026-11-24; what is here is not built yet.
Each step names its finding, **(seen)** in the code or a run, **(deduction)** when inferred. A step checked by a test
or pyright is one commit. Steps marked *owner* change another package: a minor fix, with that package's thread asked
for feedback.

## F. What utils holds

utils holds small tools with no domain that many packages share, and imports nothing else of the project **(seen**:
its files import only each other, the standard library and numpy**)**.

F3. The PC's `modules/oak` and `modules/tests` folders go: they hold only `__pycache__` folders, no code and nothing
in git **(seen)**. No commit.

F4. `waker.py`, `fps.py`, `performance_timer.py`, `timing.py` and `time_series.py` move into a `time/` folder in
utils; the rest stay at the root beside `geometry/`, `smoothing/` and `logging/`. Callers import from `__init__.py`
and do not change.
Why: the root holds 10 loose files after B1, D1, D2 and F1, five of them about time **(seen)**; CLAUDE.md Packages "A
package with many loose files groups them in folders by job"; in utils, not a `modules/time` package, since these are
shared tools with no domain (chosen 2026-11-24). Smoothing is its own folder (D1); `lfo.py` stays at the root.
Check: pyright, utils tests.

## Rules this plan touches

- Concurrency: A callback list is a `utils.Broadcast`
- Naming: Use no synonyms for one concept across modules
- Error handling: Log exceptions with context; never fail silently
- API evolution: Remove obsolete pathways instead of leaving parallel legacy ones
- Docstrings: A docstring says what the thing is; reasons sit in the package's document
- Packages: Inside a package, code that does the same is written once
- Packages: Its files each hold one class or one subject, named after it
- Settings: Use `@dataclass` for plain value objects
- Glossary: American spelling
- Testing: unit-test infrastructure
- Composition and wiring: Prefer composition over inheritance (D1)
- Time: Take time from the frame or the tick downstream instead of reading a clock (D1)
- Imports: Import from a package's `__init__.py`, never from its internal files (F2)
