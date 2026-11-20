# Poser2 — Coding Guidelines

Poser2 is a real-time, low-latency system: app-specific orchestration in `apps/`, reusable infrastructure in `modules/`.

## Writing rules

- One rule per bullet
- Keep a rule short and concrete
- A rule is general: it holds beyond the case that prompted it
- Word a rule as a preference ("prefer"), unless it has no exceptions
- Add a short reason only when it marks the rule's boundary or the rule isn't obvious
- No incident history, no examples unless the rule is ambiguous without one
- A rule describes the codebase as it should look
- Ask what a rule is trying to solve that the other rules do not cover
- Consider a rule's unwanted side effects before adding it
- Put a rule where it loads when the code it governs is written: here, or a `.claude/rules/` file whose `paths` match that code

## Project shape

- Use Python 3.12 and follow PEP 8
- `apps/` may import `modules/`; `modules/` must never import app code
- `modules/settings/` is shared infrastructure; any module may depend on it
- `modules/camera/` is the camera contract; the tracker, inference, board and render import it, never a backend package such as `modules/camera/oak/`

## Packages

- A package is one thing, and its name is that thing
- Its `__init__.py` holds what other packages use
- Its files each hold one class or one subject, named after it
- A job of three or more files has its own folder, named after the job
- Prefer no folder for one or two files
- Inside a package, code that does the same is written once
- Its tests sit in its `tests/` folder
- A script sits in a `scripts/` folder: the repo's, or its app's when only that app uses it

## Naming

- Name a concept after what it is in the domain, and use that one term everywhere it appears: file, class, method, setting, panel label, document
- Similar things share the noun and differ by a qualifier (`Frame`, `CameraFrame`)
- Use the terms in `.claude/rules/glossary.md`
- Use no synonyms for one concept across modules; when a term changes, rename it everywhere in the same change

## Docstrings

- A docstring says what the thing is
- Reasons sit in the package's document, history in the commit

## Settings

- `BaseSettings` subclasses are pure data containers with no runtime side effects
- Use `BaseSettings` for reactive configuration and `@dataclass` for plain value objects
- Components take their settings object (or a `Group` of it) in the constructor, never as a global
- Read a field when it is used; keep a copy only of a `Field.INIT` value, which cannot change after `initialize()`
- Settings connect the user and a component, not two components
- Components that `bind()` must `unbind()` the same callbacks on teardown
- Treat `Field.READ` values as snapshots; copy arrays before handing them to another thread
- Keep `Field` descriptions to one short line of panel text: what the value is and its unit
- Prefer the textbook unit for a quantity (seconds, meters, radians, pixels)
- A setting follows the quantity it controls, in a form that is easy to set (ms for a short time, degrees for an angle)
- Prefer not to change what `initialize()`, `update_from_dict()` and `to_dict()` do; every app's settings files depend on it

## Pose Frame contract

- `frame[FeatureType]` and `group[FeatureType]` never raise; missing data is NaN with score `0.0`

## Composition and wiring

- Prefer composition over inheritance for runtime assembly
- Pass collaborators and fixed structure (counts known at startup) through constructors; tunable values come from settings
- Name public data-flow methods by execution timing: `set` stores for polling, `submit` enqueues deferred work, `process` transforms and emits synchronously, `update` advances on a tick
- Name output channels verb + domain noun (`submit_frames`, `add_similarity_callback`)
- Share runtime data through `modules/board/` protocol + mixin pairs
- A value derived from pose data, a threshold on it included (a gate, a dead zone), is the pipeline's and so are its settings; the show owns only what it does with it: timings, looks, sound
- Fix a problem where it is made, not where it shows

## Concurrency

- Use threads with explicit synchronization, not async/await
- Use `Lock` and `Event` for shared mutable state and lifecycle signals
- A producer calls its callbacks itself, from the thread that made the item. It has no thread for dispatching
- A callback only hands the item over and returns
- An item handed to a callback is not changed afterwards
- An item that leaves its package is finished, its GPU work included
- The part that works on the item has its own thread and does all its work there, its inner steps included
- When that part can't keep up, its queue drops the oldest item and counts it
- A part that only needs the latest value, like the render, reads it at its own tick
- A callback list has its own lock; register under it, copy the list under it, call the callbacks outside it
- A failing callback does not stop the others
- Work that numpy or the GPU does faster is not done in a Python loop; the loop holds the GIL and delays every other thread

## Time

- Read `time.perf_counter()` for every stamp and duration; `time.time()` and `monotonic()` step 15.6 ms on Windows, coarser than a frame
- A thread that waits for a moment waits on a `utils.Waker`; a wait's timeout (`Event.wait(t)`, `Queue.get(timeout=t)`) ends on Windows' 15.6 ms tick
- Downstream of the camera, take time from the frame or the tick instead of reading a clock
- Use the wall clock only for file names and log lines

## Performance

- Prefer the design with the lower latency, measured in the context of the app
- Avoid unnecessary copies

## Types

- Use modern typing (`list[T]`, `dict[K, V]`, `X | Y`)
- Give public methods return type hints
- Use a `Protocol` for what a consumer needs from a collaborator, an `ABC` for a family of classes that share code
- Prefer `IntEnum` over string keys for dict lookups and identifiers
- Fix a type error in this order: correct the annotation, narrow with `assert` or `isinstance` where the code already guarantees the type, `cast` to the type the value really has, and suppress only as a last resort
- Suppress only when a third-party stub is provably wrong or the type system cannot express the code, never to hide an error in our own code
- Suppress with `# pyright: ignore[<rule>]` on the one line; never add a bare `# type: ignore`
- Never disable a rule in `[tool.pyright]` to clear errors
- A commit leaves no pyright errors

## Imports

- Apps and modules import another package from its `__init__.py`, never from its internal files
- Import the package as a namespace (`from modules import X`, then `X.Y`) when the qualifier adds clarity; import names directly otherwise
- Put all imports from one package on a single `from` line
- `__init__.py` defines no `__all__`
- Inside a package, its subpackages included, import relatively; tests may import absolutely

## Error handling

- Define module loggers as `logger = logging.getLogger(__name__)`
- Prefer `logging` over `print`
- Log exceptions with context; never fail silently
- A thread's loop catches an exception per item, logs it and goes on with the next

## Testing

- Rely on running the real-time system for integration; unit-test infrastructure (settings, frame/features, serialization, utility primitives, geometry)

## Running code

- Run Python and tests through Bash, never PowerShell
- Start each Bash command with the program itself (no `cd`, env-var prefix or heredoc) so it matches the allow-rules in `.claude/settings.json`
- Prefer Edit/Write over scripts for changing files
- Run a test of the app or a GPU harness only on the user's go, and only what the plan names
- Before each run, say in one line what it measures and how long it takes
- A run lasts only as long as the test needs
- Stop a run at twice the time you stated, and say so
- Diagnose a failed or hung run from its log and the code before running it again
- Stop the app in a `finally` in any probe that runs it, so a crash cannot leave it running
- Post each result as it comes

```
python -m unittest discover -s tests -t .                  # guard tests: import layering, glossary
python -m unittest discover -s apps/<app>/tests -t .
python -m unittest discover -s apps/<app>/<pkg>/tests -t .
python -m unittest discover -s modules/<pkg>/tests -t .    # camera/oak4 in .venv-oak4
```

## API evolution

- Prefer clean breaks over deprecation windows
- Update all affected call sites in the same change
- Keep settings files in sync with code changes
- Remove obsolete pathways instead of leaving parallel legacy ones

## Plans and commits

- A plan is named after the package, app or folder it is for, and sits in it (`FLOW_PLAN.md` in `modules/flow`)
- A document describes what is built; what is not built is in the plan
- A built step leaves the plan
- A plan ends with the rules it touches, one line each
- A plan is built on the user's literal "go"
- Separate what the code or a measurement shows from what is inferred, and say which is which
- A plan step that is checked by a run or a test is one commit
- The rest of a plan is one commit
- Experiments and half steps are not committed
- A finished plan is reported with the rules it touched, one line each
- Push only on the user's word
- Never delete or rewrite a plan without confirmation; propose the change first

## Threads

- A thread that finds a minor problem fixes it, and asks the thread the code belongs to for feedback
