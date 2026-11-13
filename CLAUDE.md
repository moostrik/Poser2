# Poser2 — Coding Guidelines

Poser2 is a real-time, low-latency system: app-specific orchestration in `apps/`, reusable infrastructure in `modules/`.

## Writing rules in this file

- One rule per bullet
- Add a short reason only when it marks the rule's boundary or the rule isn't obvious
- No incident history, no examples unless the rule is ambiguous without one
- A rule describes the codebase as it should look
- Put a rule where it loads when the code it governs is written: here, or a `.claude/rules/` file whose `paths` match that code

## Project shape

- Use Python 3.12 and follow PEP 8
- `apps/` may import `modules/`; `modules/` must never import app code
- `modules/settings/` is shared infrastructure; any module may depend on it
- `modules/camera/` is the camera contract; the tracker, inference, board and render import it, never a backend package such as `modules/camera/oak/`
- `apps/smile/` has its rules in `.claude/rules/smile.md` and its document in `apps/smile/SMILE.md`; its `cam` is the app's image source (a `modules/camera` backend, an image folder, a phone), not a camera backend

## Packages

- A package is one thing, and its name is that thing
- Its `__init__.py` holds what other packages use
- Its files each hold one class or one subject, named after it
- Its document opens with a table of its files: the name, what it is, who uses it
- Its tests sit in its `tests/` folder
- Command-line scripts sit in the repo's `scripts/` folder

## Naming

- Name a concept after what it is in the domain, and use that one term everywhere it appears: file, class, method, setting, panel label, document
- Use the terms in `.claude/rules/glossary.md`
- Use no synonyms for one concept across modules; when a term changes, rename it everywhere in the same change

## Docstrings

- A docstring says what the thing is
- Reasons sit in the package's document, history in the commit

## Settings

- `BaseSettings` subclasses are pure data containers with no runtime side effects
- Use `BaseSettings` for reactive configuration and `@dataclass` for plain value objects
- Components take their settings object (or a `Group` of it) in the constructor, never as a global, and read fields when they use them
- Never unpack setting values into constructor parameters; a copied value goes stale when the panel changes it
- Cache only `Field.INIT` values, since they cannot change after `initialize()`
- Use `bind()` only to react to configuration changes, never to relay runtime data between components; the one exception is a measured frame rate feeding another component's time-base setting
- Keep bound callbacks thread-safe, and never write the field that triggered the callback
- Components that `bind()` must `unbind()` the same callbacks on teardown
- Treat `Field.READ` values as snapshots; copy arrays before handing them to another thread
- Keep `Field` descriptions to one short line of panel text: what the value is and its unit

## Pose Frame contract

- `frame[FeatureType]` and `group[FeatureType]` never raise; missing data is NaN with score `0.0`

## Composition and wiring

- Prefer composition over inheritance for runtime assembly
- Pass collaborators and fixed structure (counts known at startup) through constructors; tunable values come from settings
- Move data between components with callback pipelines
- Name public data-flow methods by execution timing: `set` stores for polling, `submit` enqueues deferred work, `process` transforms and emits synchronously, `update` advances on a tick
- Name output channels verb + domain noun (`submit_frames`, `add_similarity_callback`)
- Share runtime data through `modules/board/` protocol + mixin pairs; each app composes its own `RenderBoard` from the mixins it needs, and keeps a slice only it reads in `apps/<app>/board/`
- A value derived from pose data, a threshold on it included (a gate, a dead zone), is the pipeline's and so are its settings; the show owns only what it does with it: timings, looks, sound
- Fix a problem where it is made, not where it shows

## Concurrency

- Use threads with explicit synchronization, not async/await
- Use `Lock` and `Event` for shared mutable state and lifecycle signals
- Keep shared critical sections small
- A callback stores or enqueues what it receives and returns
- A receiver does its work on its own thread
- A callback list has its own lock; register under it, copy the list under it, call the callbacks outside it
- Avoid heavy pure-Python loops in hot paths; they hold the GIL, while numpy, ONNX Runtime, OpenGL and depthai release it

## Time

- Read `time.perf_counter()` for every stamp and duration; `time.time()` and `monotonic()` step 15.6 ms on Windows, coarser than a frame
- Stamp a frame once, where the backend hands it over, with its moment of capture where the device reports one, else its arrival; downstream, take time from the frame or the tick instead of reading a clock
- Use the wall clock only for file names and log lines

## Performance

- Treat latency as a first-order concern
- Choose a data path by its latency measured from capture to arrival; a held frame rate doesn't show it
- Prefer vectorized numpy operations and avoid unnecessary copies

## Types

- Use modern typing (`list[T]`, `dict[K, V]`, `X | Y`)
- Give public methods return type hints
- Use a `Protocol` for what a consumer needs from a collaborator, an `ABC` for a family of classes that share code
- Prefer `IntEnum` over string keys for dict lookups and identifiers
- Fix a type error in this order: correct the annotation, narrow with `assert` or `isinstance` where the code already guarantees the type, `cast` to the type the value really has, and suppress only as a last resort
- Suppress only when a third-party stub is provably wrong or the type system cannot express the code, never to hide an error in our own code
- Suppress with `# pyright: ignore[<rule>]` on the one line; never add a bare `# type: ignore`
- Never disable a rule in `[tool.pyright]` to clear errors

## Imports

- Treat each direct subdirectory of `modules/` as an independent package whose `__init__.py` is its public boundary
- Import from a package's `__init__.py`, never from its internal files
- Import the package as a namespace (`from modules import X`, then `X.Y`) when the qualifier adds clarity; import names directly otherwise
- Put all imports from one package on a single `from` line
- `__init__.py` defines no `__all__`
- Inside a package, use relative imports, never the full `modules.X.Y` path

## Error handling

- Define module loggers as `logger = logging.getLogger(__name__)`
- Log exceptions with context; never fail silently
- Degrade gracefully on non-critical runtime failures

## Testing

- Rely on running the real-time system for integration; unit-test infrastructure (settings, frame/features, serialization, utility primitives, geometry)
- Keep geometry in plain modules and drawing in renderers, so geometry is testable without a GL context

## Running code

- Run Python and tests through Bash, never PowerShell
- Start each Bash command with the program itself (no `cd`, env-var prefix or heredoc) so it matches the allow-rules in `.claude/settings.json`
- Change files with Edit/Write, not scripts
- Run a test of the app or a GPU harness only on the user's go, and only what the plan names
- Before each run, say in one line what it measures and how long it takes
- Run a recording only over the moments the test needs, found first from the video file without the app; never a whole loop or a warm-up loop
- Stop a run at twice the time you stated, and say so
- Diagnose a failed or hung run from its log and the code before running it again
- Stop the app in a `finally` in any probe that runs it, so a crash cannot leave it running
- Post each result as it comes

```
python -m unittest discover -s tests -t .                  # guard tests: import layering, glossary
python -m unittest discover -s apps/<app>/tests -t .        # white_space, hd_trio, deep_flow; oak4_test in .venv-oak4
python -m unittest discover -s apps/smile/<pkg>/tests -t .  # tests, comfy, render, player, relay, ...
python -m unittest discover -s modules/<pkg>/tests -t .    # camera/oak4 in .venv-oak4
```

## API evolution

- Prefer clean breaks over deprecation windows
- Update all affected call sites in the same change
- Keep presets and settings schemas in sync with code changes
- Remove obsolete pathways instead of leaving parallel legacy ones

## Plans and commits

- A plan ends with the rules it touches and how it meets each
- A plan is built on the user's literal "go"
- Separate what the code or a measurement shows from what is inferred, and say which is which
- One plan step is one commit; experiments and half steps are not committed
- A finished plan is reported with the rules it touched and how it meets each
- Push only on the user's word
- Never delete or rewrite a plan without confirmation; propose the change first
