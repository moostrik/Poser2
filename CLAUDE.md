# Poser2 — Coding Guidelines

Poser2 is a real-time, low-latency system: app-specific orchestration in `apps/`, reusable infrastructure in `modules/`.

## Writing rules in this file

- One rule per bullet, phrased as an instruction
- Add a short reason only when it marks the rule's boundary or the rule isn't obvious
- No incident history, no examples unless the rule is ambiguous without one
- Put a rule where it loads when the code it governs is written: here, or a `.claude/rules/` file whose `paths` match that code

## Project shape

- Use Python 3.12 and follow PEP 8
- `apps/` may import `modules/`; `modules/` must never import app code
- Keep modules independent and cross-module coupling minimal
- `modules/settings/` is shared infrastructure; any module may depend on it
- Never change the White Space fixture firmware (`apps/white_space/data/firmware/`); solve everything on the app side of the wire

## Settings

- `BaseSettings` subclasses are pure data containers with no runtime side effects
- Use `BaseSettings` for reactive configuration and `@dataclass` for plain value objects
- Components take their settings object (or a `Group`/`Child` of it) in the constructor, never as a global, and read fields when they use them
- Never unpack setting values into constructor parameters; a copied value goes stale when the panel changes it
- Cache only `Field.INIT` values, since they cannot change after `initialize()`
- Use `bind()` only to react to configuration changes, never to relay runtime data between components
- Keep bound callbacks thread-safe, and never write the field that triggered the callback
- Components that `bind()` must `unbind()` the same callbacks on teardown
- Treat `Field.READ` values as snapshots; copy arrays before handing them to another thread
- Keep `Field` descriptions to one short line of panel text: what the value is and its unit

## Pose Frame contract

- `frame[FeatureType]` never raises; missing data is NaN with score `0.0`

## Composition and wiring

- Prefer composition over inheritance for runtime assembly
- Pass collaborators and fixed structure (counts known at startup) through constructors; tunable values come from settings
- Move data between components with callback pipelines
- Name public data-flow methods by execution timing: `set` stores for polling, `submit` enqueues deferred work, `process` transforms and emits synchronously, `update` advances on a tick
- Name output channels verb + domain noun (`submit_frames`, `add_similarity_callback`)
- Share runtime data through `modules/board/` protocol + mixin pairs; each app composes its own `RenderBoard` from the mixins it needs

## Concurrency

- Use threads with explicit synchronization, not async/await
- Use `Lock` and `Event` for shared mutable state and lifecycle signals
- Keep shared critical sections small
- Treat callback registration and dispatch as thread-sensitive
- Avoid heavy pure-Python loops in hot paths; they hold the GIL, while numpy, ONNX Runtime, OpenGL and depthai release it

## Performance

- Treat latency as a first-order concern
- Prefer vectorized numpy operations and avoid unnecessary copies

## Types

- Use modern typing (`list[T]`, `dict[K, V]`, `X | Y`)
- Give public methods return type hints
- Use `Protocol` or `ABC` as appropriate, and be consistent
- Prefer `IntEnum` over string keys for dict lookups and identifiers

## Imports

- Treat each direct subdirectory of `modules/` as an independent package whose `__init__.py` is its public boundary
- Import from a package's `__init__.py`, never from its internal files
- Import the package as a namespace (`from modules import X`, then `X.Y`) when the qualifier adds clarity; import names directly otherwise
- Put all imports from one package on a single `from` line
- Export only the package's own public symbols from `__init__.py`, and don't define `__all__`
- Inside a package, use relative imports, never the full `modules.X.Y` path

## Error handling

- Define module loggers as `logger = logging.getLogger(__name__)`
- Log exceptions with context; never fail silently
- Degrade gracefully on non-critical runtime failures

## Testing

- Rely on running the real-time system for integration; unit-test infrastructure (settings, frame/features, serialization, utility primitives, geometry)
- Add tests where they give long-term leverage, not ceremony
- Keep geometry in plain modules and drawing in renderers, so geometry is testable without a GL context

## Running code

- Run Python and tests through Bash, never PowerShell; PowerShell 5.1 reports unittest's stderr output as a failure
- Start each Bash command with the program itself (no `cd`, env-var prefix or heredoc) so it matches the allow-rules in `.claude/settings.json`
- Change files with Edit/Write, not scripts

```
python -m unittest discover -s apps/white_space/tests -t .
python -m unittest discover -s modules/<pkg>/tests -t .    # oak, pose, render, settings, tracker
```

## API evolution

- Prefer clean breaks over deprecation windows
- Update all affected call sites in the same change
- Keep presets and settings schemas in sync with code changes
- Remove obsolete pathways instead of leaving parallel legacy ones

## Working style

- Separate what the code or a measurement shows from what is inferred, and say which is which
- Do the plan step asked for and stop; don't start the next one
- Don't offer to revert work
- Never delete or rewrite a plan without confirmation; propose the change first
