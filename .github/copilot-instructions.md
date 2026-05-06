# Poser2 — Coding Guidelines

## Project shape

Poser2 is a real-time, low-latency system with app-specific orchestration in `apps/` and reusable infrastructure in `modules/`.

- Using Python 3.12
- Follow PEP 8.
- `apps/` can import `modules/`; `modules/` must not depend on app code
- Prefer module independence; keep cross-module coupling minimal
- `modules/settings/` is shared infrastructure by design

## Settings are pure data

- `BaseSettings` subclasses are pure data containers with no runtime side effects.
- Use `BaseSettings` for reactive configuration, `@dataclass` for plain value objects.
- Detailed rules: `.github/instructions/settings-presets.instructions.md`

## Pose Frame contract

- `frame[FeatureType]` never raises; missing data is NaN with score `0.0`.
- Detailed rules: `.github/instructions/pose-data.instructions.md`

## Composition and wiring

- Prefer composition over inheritance for runtime assembly.
- Pass dependencies through constructors
- Use callback pipelines for data flow between components
- Public data-flow methods signal execution timing: set (store for polling), submit (enqueue for deferred work), process (synchronous transform-and-emit), update (tick-driven advance/pull)
- Output channels use verb + domain noun (`submit_frames`, `add_similarity_callback`)

## Board

`modules/board/` defines protocol + mixin pairs for shared runtime data. Each app composes its own `RenderBoard` from the mixins it needs.
- Detailed rules: `.github/instructions/board.instructions.md`

## Concurrency

- The runtime uses threads and explicit synchronization, not async/await.
- Use `Lock` and `Event` for shared mutable state and lifecycle signals
- Keep shared critical sections small
- Treat callback registration and dispatch as thread-sensitive
- Native extensions (numpy, ONNX Runtime, OpenGL, depthai) release the GIL; pure Python loops do not — avoid heavy Python loops in hot paths

## Performance

Latency is a first-order concern.
- Prefer vectorized numpy operations; avoid unnecessary copies
- Detailed array/buffer rules: `.github/instructions/pose-data.instructions.md`

## Types

- Use modern Python typing (`list[T]`, `dict[K, V]`, `X | Y`).
- Public methods should include return type hints
- Use `Protocol` or `ABC` as appropriate; be consistent
- Prefer `IntEnum` over string keys for dict lookups and identifiers

## Imports

`modules/` is the namespace root; each direct subdirectory is an independent package with its own `__init__.py` as its public boundary.
- Use whichever import style keeps use sites unambiguous: import the package as a namespace (`from modules import X`; use `X.Y`) when the qualifier adds clarity, or import names directly (`from modules.X import A, B`) when they are unambiguous on their own.
- Consolidate all imports from the same package onto one line; never split a single package across multiple `from x import` statements
- Import from a package's `__init__.py` boundary, not from internal implementation files inside it.
- Keep `__init__.py` exports limited to the package's own public symbols
- Inside a package's own `__init__.py` or sub-modules, always use relative imports (`from .X import`, `from ..X import`). Never use the full `modules.X.Y` path to import from within the same package.
- Do not define `__all__` in `__init__.py`; explicit named re-exports are sufficient.

## Error handling

- Define module loggers as `logger = logging.getLogger(__name__)`
- Log exceptions with context; avoid silent failures
- Prefer graceful degradation for non-critical runtime failures

## Testing

Poser2 is tested heavily by running the real-time system.
- Prioritize unit tests for infrastructure modules (settings, frame/features, serialization, utility primitives)
- Add tests where they provide long-term leverage, not ceremony

## API evolution

- Prefer clean breaks over long deprecation windows.
- Update all affected call sites in one change
- Keep presets/settings schemas in sync with code changes
- Remove obsolete pathways instead of leaving parallel legacy patterns
