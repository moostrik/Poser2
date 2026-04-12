# Poser2 — Coding Guidelines

## Project shape

Poser2 is a real-time, low-latency system with app-specific orchestration in `apps/` and reusable infrastructure in `modules/`.

- `apps/` can import `modules/`; `modules/` must not depend on app code
- Prefer module independence; keep cross-module coupling minimal
- `modules/settings/` is shared infrastructure by design

## Follow PEP 8

All new and touched code should follow PEP 8.
- Group imports as: standard library, third-party, local

## Settings are pure data

`BaseSettings` subclasses are pure data containers with no runtime side effects.
Use `BaseSettings` for reactive configuration (callbacks on change, serialization/preset support) and `@dataclass` for plain value objects.

- Detailed backend rules: `.github/instructions/settings-backend.instructions.md`
- Schema/preset workflow: `.github/instructions/settings-presets.instructions.md`

## Pose frame contract

`frame[FeatureType]` never raises; missing data is NaN with score `0.0`.

- Detailed rules: `.github/instructions/pose-data.instructions.md`
- Processing/stage rules: `.github/instructions/pose-processing.instructions.md`

## Composition and wiring

Prefer composition over inheritance for runtime assembly.

- Pass dependencies through constructors
- Use callback pipelines for data flow between components
- Use lambda-deferred creation where startup ordering or runtime settings values require it

## Concurrency model

The runtime uses threads and explicit synchronization, not async/await.

- Use `Lock` and `Event` for shared mutable state and lifecycle signals
- Keep shared critical sections small
- Treat callback registration and dispatch as thread-sensitive

GIL note: native extensions (numpy, ONNX Runtime, OpenGL, depthai) release the GIL; pure Python loops do not — avoid heavy Python loops in hot paths.

## Low-latency performance

Latency is a first-order concern.
- Prefer vectorized numpy operations; avoid unnecessary copies
- Detailed array/buffer rules in `.github/instructions/pose-data.instructions.md`

## Types and interfaces

Use modern Python typing (`list[T]`, `dict[K, V]`, `X | Y`).
- Public methods should include return type hints
- Use `Protocol` or `ABC` as appropriate; be consistent

## Error handling and logging

- Define module loggers as `logger = logging.getLogger(__name__)`
- Log exceptions with context; avoid silent failures
- Prefer graceful degradation for non-critical runtime failures

## Testing approach

Poser2 is tested heavily by running the real-time system.

- Prioritize unit tests for infrastructure modules (settings, frame/features, serialization, utility primitives)
- Keep runtime validation (live execution) for real-time integration behavior
- Add tests where they provide long-term leverage, not ceremony

## API evolution

Prefer clean breaks over long deprecation windows — single-developer codebase where parallel legacy paths cause more bugs than they prevent.

- Update all affected call sites in one change
- Keep presets/settings schemas in sync with code changes
- Remove obsolete pathways instead of leaving parallel legacy patterns
