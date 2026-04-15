# Poser2 — Coding Guidelines

## Project shape

Poser2 is a real-time, low-latency system with app-specific orchestration in `apps/` and reusable infrastructure in `modules/`.

Architecture rule:
- `apps/` can import `modules/`
- `modules/` should not depend on app code
- Prefer module independence; keep cross-module coupling minimal
- `modules/settings/` is shared infrastructure by design

## Follow PEP 8 (from now on)

All new code and touched code should follow PEP 8 style conventions.

Practical expectations:
- Use snake_case for functions, variables, and module names
- Use PascalCase for class names
- Keep line length reasonable (target 88, hard max 100 when needed for readability)
- Keep imports grouped as: standard library, third-party, local
- Prefer explicit, readable code over compact clever code

## Settings are pure data

`BaseSettings` subclasses are pure data containers with no runtime side effects.

Rules:
- Settings classes declare `Field`, `Group`, and `Child` descriptors only
- Runtime behavior depending on settings belongs in app `main.py` or the consuming module
- Detailed preset/schema rules live in `.github/instructions/settings-presets.instructions.md`

Use `BaseSettings` for reactive configuration, and `@dataclass` for simple value objects.

## Frame contract: all data is always there

Frame ECS follows an always-present contract: `frame[FeatureType]` never raises, and missing data is represented as NaN with score `0.0`.

Rules:
- Consumers read data directly and handle NaN semantics
- Detailed feature and stage rules live in `.github/instructions/frame-features.instructions.md`

## Composition and wiring

Prefer composition over inheritance for runtime assembly.

Rules:
- Pass dependencies through constructors
- Use callback pipelines for data flow between components
- Inter-module callback wiring belongs in app `start()`
- Module-internal callback wiring is fine within the module itself

## Concurrency model

The runtime uses threads and explicit synchronization, not async/await.

Rules:
- Use `Lock` and `Event` for shared mutable state and lifecycle signals
- Keep shared critical sections small
- Treat callback registration and dispatch as thread-sensitive
- All OpenGL calls must run on the render thread

GIL note:
- Native extensions (numpy, ONNX Runtime, OpenGL bindings, depthai, etc.) can release the GIL
- Pure Python loops do not; avoid heavy Python loops in hot paths

## Low-latency performance rules

Latency is a first-order concern.

Rules:
- Avoid Python-level loops over numeric arrays in hot paths
- Prefer vectorized numpy operations
- Pre-allocate large buffers when it improves stability and latency
- Avoid unnecessary copies
- For pose feature arrays: construction takes ownership and may set arrays read-only

## Types and interfaces

Use modern Python typing consistently (`list[T]`, `dict[K, V]`, `X | Y`).

Rules:
- Public methods should include return type hints
- Use `Protocol` for structural typing where multiple implementations share behavior
- Use `ABC` when enforcing inheritance-based contracts

## Error handling and logging

Rules:
- Define module loggers as `logger = logging.getLogger(__name__)`
- Wrap callback fan-out so one failing callback does not break the pipeline
- Log exceptions with context; avoid silent failures
- Prefer graceful degradation for non-critical runtime failures

## Testing approach

Poser2 is tested heavily by running the real-time system.

Guidelines:
- Prioritize unit tests for infrastructure modules (settings, frame/features, serialization, utility primitives)
- Keep runtime validation (live execution) for real-time integration behavior
- Add tests where they provide long-term leverage, not ceremony

## API evolution

Prefer clean breaks over long deprecation windows.

Rules:
- Update all affected call sites in one change
- Keep presets/settings schemas in sync with code changes
- Remove obsolete pathways instead of leaving parallel legacy patterns
