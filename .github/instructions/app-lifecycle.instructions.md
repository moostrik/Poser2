---
description: "Use when editing app composition, startup/shutdown flow, or app-level callback wiring."
applyTo: "apps/*/main.py, apps/*/render.py, apps/*/settings.py"
---
# App Lifecycle Guidelines

## App responsibilities

- `main.py` composes the app from reusable modules
- App code wires inter-module callbacks and lifecycle orchestration
- App code should not leak app-specific policy into shared modules

## Startup pattern

- Load and initialize settings early
- Construct core components from settings
- Start workers/services only after wiring is complete

## Shutdown pattern

- Stop external inputs and background workers cleanly
- Release resources in deterministic order
- Prefer explicit stop/teardown paths over implicit garbage-collection cleanup

## Settings boundaries

- `apps/*/settings.py` defines data only (Fields/Groups/Child)
- No runtime side effects in settings definitions
- Keep settings schema synchronized with preset JSON files

## Data flow

- Wire producers to board setters via callbacks; consumers read via protocol-typed references
- `RenderBase` owns the per-frame tick; register render-rate callbacks via `renderer.add_update_callback()`
- Avoid bypass paths that make stage ownership ambiguous
