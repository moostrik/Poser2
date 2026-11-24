# Utils plan

Proposed by the utils rescan of 2026-11-24 (deep-flow at 38929e26); not built. Each step names its finding,
**(seen)** in the code or a run, **(deduction)** when inferred. A step checked by a test or pyright is one commit.

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
