# white_space's plan

white_space held up to the rules (2026-11-13): its steps are built. What is left is the check: both apps' probes run
once and are compared with today's runs on the code before the plan: white_space (`scripts/white_space_probe.py
--record`, `20250514-130512_4_wide_mono` 18–30 s, run C1follow) and deep_flow, which the recorder's pose hand-over
touches (`scripts/cadence_probe.py`, `measure` 9–19 s, run C1fix).

## Rules it touches

- A callback only hands the item over and returns; the camera's thread runs only the camera.
- Read `time.perf_counter()` for every stamp and duration.
- Inside a package, import relatively; one `from` line per package.
- Name public data-flow methods by execution timing.
- Use the terms in the glossary ("beam light").
- Define module loggers as `logger = logging.getLogger(__name__)`.
