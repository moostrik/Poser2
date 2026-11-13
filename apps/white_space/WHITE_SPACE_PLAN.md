# white_space's plan

white_space held up to the rules (2026-11-13): what does not meet them, one step each, in order. Each step is one
commit, checked by the tests named. After the last step both apps' probes run once and are compared with today's
runs on the same code before the plan: white_space (`scripts/white_space_probe.py --record`,
`20250514-130512_4_wide_mono` 18–30 s, run C1follow) and deep_flow, which step 2 touches
(`scripts/cadence_probe.py`, `measure` 9–19 s, run C1fix).

## 3. The motor and the light sender on `perf_counter()`

`light/motor.py` reads `monotonic()` for the fall edge, the fall timing and the simulated motor's tick;
`inout/osc_light_sender.py` for its retry and hold deadlines. `monotonic()` is GetTickCount64 here, 15.6 ms steps,
half a light tick at 32 fps, against 0.1 µs for `perf_counter()`; a call costs 17.6 ns against 37.2 ns (measured
2026-11-13, `.venv`). Both read `perf_counter()`.
Checked by: `apps/white_space/tests/test_playhead_motor.py`, `test_osc_light_protocol.py`.

## 4. The tracker journal on `perf_counter()`

The journal stamps its entries with `time.time()`, which cannot be lined up with the tracklets' times. Its entries
take `perf_counter()`; the meta line keeps the wall clock once, to find a capture by the time of day.
Checked by: `apps/white_space/tests/test_tracker_journal.py`.

## 5. Imports

`render/` imports its own app absolutely (`from apps.white_space...`, `render.py` and four layer files), some from
internal files; they import relatively. `main.py`'s two `from modules.pose` lines become one. `main.py`'s import of
`.pose.pipeline` stays the stated exception it is: through the package it is circular (the package reads the
settings, the settings import the package).
Checked by: the guard tests and white_space's tests.

## 6. Names and logging

`notify_fall` (`Conductor`, `MotorController`) stores the fall's moment: `set_fall`, its OSC binding with it.
"Lamp" in panel text and docstrings (`light/frame.py`, `light/layers/beam/blue_sound.py`, `light/playhead.py`,
`light/settings.py`, `inout/osc_light_sender.py`, `render/layers/beam_light_projection.py`) is "beam light".
`main.py`'s `logging.info` goes through its `logger`.
Checked by: the guard tests and white_space's tests.

## Rules it touches

- Read `time.perf_counter()` for every stamp and duration: 3, 4.
- Inside a package, import relatively; one `from` line per package: 5.
- Name public data-flow methods by execution timing: 6.
- Use the terms in the glossary ("beam light"): 6.
- Define module loggers as `logger = logging.getLogger(__name__)`: 6.
