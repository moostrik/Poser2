# white_space's plan

white_space held up to the rules (2026-11-13): what does not meet them, one step each, in order. Each step is one
commit, checked by the tests named. After the last step both apps' probes run once and are compared with today's
runs on the same code before the plan: white_space (`scripts/white_space_probe.py --record`,
`20250514-130512_4_wide_mono` 18–30 s, run C1follow) and deep_flow, which step 2 touches
(`scripts/cadence_probe.py`, `measure` 9–19 s, run C1fix).

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

- Inside a package, import relatively; one `from` line per package: 5.
- Name public data-flow methods by execution timing: 6.
- Use the terms in the glossary ("beam light"): 6.
- Define module loggers as `logger = logging.getLogger(__name__)`: 6.
