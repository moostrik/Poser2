# smile's plan

smile (`apps/smile`, not the modules it uses) held up to the rules (2026-11-13): its steps are built. What is left is
the check: smile runs once on the image input (`use_image`, 20 s, as before the plan: 25 camera frames a second, a
face in every one; with `credit_scroll` on, to see the name scroll on the walls) and, with ComfyUI started
(`scripts/start_comfy.bat`), through one whole visitor. Its command-line tools (`comfy/models.py`, `comfy/smoke.py`,
`online/fake.py`: their folder, logger names and `print`) need no change now (settled in the rules thread, 06fd4666).

## Rules it touches

- The camera's thread runs only the camera; a callback only hands the item over and returns.
- A worker thread's loop catches what its work raises, logs it and goes on (smile).
- Read `time.perf_counter()` for every stamp and duration.
