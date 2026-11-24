# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

The new smile goes on location. The Mako, the lights and the sphere are on site; the site PC runs the old smile.
smile runs in `.venv` (depthai 3, `scripts\install.bat`), as deep_flow and low_flow do; white_space and hd_trio run in
`.venv-oakd` (depthai 2, `scripts\install.bat oakd`).

## A. Before leaving

- A1 A committed state that passes smile's tests, the guard tests and pyright; no other thread's uncommitted work in
  it.
- A2 A live smile start on the OAK4 (`-oak`) in `.venv`, ComfyUI running, Matthias in front: one visitor through,
  from the flash to the saved grin and portrait.
- A3 The rule lines, for Matthias to change at the desktop (a PC session does not edit them):
  - `CLAUDE.md`, the test commands: `python -m unittest discover -s modules/<pkg>/tests -t .    # camera/oak in .venv-oakd`
  - `.claude/rules/camera.md`: `- `modules/camera/oak4/` runs on depthai 3 in `.venv` (`install.bat`) and
    `modules/camera/oak/` on depthai 2 in `.venv-oakd` (`install.bat oakd`); an app imports one of them, never both`
- A4 What the site needs that git does not bring, in one folder on the scratch disk, `F:\_scratch\SMILE`:
  `models\`, a copy of `apps\smile\ComfyUI\models` (the 28 models, 48.5 GB); `online_token.txt` from
  `apps\smile\data`; `input\`, the photos for the panel's still image from `apps\smile\data\input`. The copy runs on
  Matthias's go.

## B. Install on site

- B1 No app runs. The old install's `.venv` moved aside to `.venv.old`, kept until C1 passes; `git pull`;
  `scripts\install.bat` (`.venv`, depthai 3); `apps\smile\scripts\install.bat --mirror F:\_scratch\SMILE\models`
  (the disk from A4): ffmpeg, ComfyUI, its models and the smoke test pass; `online_token.txt` and `input\` from the
  disk into `apps\smile\data`.
- B2 The face detector's TensorRT engines built for the site's GPU: `scripts\makemodels.bat smile`; the log says
  `Face detector: … ready (TRT)`.

## C. Checks on site

- C1 The README's ON SITE CHECKS, each one; a check that passes gets its date there.

## Open questions

## The rules it touches

- `.claude/rules/camera.md`, which backend runs in which venv: A3 (Matthias edits it)
- `CLAUDE.md` Running code, the test commands per venv: A3 (Matthias edits it)
- `CLAUDE.md` Running code, a run only on the user's go: A2, A4
- `.claude/rules/smile.md`, README.md current (on-site checks): C1
