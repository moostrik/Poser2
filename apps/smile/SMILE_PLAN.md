# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

The new smile goes on location. The Mako, the lights and the sphere are on site; the site PC runs the old smile.
smile runs in `.venv` (depthai 3, `scripts\install.bat`), as deep_flow and low_flow do; white_space and hd_trio run in
`.venv-oakd` (depthai 2, `scripts\install.bat oakd`).

## A. The clean install

- A1 smile installed from nothing beside the working checkout, as the site makes it, in `C:\Developer\SmileClean`;
  the checkout and its venvs are not touched:
  1. `git clone` from GitHub of the branch the site pulls (pushed on Matthias's word), `git lfs pull`.
  2. `scripts\install.bat` (`.venv`, depthai 3).
  3. `apps\smile\scripts\install.bat --mirror C:\Developer\Poser2\apps\smile\ComfyUI\models`: ffmpeg, ComfyUI, its
     models and the smoke test. The site mirrors from its old install the same way.
  4. `scripts\makemodels.bat smile`: the face detector's engines, which git does not hold, built from the clone's own
     ONNX.
  5. By hand, not in git by design: the online token, `Smile.json`, one recording for `-sim`; ffmpeg and the Vimba
     SDK are the PC's.
  6. smile's tests in the clean `.venv`, one `-sim` start with a visitor through.

  Anything missing beyond 5 is a defect, fixed in the install or `apps/smile/README.md`; the list in 5 goes into that
  README's setup.

## B. Before leaving

- B1 A committed state that passes smile's tests, the guard tests and pyright; no other thread's uncommitted work in
  it.
- B2 A live smile start on the OAK4 (`-oak`) in `.venv`, ComfyUI running, Matthias in front: one visitor through,
  from the flash to the saved grin and portrait.
- B3 The rule lines, for Matthias to change at the desktop (a PC session does not edit them):
  - `CLAUDE.md`, the test commands: `python -m unittest discover -s modules/<pkg>/tests -t .    # camera/oak in .venv-oakd`
  - `.claude/rules/camera.md`: `- `modules/camera/oak4/` runs on depthai 3 in `.venv` (`install.bat`) and
    `modules/camera/oak/` on depthai 2 in `.venv-oakd` (`install.bat oakd`); an app imports one of them, never both`

## C. Install on site

- C1 No app runs. The old install's `.venv` moved aside to `.venv.old`, kept until D1 passes; `git pull`;
  `scripts\install.bat` (`.venv`, depthai 3); `apps\smile\scripts\install.bat`, mirrored
  from the old install's `ComfyUI\models`: ffmpeg, ComfyUI, its models and the smoke test pass.
- C2 The face detector's TensorRT engines built for the site's GPU: `scripts\makemodels.bat smile`; the log says
  `Face detector: … ready (TRT)`.

## D. Checks on site

- D1 The README's ON SITE CHECKS, each one; a check that passes gets its date there.

## Open questions

## The rules it touches

- `.claude/rules/camera.md`, which backend runs in which venv: B3 (Matthias edits it)
- `CLAUDE.md` Running code, the test commands per venv: B3 (Matthias edits it)
- `CLAUDE.md` Running code, a run only on the user's go: A1, B2
- `.claude/rules/smile.md`, README.md current (setup, on-site checks): A1, D1
