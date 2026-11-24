# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

The new smile goes on location. The Mako, the lights and the sphere are on site; the site PC runs the old smile.
Before it goes, the venvs change: deep_flow, low_flow and smile use depthai 3 (`modules/camera/oak4`; smile also the
Vimba camera, vendored in `modules/camera/vimba`); white_space and hd_trio use depthai 2 (`modules/camera/oak`). The two
cannot share an install (the same package name). Each backend's import error names the venv to start with
(`modules/camera/depthai_backend.py`); the launcher does not switch venvs itself.

## A. The venv change: depthai 3 in `.venv`, depthai 2 the special one

- A1 The code: `.venv` holds depthai 3 (today's `.venv-oak4`), and `.venv-oak` depthai 2, named after its backend
  `modules/camera/oak` and its dependency group `oak`. One commit, made once A2's checks pass (they run in the venvs
  A2 builds):
  - `pyproject.toml`: `default-groups = ["oak4"]`; `[tool.pyright] venv = ".venv"`; the comment over the groups.
  - `uv.lock`: re-locked (`uv lock`); its `[manifest]` holds the default groups, and `uv sync --locked` fails without.
  - `pyright.oak.json`: `"venv": ".venv-oak"`.
  - `scripts/install.bat`: with no argument it builds `.venv` with `oak4`; `install.bat oak` builds `.venv-oak`
    (`--no-group oak4 --group oak`, `uv venv .venv-oak --prompt poser2-oak`); the comments and the closing lines.
  - `scripts/commit_rules.py` `PYRIGHT_RUNS`: `(".venv", [])` and `(".venv-oak", ["-p", "pyright.oak.json"])`.
  - `modules/camera/depthai_backend.py`: `VENVS`, the module docstring and the two messages name `.venv`
    (depthai 3) and `.venv-oak` (depthai 2, `install.bat oak`).
  - Start scripts: `scripts/start/DF.bat`, `scripts/timing.bat` (and its comment) and smile's `startSMILE.bat`,
    `install.bat` (`VENV_DIR` and its header comment), `clip_info.bat`, `make_test_workflow.bat`, `start_relay.bat`
    to `.venv`; `scripts/start/HD.bat` and `WS.bat` to `.venv-oak`; `scripts/makemodels.bat` builds white_space's and
    hd_trio's engines with `.venv-oak` (TensorRT is in both; the engines do not depend on depthai).
  - Docs and docstrings that name `.venv-oak4` (31 files today): README.md, `apps/smile/README.md`, `SMILE.md`,
    `OAK4.md`, `LOW_FLOW_PLAN.md`, deep_flow's and low_flow's `main.py`, the oak4 and oak_device tests' docstrings,
    `oak4_models.py`, `oak4_probe.py`, `trial_runs.py`; and the `.venv` lines that mean depthai 2: README.md's and the
    docs' (white_space, hd_trio), `modules/camera/oak/__init__.py` and the `oak/tests` docstrings.
  - `.github/workflows/tests.yml`: both jobs build a plain `.venv`; the depthai 2 job exports
    `--no-group oak4 --group oak`, the depthai 3 job the default groups; smile's test folders move to the depthai 3
    job; the job names and the comment over them.
  - The rule lines, for Matthias to change at the desktop (a PC session does not edit them):
    - `CLAUDE.md`, the test commands: `python -m unittest discover -s modules/<pkg>/tests -t .    # camera/oak in .venv-oak`
    - `.claude/rules/camera.md`: `- `modules/camera/oak4/` runs on depthai 3 in `.venv` (`install.bat`) and
      `modules/camera/oak/` on depthai 2 in `.venv-oak` (`install.bat oak`); an app imports one of them, never both`

  Check: no `.venv-oak4` left (`git grep`); `uv lock --check`; in `.venv` the guard tests, deep_flow's, low_flow's and smile's tests and
  `modules/camera/oak4`'s; in `.venv-oak` white_space's and hd_trio's tests and `modules/camera/oak`'s; pyright in both
  (`commit_rules.py`); one start each, after A2 has built the venvs: deep_flow, low_flow and smile (`-sim`) in `.venv`,
  white_space (`-sim`) and hd_trio in `.venv-oak`.

- A2 The venvs on this PC: both built new in this checkout, the old ones kept until the new ones pass. Only `.venv` and
  `.venv-oak4` move; every other file (recordings, outputs, ComfyUI, tokens, settings state) stays.
  1. No app runs (`tasklist`); ComfyUI's own venv (`apps/smile/ComfyUI/.venv`) stays as it is.
  2. Today's `.venv` (depthai 2) and `.venv-oak4` (depthai 3) are moved aside to `.venv.old` and `.venv-oak4.old`
     (a moved venv no longer runs: its scripts hold their own path), not deleted.
  3. `scripts\install.bat` (LFS pull, `.venv` with depthai 3 from `uv.lock`), then `scripts\install.bat oak`
     (`.venv-oak`, depthai 2).
  4. The models in `data/` and each app's own `data/models`: in git LFS, pulled by the install; `git lfs ls-files`
     against the folders shows none missing.
  5. smile: `apps\smile\scripts\install.bat` checks ffmpeg, ComfyUI and its models and runs the smoke test (the
     warm-up in ComfyUI). The Vimba SDK (`VIMBA_HOME`) is the PC's, not the venv's: smile's log at start says whether
     VimbaC was found.
  6. Each app's tests in its venv, the guard tests and pyright (`commit_rules.py`).
  7. One start each: smile `-sim` (one visitor through), deep_flow, low_flow, white_space `-sim`, hd_trio; the
     TensorRT engines that exist load (the venvs hold the TensorRT `uv.lock` pins).

  Not checked here: the Mako (it is on site; this PC has no Vimba camera), hd_trio's and white_space's cameras unless
  connected. The `.old` venvs go on Matthias's word.

## B. The clean install

- B1 smile installed from nothing beside the working checkout, as the site makes it, in `C:\Developer\SmileClean`,
  after A2; the checkout and its venvs are not touched:
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

## C. Before leaving

- C1 A committed state that passes smile's tests, the guard tests and pyright; no other thread's uncommitted work in
  it.
- C2 The site's `Smile.json` brought here, committed as it is, then migrated to this version's settings:
  `online.active` to `enabled`, `render.trail_active` to `trail_enabled`, the grin and saver groups under `sequence`;
  a `-sim` start on it, which loads it without a missing-setting warning.

## D. Install on site

- D1 No app runs. The old install's `.venv` moved aside to `.venv.old`, kept until E1 passes; `git pull`;
  `scripts\install.bat` (`.venv`, depthai 3); `apps\smile\scripts\install.bat`, mirrored
  from the old install's `ComfyUI\models`: ffmpeg, ComfyUI, its models and the smoke test pass.
- D2 The face detector's TensorRT engines built for the site's GPU: `scripts\makemodels.bat smile`; the log says
  `Face detector: … ready (TRT)`.

## E. Checks on site

- E1 The README's ON SITE CHECKS, each one; a check that passes gets its date there.

## Open questions

## The rules it touches

- `.claude/rules/camera.md`, which backend runs in which venv: A1 (Matthias edits it)
- `CLAUDE.md` Running code, the test commands per venv: A1 (Matthias edits it)
- `CLAUDE.md` API evolution, every call site in the same change: A1
- `CLAUDE.md` Running code, a run only on the user's go: A1, A2, B1, C2
- `CLAUDE.md` API evolution, settings files in sync with the code: C2
- `.claude/rules/smile.md`, README.md current (setup, on-site checks): A1, B1, E1
- `CLAUDE.md` Plans and commits, a step checked by a run is one commit: A1 (committed after A2's checks)
