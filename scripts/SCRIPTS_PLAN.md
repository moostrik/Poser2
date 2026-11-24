# The scripts' plan

What is next for the repository's scripts: the install (`install.bat`), the start scripts and the environments they
make. What spans the apps otherwise is in the [refactor's plan](../REFACTOR_PLAN.md).

The apps and their depthai: deep_flow, low_flow and smile use depthai 3 (`modules/camera/oak4`; smile also the Vimba
camera, vendored in `modules/camera/vimba`); white_space and hd_trio use depthai 2 (`modules/camera/oak`). The two
cannot share an install (the same package name). Each backend's import error names the venv to start with
(`modules/camera/depthai_backend.py`); the launcher does not switch venvs itself.

## A. depthai 3 in `.venv`, depthai 2 the special one

- A1 `.venv` holds depthai 3 (today's `.venv-oak4`), and `.venv-oak` depthai 2, named after its backend
  `modules/camera/oak` and its dependency group `oak`. In one commit:
  - `pyproject.toml`: `default-groups = ["oak4"]`; `[tool.pyright] venv = ".venv"`; the comment over the groups.
  - `pyright.oak.json`: `"venv": ".venv-oak"`.
  - `scripts/install.bat`: with no argument it builds `.venv` with `oak4`; `install.bat oak` builds `.venv-oak`
    (`--no-group oak4 --group oak`); the comments and the closing lines.
  - `scripts/commit_rules.py` `PYRIGHT_RUNS`: `(".venv", [])` and `(".venv-oak", ["-p", "pyright.oak.json"])`.
  - `modules/camera/depthai_backend.py`: the two messages name `.venv` (depthai 3) and `.venv-oak` (depthai 2,
    `install.bat oak`).
  - Start scripts: `scripts/start/DF.bat`, `scripts/timing.bat` and smile's `startSMILE.bat`, `install.bat`
    (`VENV_DIR`), `clip_info.bat`, `make_test_workflow.bat`, `start_relay.bat` to `.venv`; `scripts/start/HD.bat`
    and `WS.bat` to `.venv-oak`; `scripts/makemodels.bat` builds white_space's and hd_trio's engines with `.venv-oak`
    (TensorRT is in both; the engines do not depend on depthai).
  - Docs and docstrings that name `.venv-oak4` (31 files today): README.md, `apps/smile/README.md`, `SMILE.md`,
    `OAK4.md`, `LOW_FLOW_PLAN.md`, deep_flow's and low_flow's `main.py`, the oak4 and oak_device tests' docstrings,
    `oak4_models.py`, `oak4_probe.py`, `trial_runs.py`; and README.md's and the docs' `.venv` lines that mean
    depthai 2 (white_space, hd_trio).
  - `.github/workflows/tests.yml`: the depthai 3 job uses `.venv`, the depthai 2 job `.venv-oak`.
  - The rule lines, for Matthias to change at the desktop (a PC session does not edit them):
    - `CLAUDE.md`, the test commands: `python -m unittest discover -s modules/<pkg>/tests -t .    # camera/oak in .venv-oak`
    - `.claude/rules/camera.md`: `- `modules/camera/oak4/` runs on depthai 3 in `.venv` (`install.bat`) and
      `modules/camera/oak/` on depthai 2 in `.venv-oak` (`install.bat oak`); an app imports one of them, never both`

  Check: no `.venv-oak4` left (`git grep`); in `.venv` the guard tests, deep_flow's, low_flow's and smile's tests and
  `modules/camera/oak4`'s; in `.venv-oak` white_space's and hd_trio's tests and `modules/camera/oak`'s; pyright in both
  (`commit_rules.py`); one start each, after B1 has built the venvs: deep_flow, low_flow and smile (`-sim`) in `.venv`,
  white_space (`-sim`) and hd_trio in `.venv-oak`.

## B. New venvs on this PC

- B1 Both venvs built new in this checkout, the old ones kept until the new ones pass. Only `.venv` and `.venv-oak4`
  move; every other file (recordings, outputs, ComfyUI, tokens, settings state) stays. smile's clean install in its
  own folder is `apps/smile/SMILE_PLAN.md` H1.
  1. No app runs (`tasklist`); ComfyUI's own venv (`apps/smile/ComfyUI/.venv`) stays as it is.
  2. Today's `.venv` (depthai 2) and `.venv-oak4` (depthai 3) are moved aside to `.venv.old` and `.venv-oak4.old`
     (a moved venv no longer runs: its scripts hold their own path), not deleted.
  3. `scripts\install.bat` (LFS pull, `.venv` with depthai 3 from `uv.lock`), then `scripts\install.bat oak`
     (`.venv-oak`, depthai 2).
  4. The models in `data/` and each app's own `data/models`: in git LFS, pulled by the install; `git lfs ls-files`
     against the folders shows none missing.
  5. TensorRT engines: `scripts\makemodels.bat all` with `rebuild`, since the venvs are new.
  6. smile: `apps\smile\scripts\install.bat` checks ffmpeg, ComfyUI and its models and runs the smoke test (the
     warm-up in ComfyUI). The Vimba SDK (`VIMBA_HOME`) is the PC's, not the venv's: smile's log at start says whether
     VimbaC was found.
  7. Each app's tests in its venv, the guard tests and pyright (`commit_rules.py`).
  8. One start each: smile `-sim` (one visitor through), deep_flow, low_flow, white_space `-sim`, hd_trio.

  Not checked here: the Mako (it is on site; this PC has no Vimba camera), hd_trio's and white_space's cameras unless
  connected. The `.old` venvs go on Matthias's word.

## Open questions

## The rules it touches

- `.claude/rules/camera.md`, which backend runs in which venv: A1 (Matthias edits it)
- `CLAUDE.md` Running code, the test commands per venv: A1 (Matthias edits it)
- `CLAUDE.md` API evolution, every call site in the same change: A1
- `CLAUDE.md` Running code, a run only on the user's go: A1, B1
