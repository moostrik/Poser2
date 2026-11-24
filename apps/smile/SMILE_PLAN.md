# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

The new smile goes on location. The Mako, the lights and the sphere are on site; the site PC runs the old smile.
smile runs in `.venv` (depthai 3, `scripts\install.bat`), as deep_flow and low_flow do; white_space and hd_trio run in
`.venv-oakd` (depthai 2, `scripts\install.bat oakd`). Before it goes, the install does a new PC's setup for every app:
what the READMEs list by hand today and a script can do, the scripts do; what stays by hand is one list.

## A. The install

Each step leaves the install working; each is checked on this PC, in `C:\Developer\Poser2` or the clone in
`C:\Developer\SmileClean`.

- A1 The PC part in the root `scripts\install.bat`, for every app, before the venv: each missing piece asked (Y/N)
  and installed by winget: the Visual C++ Redistributable (`Microsoft.VCRedist.2015+.x64`; present when
  `HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64` has `Installed` 1), uv (`astral-sh.uv`, used by its
  full path in the same run), ffmpeg (`Gyan.FFmpeg`), VS Code (`Microsoft.VisualStudioCode`); then
  `git lfs install`. Checked, not installed: the NVIDIA driver (580 or newer, from `nvidia-smi`; CUDA 13) and the
  iGPU's Quick Sync (a short ffmpeg `h264_qsv` encode), each with its fix named. Git itself comes before the clone.
  Check: on this PC every piece is found and nothing is offered; in a shell without ffmpeg on the PATH it offers
  ffmpeg, answered Y, and winget installs it.
- A2 `install.bat <app>`: the app's venv, then its TensorRT engines (`makemodels.bat <app>`, which skips the ones
  there) and, for deep_flow and low_flow, the OAK4 archives `scripts/oak4_models.py` fetches, when missing.
  `install.bat` and `install.bat oakd` keep building the bare venvs. Which app runs in which venv is written once,
  in `scripts\app_venv.bat <app>` (sets `VENV`: `.venv-oakd` for white_space and hd_trio, `.venv` for the others),
  which `install.bat`, `makemodels.bat` and `timing.bat` call in place of their own lists. `makemodels.bat` pauses
  only when run on its own, not when `install.bat` calls it. Check: `install.bat smile` in the clone builds nothing
  new; with smile's engines removed it builds them again; `install.bat deep_flow` here, with one zoo archive moved
  aside, fetches it back; `timing.bat` still picks each app's venv.
- A3 The OAK4 person detector, `data/models/yolov8n_coco_640x512_rvc4.tar.xz` (compiled by hand through HubAI), in
  git LFS: `.gitattributes` gets `data/models/*.tar.xz` as LFS, `.gitignore` lets that one archive through; the
  zoo's three stay ignored and fetched (A2). Check: `git lfs ls-files` lists it; the clone gets it with
  `git lfs pull`.
- A4 `install.bat <app> --startup`: a shortcut to the app's start script (`scripts\start\<app>.bat`, smile's
  `apps\smile\scripts\startSMILE.bat`) in the user's Startup folder, replaced when it is there. Check: the shortcut
  appears in `shell:startup` and opens the right script; a second run leaves one shortcut.
- A5 smile's ComfyUI venv on uv's Python 3.12 (`uv venv --seed --python 3.12` in `install_comfy.ps1`, so its pip
  steps run as now) in place of the py launcher; a missing Python 3.12 is uv's to fetch. The model tool's docstring
  (`py -3.12 -m apps.smile.comfy.models`) names `.venv\Scripts\python`. Check: in the clone, ComfyUI's venv deleted
  and built again this way, the smoke test passes.
- A6 What the site needs that git does not bring, in one folder on the scratch disk, `F:\_scratch\SMILE`:
  `models\`, a copy of `apps\smile\ComfyUI\models` (the 28 models, 48.5 GB); `online_token.txt` from
  `apps\smile\data`; `input\`, the photos for the panel's still image from `apps\smile\data\input`. The copy runs on
  Matthias's go.
- A7 smile's `install.bat --from DIR` takes the folder of A6 in place of its `--mirror`: it hands `DIR\models` to the
  model tool as its `--mirror` (the tool keeps that option: verified by hash, the rest from the sources) and copies
  `DIR\online_token.txt` and `DIR\input\` into `apps\smile\data` where they are missing, never over a file that is
  there; without `--from` it asks for the folder. Check: in the clone with the token and `data\input` removed,
  `install.bat --from F:\_scratch\SMILE` puts them back and passes the smoke test; a second run copies nothing.
- A8 The READMEs: one "New PC" section at the top of the root README: Git and the clone, `scripts\install.bat
  <app> [--startup]`, then what stays by hand, per app, with its link: the NVIDIA driver and the Intel graphics
  driver; for smile the Vimba 6 SDK, the Mako's UserSet2, Max and the Windows sound, the PC section (adapter,
  jumbo frames, power plan, update policy, BIOS); Max for the other apps; `oakctl device update` for the OAK4
  apps; `docs/WINDOWS.md` for a show machine. VS Code's extensions, PowerShell 7 and the FP16 conversion stay a
  development note. A manual step the install now does is struck through (`~~ ~~`) where it stands, until D1 passes
  on site, then deleted: in smile's README its Visual C++, Python, Git (but the clone), ExecutionPolicy and ffmpeg
  steps and the separate `makemodels.bat smile` line; in the root README its Visual C++, Visual Studio, uv,
  `git lfs install`, ExecutionPolicy and ffmpeg steps; each struck once; smile's README keeps its room setup.
  Check: every step the install does is struck through in both READMEs, and every step that stays by hand is in the
  "New PC" list once.

## B. Before leaving

- B1 The rehearsal: a new clone of the pushed branch (pushed on Matthias's word) in place of
  `C:\Developer\SmileClean`, installed only as the README's "New PC" says: `scripts\install.bat smile`, smile's
  `install.bat --from F:\_scratch\SMILE`; smile's tests, one `-sim` start with ComfyUI and a visitor through.
- B2 A committed state that passes smile's tests, the guard tests and pyright; no other thread's uncommitted work in
  it.
- B3 A live smile start on the OAK4 (`-oak`) in `.venv`, ComfyUI running, Matthias in front: one visitor through,
  from the flash to the saved grin and portrait.
- B4 The rule lines, for Matthias to change at the desktop (a PC session does not edit them):
  - `CLAUDE.md`, the test commands: `python -m unittest discover -s modules/<pkg>/tests -t .    # camera/oak in .venv-oakd`
  - `.claude/rules/camera.md`: `- `modules/camera/oak4/` runs on depthai 3 in `.venv` (`install.bat`) and
    `modules/camera/oak/` on depthai 2 in `.venv-oakd` (`install.bat oakd`); an app imports one of them, never both`

## C. Install on site

- C1 No app runs. The old smile (not a Poser2 checkout) stays where it is, untouched, until D1 passes; its entry in
  the Startup folder is taken out. Git by hand if the PC has none; `git clone` of the branch into
  `C:\Developer\Poser2`, `git lfs pull`;
  `scripts\install.bat smile --startup` (the PC part, `.venv`, the face detector's engines for the site's GPU);
  `apps\smile\scripts\install.bat --from F:\_scratch\SMILE` (the disk from A6): ComfyUI, its models, the token and
  the photos in place, the smoke test passes; the log says `Face detector: … ready (TRT)`.

## D. Checks on site

- D1 The README's ON SITE CHECKS, each one; a check that passes gets its date there. Then the struck-through
  install steps of A8 are deleted from both READMEs.

## Open questions

## The rules it touches

- `.claude/rules/camera.md`, which backend runs in which venv: B4 (Matthias edits it)
- `CLAUDE.md` Running code, the test commands per venv: B4 (Matthias edits it)
- `CLAUDE.md` Running code, a run only on the user's go: A1–A7, B1, B3
- `CLAUDE.md` API evolution, a clean break and every call site in the same change: A2, A7
- `CLAUDE.md` Packages, code that does the same is written once: A2 (one app-to-venv table)
- `CLAUDE.md` Packages, a script sits in a `scripts/` folder: A1, A2, A4
- `.claude/rules/docs.md`, each fact in one place: A8, D1 (the struck lines go after D1)
- `.claude/rules/smile.md`, README.md current (setup, on-site checks): A7, A8, D1
