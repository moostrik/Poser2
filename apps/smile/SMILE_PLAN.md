# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

The new smile goes on location. The Mako, the lights and the sphere are on site; the site PC runs the old smile.
A new PC is installed as the root README's "New PC" says (`scripts\install.bat <app>`); smile's site folder, what git
does not bring, is on the scratch disk, `F:\_scratch\SMILE`.

## A. Before leaving

- A1 The rehearsal: a new clone of the pushed branch (pushed on Matthias's word) in place of
  `C:\Developer\SmileClean`, installed only as the README's "New PC" says: `scripts\install.bat smile`, smile's
  `install.bat --from F:\_scratch\SMILE`; smile's tests, one `-sim` start with ComfyUI and a visitor through.
- A2 A committed state that passes smile's tests, the guard tests and pyright; no other thread's uncommitted work in
  it.
- A3 A live smile start on the OAK4 (`-oak`) in `.venv`, ComfyUI running, Matthias in front: one visitor through,
  from the flash to the saved grin and portrait.
- A4 The rule lines, for Matthias to change at the desktop (a PC session does not edit them):
  - `CLAUDE.md`, the test commands: `python -m unittest discover -s modules/<pkg>/tests -t .    # camera/oak in .venv-oakd`
  - `.claude/rules/camera.md`: `- `modules/camera/oak4/` runs on depthai 3 in `.venv` (`install.bat`) and
    `modules/camera/oak/` on depthai 2 in `.venv-oakd` (`install.bat oakd`); an app imports one of them, never both`

## B. Install on site

- B1 No app runs. The old smile (not a Poser2 checkout) stays where it is, untouched, until C1 passes; its entry in
  the Startup folder is taken out. Git by hand if the PC has none; `git clone` of the branch into
  `C:\Developer\Poser2`, `git lfs pull`;
  `scripts\install.bat smile --startup` (the PC part, `.venv`, the face detector's engines for the site's GPU);
  `apps\smile\scripts\install.bat --from F:\_scratch\SMILE` (the scratch disk): ComfyUI, its models, the token and
  the photos in place, the smoke test passes; the log says `Face detector: … ready (TRT)`.

## C. Checks on site

- C1 The README's ON SITE CHECKS, each one; a check that passes gets its date there. Then the struck-through
  install steps in the root README and smile's README are deleted.

## Open questions

## The rules it touches

- `.claude/rules/camera.md`, which backend runs in which venv: A4 (Matthias edits it)
- `CLAUDE.md` Running code, the test commands per venv: A4 (Matthias edits it)
- `CLAUDE.md` Running code, a run only on the user's go: A1, A3
- `.claude/rules/docs.md`, each fact in one place: C1 (the struck lines go)
- `.claude/rules/smile.md`, README.md current (on-site checks): C1
