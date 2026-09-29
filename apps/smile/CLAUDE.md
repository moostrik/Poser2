# SMILE

Art installation: a camera captures a face, an image generator turns it into a portrait, and the
result is shown with music (Max patch via OSC) and lights (Art-Net). Entry points: `smile.py`
(main app, `modules/Forgettable.py`) and `pl_player.py` (image players). Python 3.12, venv in
`.venv`, installed with `scripts/install.bat`.

## Image generator

ComfyUI server (installed in `ComfyUI/` from the `comfy/` bundle, both git-ignored, API on
127.0.0.1:8188). `modules/comfy/`: `Generator` sends each capture to ComfyUI and turns its
websocket events into `GenMessage`s for the render (CAM_DEPTH → DIFFUSION → DIFF_LOOP → IDLE);
`workflow.py` adapts the API export in `data/comfy/` (face crop removed — SMILE crops and rotates
the face itself — depth outputs added). Keep one workflow per visitor: a second workflow in
between drops the first from ComfyUI's cache and reloads the models.

## Settings

Settings system copied from poser2 (`modules/settings/`); root tree in `modules/smile_settings.py`,
presets in `data/settings/`, web panel at `http://localhost:667`.

- `BaseSettings` subclasses are pure data containers with no runtime side effects
- Use `BaseSettings` for reactive configuration and `@dataclass` for plain value objects
- Components take their settings object (or a `Group`/`Child` of it) in the constructor, never as a global, and read fields when they use them
- Never unpack setting values into constructor parameters; a copied value goes stale when the panel changes it
- Cache only `Field.INIT` values, since they cannot change after `initialize()`
- Use `bind()` only to react to configuration changes, never to relay runtime data between components
- Keep bound callbacks thread-safe, and never write the field that triggered the callback
- Components that `bind()` must `unbind()` the same callbacks on teardown
- Treat `Field.READ` values as snapshots; copy arrays before handing them to another thread
- Keep `Field` descriptions to one short line of panel text: what the value is and its unit
