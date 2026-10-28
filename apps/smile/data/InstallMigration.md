# A better install protocol: one script, a model manifest, a private model mirror

**Simplified the same evening (2026-10-26):** the user found the private Hugging Face store, its upload/export tooling
and the models folder beside the repo overkill: "with this system on two PCs and a backup of the repository on my
backup disk we keep the important data; it is easy to copy a couple of files from a USB drive". So decisions 1 and 7
below are reversed: the models live in `ComfyUI\models` as ComfyUI has it (an update copies the folder into the new
install), they come from their original sources (the manifest keeps URL, size and sha256) or from a local folder the
install asks for, the three private LoRAs and the four pack archives live in the repo as git LFS files
(`data/comfy/loras`, `comfy/packs`), and a backup is a copy of the SMILE folder. The HF repo was deleted, and so was
`comfy/inventory.py` (the on-site inventory never ran; the locks and the manifest are the record of an install). The
rest of this document stays as the record of how the install path was built; where it names the inventory, the HF
repo or `..\smile-models`, that is history.

**Status (2026-10-26, evening): the install path is built and runs on the dev PC (IAAI).** `scripts\install.bat`
installs app, ComfyUI and models from `comfy\models.json`, `comfy\comfy.lock` and `comfy\requirements.lock` and ends
in a passing smoke test (the warm-up through a fresh ComfyUI in 20 s). What this round changed against the plan below,
and what is left, is in **Migration** at the end. Earlier status: the plan of 2026-10-06 (deferred until after the
presentation), merged with the clean-install plan of 2026-10-26: the installation runs at ICK, SMILE gets installed on
a second PC (24 GB card) to test the `online` branch, and the install must stay updatable. The additions are marked
**(2026-10-26)**: an inventory of the ICK PC as step 0, packs pinned by git commit, the patches removed, an update
path, the models in a folder outside ComfyUI, a smoke test at the end of the install, no downloads at runtime.

## Context

Today the ComfyUI side is installed by hand from `comfy/`, a 27 GiB "bundle" that was copied from the
old PC over USB: `comfy/setup.ps1` clones ComfyUI, fetches the node packs, applies the patches, builds
the venv and robocopies `comfy/models/` into `ComfyUI/models/`. `scripts/install.bat` only does the
app side and refuses to run twice. The bundle idea is now redundant, and it has drifted: the startup
preset `data/settings/smile.json` runs `realvisxlV40_v40Bakedvae` + the DMD2 LoRA, which exist only in
`ComfyUI/models/` on this PC, not in the bundle. A new PC built from the bundle would silently run the
Lightning checkpoint without distillation, because `Workflow.build()` drops a model ComfyUI does not
have without a word. The bundle also carries 10 GB the app never uses (Qwen 4B, MiDaS cache, animal
LivePortrait), there is no checksum anywhere (the comfy README itself warns about zero-filled
downloads), and nothing re-copies `aiia_grin` after it changes.

Decided with the user (2026-10-06):

1. **Storage**: one private Hugging Face repo of the user holds the exact verified files under their
   ComfyUI-relative paths (free up to 100 GB private; the set is ~18.5 GB). Each file also lists its
   public canonical URL as a fallback. A local mirror folder (USB SSD, or the iCloud Drive folder)
   serves offline installs, filled by an export step. The website is not used.
2. **Model set**: only what `data/comfy/smile_workflow.json` and the `smile.json` preset run (table
   below). The workflow and the code defaults change to that checkpoint + DMD2.
3. **Layout**: one re-runnable `scripts\install.bat` for app + ComfyUI + models; running it again is
   the install check. ComfyUI-side sources stay in `comfy/` (`patches/`, `aiia_grin/`, the manifest);
   the setup script moves to `scripts/`; `comfy/README.md` folds into `README.md`; the `comfy/models/`
   bundle convention disappears.
4. The old AIIAv3 workflows in `comfy/workflows/` are dropped (git history keeps them), and with them
   the `comfyui_controlnet_aux` pack (MiDaS) and its HF cache.

Added 2026-10-26:

5. **The ICK PC is the reference**, not the dev PC of October 6: presets and settings changed since.
   `comfy/inventory.py` (in the repo, stdlib only, read-only) records it: ComfyUI commit, every pack
   with its version and a diff against its Comfy Registry archive (hand edits we forgot), every model
   file with size and sha256, `pip freeze`. Its output (`output/inventory/`) fills the manifest, the
   `requirements.lock` and the pack pins below. Nothing else starts before it has run.
6. **Updatable**: ComfyUI core and the packs are pinned by git commit in `comfy/comfy.lock`, the
   patches go (section 7), and `install_comfy.ps1 -Update` builds the newest versions next to the
   install and switches over only when the smoke test passes (section 8).
7. **Models outside ComfyUI**: the models folder lives beside the repo (default `<repo>\..\smile-models`,
   `-Models` overrides) and `ComfyUI\models` is a directory junction to it, so a ComfyUI install can be
   thrown away and rebuilt, and packs that build their own paths under `models/` still find their files
   (`extra_model_paths.yaml` does not reach those). The model tool's default `--models` is that folder.
8. **No downloads at runtime**: `start_comfy.bat` sets `HF_HUB_OFFLINE=1`; a missing model is an error
   in the ComfyUI log instead of a silent download of whatever is current.

## Target layout

```
scripts/
  install.bat             ONE entry point: prerequisites, LFS, app venv, then install_comfy.ps1, then models
  install_comfy.ps1       from comfy/setup.ps1: ComfyUI at the pinned commit, 7 packs, patches, aiia_grin, venv; idempotent
  (start_comfy.bat, startSMILE.bat, make_test_workflow.bat, clip_info.bat unchanged)
comfy/
  models.json             the manifest: HF repo id + every model file (path, size, sha256, public source)
  comfy.lock              (2026-10-26) ComfyUI commit, each pack as git URL + commit, the torch index
  requirements.lock       pip freeze of the verified ComfyUI venv (exact reproduction)
  inventory.py            (2026-10-26) records an install: the reference for the locks
  patches/                (2026-10-26) empty when section 7 is done
  aiia_grin/              unchanged; installed as a junction (2026-10-26), not a copy
modules/comfy/models.py   the model tool (stdlib only, runs with `py -3.12` before any venv exists)
modules/comfy/tests/test_models.py
```

Deleted: `comfy/README.md`, `comfy/setup.ps1`, `comfy/workflows/`. `.gitignore` loses `comfy/models/`.
The 27 GiB `comfy/models/` folder on this PC is **not** deleted by the implementation: the user deletes it
by hand once the HF upload and the USB export are verified (see "Migration on this PC").

## The model set (manifest contents)

All paths relative to `ComfyUI/models/`. Sizes from this PC's working install.

| Path | Size | Public source (to confirm by LFS sha256, see `sources`) |
|---|---|---|
| `checkpoints/realvisxlV40_v40Bakedvae.safetensors` | 6,938,040,706 | HF `SG161222/RealVisXL_V4.0` (`RealVisXL_V4.0.safetensors`), Civitai name |
| `loras/dmd2_sdxl_4step_lora_fp16.safetensors` | 393,854,592 | HF `tianweiy/DMD2` |
| `loras/clothing_slider.safetensors` | 23,474,312 | none (user's own, ai-toolkit) |
| `loras/looking_at_viewer.safetensors` | 8,789,076 | none (no metadata) |
| `loras/dark.safetensors` | 34,340,216 | none (no metadata) |
| `controlnet/control_instant_id_sdxl.safetensors` | 2,502,139,136 | HF `InstantX/InstantID` `ControlNetModel/diffusion_pytorch_model.safetensors` |
| `instantid/ip-adapter_instant_id_sdxl.bin` | 1,691,134,141 | HF `InstantX/InstantID` `ip-adapter.bin` |
| `controlnet/t2i-depth_diffusers.safetensors` | 158,060,416 | HF `TencentARC/t2i-adapter-depth-midas-sdxl-1.0` `diffusion_pytorch_model.fp16.safetensors` (to confirm) |
| `geometry_estimation/depth_anything_3_mono_large.safetensors` | 1,336,748,056 | HF `Comfy-Org/Depth-Anything-3` `geometry_estimation/depth_anything_3_mono_large.safetensors` (ComfyUI's own blueprint URL) |
| `insightface/models/antelopev2/{1k3d68,2d106det,genderage,glintr100,scrfd_10g_bnkps}.onnx` | 427,550,200 | none: antelopev2 ships as one zip (insightface release), no per-file URL |
| `liveportrait/{appearance_feature_extractor,motion_extractor,spade_generator,stitching_retargeting_module,warping_module}.safetensors`, `landmark.onnx`, `landmark_model.pth` | 750,277,450 | HF `Kijai/LivePortrait_safetensors` |
| `ultralytics/face_yolov8n.pt` | 6,230,011 | HF `Bingsu/adetailer` |
| `LLM/Qwen-VL/Qwen3-VL-2B-Instruct/` (`model.safetensors` + `config.json`, `generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `chat_template.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`) | 4,266,645,006 | HF `Qwen/Qwen3-VL-2B-Instruct` |

Total ≈ 18.5 GB. Not in the manifest (the `check` command lists them as *extra*, the user removes them
by hand if wanted): `realvisxlV40_v40LightningBakedvae`, `mopMixtureOfPerverts_v31`,
`Black_and_white.safetensors`, `upscale_models/RealESRGAN_x2plus.pth`, `liveportrait/animal/`,
`LLM/Qwen-VL/Qwen3-VL-4B-Instruct/`, the `.cache` folder inside the Qwen folder, the MiDaS HF cache.
The torch-hub files of the face-crop nodes (`2DFAN4`, blazeface, ~97 MB) are used only by
`smile_test.json` in the ComfyUI page and download themselves at first use: documented, not in the manifest.

### Manifest shape, `comfy/models.json`

```json
{
  "repo": "<hf-user>/smile-models",
  "files": [
    {"path": "checkpoints/realvisxlV40_v40Bakedvae.safetensors",
     "size": 6938040706, "sha256": "…",
     "source": "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors",
     "note": "SDXL checkpoint, VAE baked in"},
    {"path": "loras/clothing_slider.safetensors", "size": 23474312, "sha256": "…", "source": null,
     "note": "own LoRA (ai-toolkit)"}
  ]
}
```

The primary URL of every file is `https://huggingface.co/<repo>/resolve/main/<path>`; `source` is the
fallback (null = none). `sha256` is what makes a mirror or a download trustworthy.
The implementation ships the manifest complete: all 33 entries with their sizes, the `source` URLs above,
and the sha256 values computed from this PC's `ComfyUI/models` (reading 18.5 GB once, ~1 min). Only
`repo` is a placeholder (`<hf-user>/smile-models`) until the user has the HF repo; the tool refuses the
HF source with a clear message while it is a placeholder, and still installs from a mirror or `source`.

## 1. The model tool: `modules/comfy/models.py`

Stdlib only (`json`, `hashlib`, `urllib.request`, `shutil`, `argparse`, `logging`, `getpass`), no
import of any sibling module, so `py -3.12 -m modules.comfy.models …` runs from the repo root before a
venv exists (`modules` is a namespace package without `__init__.py`, as `pytest.ini` notes; verified to
import that way). Console tool: `logging` to stderr (`basicConfig`, `%(levelname)s %(message)s`), a
one-line `\r` progress for downloads when stderr is a tty, else a line per 10 %. Defaults: manifest `<repo>/comfy/models.json`, models
folder `<repo>/ComfyUI/models` (from `Path(__file__).resolve().parents[2]`), both overridable
(`--manifest`, `--models`).

Subcommands:

- `check [--quick]`: every manifest file is `ok`, `missing`, `wrong size` or `corrupt` (sha256 with
  `hashlib.file_digest`, ~1-2 GB/s; `--quick` stops at the size). Lists *extra* files in the manifest's
  folders as information. Exit 1 when anything is not ok.
- `install [--mirror DIR] [--quick]`: for each file that is not ok, in order: copy from the mirror if
  it has the path (then verify); else download from the HF repo; else from `source`. A download goes to
  `<path>.part`, resumes with `Range` when a `.part` exists (206 expected, else start over), is renamed
  into place only after the sha256 matches, and a hash mismatch deletes the file and tries the next
  source. Prints a summary; exit 1 when something is still missing.
  - HF auth: token from `HF_TOKEN`, else `%USERPROFILE%\.cache\huggingface\token` (where `hf auth login`
    writes it), else `getpass` prompt once, saved to that file. Without a token the HF repo is skipped
    with a note, and only `source` is tried (so a PC without the token still gets every public model).
  - Redirects: `…/resolve/main/…` answers 302 to a pre-signed CDN URL on another host that rejects an
    `Authorization` header, and that 302 carries `x-linked-etag` (the LFS sha256) and `x-linked-size`
    (verified with curl). So `resolve(url, token)` does a HEAD with the token as
    `add_unredirected_header` through an opener that returns the 302 instead of following it, and a
    source whose etag or size disagrees with the manifest is skipped before a byte is downloaded (the
    zero-filled-download case). The GET then goes to `Location` without the token, with `Range`.
    Python 3.12's redirect handler would forward a header set with `add_header`: never do that.
- `export DIR`: verifies, then copies the manifest's files into `DIR/<path>` (skipping ones already there
  with the right size and hash) and writes `DIR/models.json` beside them. This fills the USB SSD and the
  iCloud Drive folder.
- `hash [--write] [--add PATH …]`: computes size + sha256 of the manifest's files present under the
  models folder and prints `path size sha256 unchanged|CHANGED|new`; `--write` updates the manifest
  (keeping `source` and `note`), `--add` adds entries for new paths with `source: null`. This is how the
  manifest is updated after a model change on the dev PC.
- `sources [--repo]`: HEADs every public `source` and compares its `x-linked-etag`/`x-linked-size` with
  the entry (confirms a fallback without downloading it; a mismatch is reported and, with `--write`, sets
  `source` to null). `--repo` does the same for the HF repo's copies through
  `/api/models/<repo>/tree/main?recursive=true` (`lfs.oid`, `size`) with the token: the check after `upload`.
- `upload`: verifies, then runs the ComfyUI venv's `ComfyUI\.venv\Scripts\hf.exe upload <repo> <models-dir> . --include <each manifest path> --private`
  (huggingface_hub 1.33 is in that venv; the LFS upload protocol is not worth re-implementing). Prints the
  command and fails clearly when the venv or the token is missing.

Keep the network behind one seam, `_http(url, method, headers) -> Response(status, headers, read)`, so the
tests replace it with a fake keyed by URL. ComfyUI never offers a `.part` file (`folder_paths` filters by
extension), so a half download is invisible to it.

## 2. `scripts/install_comfy.ps1` (from `comfy/setup.ps1`), idempotent

Parameters `-Target` (default `<repo>\ComfyUI`) and `-ComfyCommit` as now; `$src` becomes `<repo>\comfy`.
Exit non-zero on any failure (`$ErrorActionPreference = 'Stop'`, the body in `try/catch` that prints the
error and `exit 1`, `$LASTEXITCODE` checked after every `git`/`pip`/`tar` call) so `install.bat` can stop.

- **Prerequisites** as now (NVIDIA driver ≥ 580, `py -3.12`).
- **ComfyUI core**: clone when `$Target\.git` is missing; `git rev-parse --verify <commit>^{commit}` or
  fetch; `git checkout -B smile <commit>`; `git status --porcelain` must be empty for tracked files
  (core stays unmodified).
- **Packs** (2026-10-26): from `comfy/comfy.lock`, each a git clone at its commit (the registry archive
  of a version and the pack's tagged commit hold the same files; the inventory's diff proves it per
  pack). A folder that exists is fetched and checked out at the pin; a dirty one fails with its name.
  The list: `comfyui-advancedliveportrait`, `comfyui_instantid`, `ComfyUI-QwenVL` (what SMILE runs),
  `comfyui-liveportraitkj`, `comfyui-kjnodes`, `comfyui-videohelpersuite` (only `smile_test.json`
  in the ComfyUI page: the face crop and the video recorder), `ComfyUI-Crystools` (optional, GPU
  monitor). `comfyui_controlnet_aux` goes.
- **Patches**: `git apply --check` first; when that fails, `git apply --reverse --check` succeeding means
  already applied → skip; otherwise fail with the patch name. (2026-10-26: none left after section 7;
  the step stays for a future stopgap.)
- **aiia_grin** (2026-10-26): `custom_nodes\aiia_grin` is a directory junction to `comfy\aiia_grin`
  (`New-Item -ItemType Junction`), so an edit needs a ComfyUI restart and nothing else. A folder that
  is a copy from the old setup is replaced.
- **venv**: create when missing. Install from `comfy/requirements.lock` (pip freeze of the verified venv:
  192 lines, no local-wheel entries, so it reproduces from the indexes; minus the plain `onnxruntime` line,
  plus `--extra-index-url https://download.pytorch.org/whl/cu130` so `torch==2.14.0+cu130` resolves). The pack `requirements.txt` files are not run (the lock supersedes
  them; liveportraitkj's `numpy<=1.26.4` pin would otherwise downgrade numpy; ComfyUI's own
  `requirements.txt` is in the lock too). torch is pinned by the lock (`torch==2.14.0+cu130`), where today
  a fresh PC gets whatever is newest. A stamp file `ComfyUI\.venv\smile-install.stamp` holding the sha256
  of `requirements.lock` skips the pip step when it matches, so the lock is the one place to change pins.
  The onnxruntime GPU fix (`--force-reinstall --no-deps onnxruntime-gpu==1.30.0`) runs only when
  `python -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers()"` fails.
- **Models junction** (2026-10-26): `ComfyUI\models` → the models folder (`-Models`), created when the
  folder is not a junction yet; an existing real `models\` folder with files in it is moved into the
  models folder first (name clashes fail, nothing is overwritten).
- No models step here: `install.bat` runs the model tool after this script. Final line says so.
- `scripts/start_comfy.bat` line 16: the hint "see comfy\README.md" becomes "run scripts\install.bat".

## 3. `scripts/install.bat`

Keep the colour-echo style and sections; extra arguments go to the model tool (`scripts\install.bat --mirror E:\smile-models`).

1. Checks: `git`, `git lfs`, `py -3.12` (new), `ffmpeg` (warn only, as now).
2. `git lfs pull`.
3. App venv: create when missing; always `pip install -r requirements.txt` (idempotent). Remove the
   "already exists → Install not successful" branch.
4. `powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT_DIR%\scripts\install_comfy.ps1"`; stop on error.
5. `pushd "%ROOT_DIR%"`, `py -3.12 -m modules.comfy.models install %*` (so `install.bat --mirror E:\smile-models`
   passes through), `popd`; stop on error. Runs with the system Python before any venv; prompts for the
   token once when the HF source is needed.
6. **Smoke test** (2026-10-26): starts ComfyUI from the venv (the way `start_comfy.bat` does, in the
   background), waits for `/system_stats`, checks the log for `Device: cuda` and the venv for the CUDA
   onnxruntime provider, queues the warm-up workflow (`modules.comfy.workflow.build_warmup`, through
   the app venv) and waits for its result, then stops ComfyUI. A failure names the step.
7. "Install complete" only when every step passed; a second run prints skips everywhere.

## 4. Patches: down to none (round 2a, **done 2026-10-26**: `comfy/patches` is empty, four unpatched packs; 4.1 proven by 16 identical grin frames on one portrait, old code with patch against new code without)

ComfyUI core is never patched. A patch on a pack is a stopgap: the install applies it, recognises it as applied,
or stops by its name, so an update cannot drop one unnoticed. Round 2 makes them unnecessary, so that any version
of a pack installs clean. What each patch is and what replaces it:

1. **`advancedliveportrait-mask.patch`** (the only one that changes the picture). The pack's `prepare_source`
   (`nodes.py` ~354) warps its paste mask with a transform made from `face_region`, and when the face square was
   clipped at the image edge (`is_changed`) it then remakes the transform from `crop_region` but keeps the mask
   where it was: pasted face and mask disagree. The patch makes both from `crop_region`. Grin Frames
   (`comfy/aiia_grin/__init__.py`) already redoes every per-frame step itself (`render_frame`) and only takes the
   prepared source (`psi`) from the pack's `ExpressionEditor.run`. It will make the two values itself: after
   `editor.run`, call the pack's engine `detect_face(img_rgb, crop_factor)` once more (the same detector, the same
   answer, milliseconds), build `crop_trans_m` with the pack's `create_transform_matrix` from `crop_region` and
   `mask_ori` by warping `engine.GetMaskImg()` with it, and use those in `render_frame` instead of `psi.crop_trans_m`
   and `psi.mask_ori`. The debug print the patch removed is already swallowed (`redirect_stdout`). Proof: the
   record of a saved clip (`modules.save.record --preset`) rendered again with the patch and without it gives
   the same frames (`test_grin_mask.py` in `comfy/aiia_grin` is not possible without ComfyUI: the user compares
   two clips, or `smile_test.json` with a fixed seed, by eye and by a frame diff).
2. **`liveportraitkj-import.patch`** (a broken relative import in `face_alignment/api.py`). The pack serves only
   the face crop of `smile_test.json` in the ComfyUI page (the subgraph with `LivePortraitCropper`,
   `LivePortraitLoadFaceAlignmentCropper`, `DownloadAndLoadLivePortraitModels` and kjnodes' `ImageResizeKJv2`;
   `Workflow` strips it for SMILE, `testflow` puts it back behind the "use face crop" switch). It goes:
   - a command `python -m modules.face.crop <photo> [-o out.png]` crops a photo as SMILE does (`FaceFinder` +
     `portrait_crop`, the eyes levelled, the workflow's portrait size) so any photo can be tested in the page;
   - `testflow` loses the crop branch: the switch becomes a second `LoadImage` (the cropped photo) or goes
     altogether (a `LoadImage` is a file picker already);
   - the user deletes the crop subgraph in the ComfyUI page and re-exports `data/comfy/smile_workflow.json`;
     `Workflow.__init__` loses the crop removal (`_find(LivePortraitCropper)`, `_remove(crop_nodes)`), the tests
     that assert the crop is gone (`test_face_crop_removed_and_rewired`) assert there was none;
   - `comfyui-liveportraitkj` and `comfyui-kjnodes` leave `comfy.lock`; `liveportrait/landmark_model.pth` (only
     liveportraitkj loads it) leaves the manifest (`check` then lists it as extra, the user deletes it).
3. **`qwenvl-8bit-dtype.patch`** (8-bit loading gets `torch.float16`). SMILE runs `None (FP16)`
   (`smile_workflow.json` node 1002:17); the 8-bit path is only the pack's fallback under VRAM pressure, which a
   24 GB card does not reach. The one-liner is offered upstream (`1038lab/ComfyUI-QwenVL`), the file is dropped,
   and the README's log-lines note says what a `dtype` error in 8-bit mode means.
4. Optional, same spirit: `comfyui-videohelpersuite` serves only the video recorder of `smile_test.json`;
   ComfyUI core's `CreateVideo` + `SaveVideo` nodes do the same, so `testflow` can use those and the pack leaves
   the lock. `ComfyUI-Crystools` (GPU monitor) stays, unpatched and optional.

End state: `comfy/patches` is empty (the installer's patch step stays for a future stopgap), four packs in the
lock (`advancedliveportrait`, `instantid`, `QwenVL`, `Crystools`), each at its registry version, none modified.

## 5. Updating (round 2b, **built 2026-10-26**: `install_comfy.ps1 -Update` / `-Accept` / `-Rollback` as below; the first update on the dev PC, ComfyUI `4ef23c34` to `cdf272f3` and QwenVL 2.2.0 to 2.3.2, gave the same fixed-seed portrait and needed one workflow change, the new `video_frame_size` input of the describe node)

The install is already rebuildable (the models live beside it, `install.bat` is idempotent), so an update is:
new pins, a fresh build next to the old one, the proof, then the switch. `scripts\install_comfy.ps1 -Update`:

1. `ComfyUI\` is renamed `ComfyUI-previous\` (the junctions inside it keep working; nothing is deleted).
2. The pins are renewed in memory: ComfyUI at `origin/master` HEAD (or `-ComfyCommit`), each pack at the
   registry's `latest_version` (or the version given); the venv is built **unlocked**: ComfyUI's own
   `requirements.txt`, then every pack's `requirements.txt`, then the known extras of today's lock that no pack
   declares (`numba`, the `numpy` the install runs, `onnxruntime-gpu` forced last). Everything else of the
   install path runs as is: junctions, patches (none), the CUDA provider check.
3. The proof: `modules.comfy.smoke` (`--comfy ComfyUI`) as in every install, plus `smoke.py --save` writes the
   portrait and the last grin frame of a **fixed-seed** run of `data/settings/GenerationTest.json` to
   `output\smoke\<stamp>\`; the user puts them next to the previous install's (made the same way before the
   update) and judges. A seed that gives the same picture is not expected across torch versions, a picture of the
   same quality is.
4. On a pass the user says so (`-Accept`): `comfy.lock` gets the new commit and versions (`git rev-parse` and the
   packs' `pyproject.toml`), `requirements.lock` the new `pip freeze --all` (minus the plain `onnxruntime`), the
   stamp file is written, and `ComfyUI-previous\` can be deleted. On a failure `ComfyUI-next` stays for a look,
   and `-Rollback` renames `ComfyUI-previous\` back; nothing of the locks changed.
5. First on the dev PC, never on the ICK PC during the show; the ICK PC then gets the new locks with `git pull`
   and `install.bat` (a changed lock rebuilds the venv, a changed commit or version re-clones).

Order of work: 4.1 (the mask, the one that matters), 4.2, 4.3, 4.4, then 5. Each step keeps `install.bat` passing.

## 6. Code changes

- `data/comfy/smile_workflow.json`: node `1002:20` `ckpt_name` → `realvisxlV40_v40Bakedvae.safetensors`.
  Regenerate `data/comfy/smile_test.json` with `scripts\make_test_workflow.bat` (the user runs it: it
  needs the app venv; it does not need ComfyUI).
- `modules/comfy/settings.py` (`PortraitSettings`, lines 51-63): defaults become the `smile.json` values so
  a fresh preset is a working model: `checkpoint_options` and `checkpoint` →
  `realvisxlV40_v40Bakedvae.safetensors`; `lora_options` → `["none", "dmd2_sdxl_4step_lora_fp16.safetensors"]`;
  `distill_lora` → `dmd2_sdxl_4step_lora_fp16.safetensors`; `distill_strength` 1.0; `steps` 4; `cfg` 1.2;
  `sampler` `lcm` (and `sampler_options` `["lcm"]`); `scheduler` stays `exponential`. `GenerationTest.json`
  and `Grintest.json` set no checkpoint or distill LoRA and so follow the new defaults (their own steps,
  cfg and sampler stay; they are test presets, the user adjusts them if wanted).
  Existing tests that assume no distill node by default: `test_lora_chain` and `test_distill_lora` in
  `modules/comfy/tests/test_workflow.py` are updated (the chain starts at the distill node by default;
  switching the distill LoRA off makes it start at the checkpoint).
- `modules/comfy/workflow.py` `build()` (lines 201-217): when the checkpoint, the distill LoRA or a slot's
  LoRA is skipped because it is not in ComfyUI's option list, log a `WARNING` naming the file and what is
  used instead, once per name per `Workflow` instance (a `set` on `self`), so a fresh install that misses a
  model is visible in the log instead of silent. `workflow.py` has no logger yet: add
  `logger = logging.getLogger(__name__)` as `Generator.py:35` does. Test with
  `self.assertLogs('modules.comfy.workflow', 'WARNING')` (the repo's idiom) for a missing checkpoint,
  distill LoRA and slot file, and `assertNoLogs` when every file is in the options. Test in `modules/comfy/tests/test_workflow.py` with `caplog`.
- `CLAUDE.md`: the Image generator paragraph (line 77: "installed in `ComfyUI/` from the `comfy/` bundle;
  `ComfyUI/` and `comfy/models/` are git-ignored") → installed by `scripts/install.bat` (`install_comfy.ps1`
  + the manifest `comfy/models.json`, models from the private HF repo or a mirror); line 119 (`aiia_grin`
  copied again by hand) → copied by every run of the install script.
- `.gitignore`: the `comfy/models/` line stays until the user has deleted that folder (otherwise
  `git status` lists 27 GiB as untracked); its removal is the last migration step below.

## 7. Docs: `README.md`

Replace "Install script" and "Image generator (ComfyUI)" under INSTALLATION with one section:

- prerequisites stay as they are (VC++, Python 3.12, Git + LFS, ExecutionPolicy, ffmpeg, NVIDIA driver ≥ 580);
- **one command** `scripts\install.bat [mirror]`: what it does, that it is safe to run again and then
  acts as the check, the HF token prompt (where it is stored, never in git), the mirror folder
  (USB SSD / iCloud Drive) for a PC without internet, how long the download is (~18.5 GB);
- **after a model change**: `hash`, `sources`, `upload`, `export`, in that order, and the extra-files note;
- **ComfyUI log lines** (the "Safe to ignore" / "Not ignorable" list from `comfy/README.md`, trimmed) under
  RUN → "Pieces on their own", since it is about running, not installing;
- the pinned versions live in `comfy/comfy.lock` and `comfy/requirements.lock` (say so; no table), and
  the update procedure of section 5 in a few lines;
- the models folder beside the repo and the junction (2026-10-26).

Delete `comfy/README.md`, `comfy/setup.ps1`, `comfy/workflows/`.

## 8. Tests: `modules/comfy/tests/test_models.py`

The repo's idiom: `unittest.TestCase`, `tempfile.TemporaryDirectory()` + `addCleanup`, files of a few
bytes, `_http` replaced by a fake keyed by URL (302 with `x-linked-*` for `resolve` URLs, 206 honouring
`Range` for the CDN URL, recording the headers it received):

- manifest load/save round trip, `hf_url`; `hash --write` keeps `source`/`note`, `--add` adds `source: null`;
- `check`: ok / missing / wrong size / corrupt, the extras list (a `put_x_here` and a `.cache` file ignored),
  `--quick` catches the size but not the hash;
- `install` from a mirror: copies, verifies, rejects a corrupt mirror copy and fetches from HF instead;
- download: the token only on the HF HEAD and never on the CDN GET; a source whose `x-linked-etag`
  disagrees is skipped; a resume from a half `.part` sends `Range: bytes=N-`; a wrong-hash download deletes
  the `.part` and falls to `source`; no `.part` left after success;
- token lookup order (env, file, patched `getpass`) and the saved file;
- `export` copies, skips what is already right, writes `models.json`; `main([...])` exit codes 0/1.

## Migration

### Done on the dev PC IAAI, 2026-10-26 (round 1: the install path)

The reference turned out to be **the install that runs**, not the ICK PC's bytes: nothing was run on the ICK PC
(its repository is not touched remotely), only its three private LoRAs were copied over. The 28 public models were
taken from the dev install `C:\Developer\ComfyUI\models` (every one happens to equal its canonical Hugging Face file,
`sources` confirmed it, but that is a record, not a gate), Depth Anything 3 was downloaded, and `hash --write` wrote
what is in the folder into the manifest after the smoke test passed. The pins are what the dev PC ran: ComfyUI
`4ef23c34`, the packs as **registry archives by version** (not git commits: matching a commit to an archive byte for
byte was not worth it, and the archives are what both PCs run), the venv a `pip freeze` of the dev venv, installed
with `--no-deps` (facenet-pytorch's `numpy<2` metadata contradicts the numpy 2.5.3 that runs).

- `modules/comfy/models.py` + tests (section 1, 8), `comfy/models.json` (32 files; the doc's 33 was a miscount), `comfy/comfy.lock`,
  `comfy/requirements.lock`, `scripts/install_comfy.ps1`, `scripts/install.bat`, `modules/comfy/smoke.py` (section 2, 3),
  the code changes (section 6), README and CLAUDE.md (section 7); `comfy/README.md`, `comfy/setup.ps1`, `comfy/workflows/` deleted.
- Found on the way: the registry serves some packs as a zip named `node.tar.gz` (`inventory.py` sniffs it); the pack archives
  have CRLF files, so the patches apply with `--ignore-whitespace`; clones are made with `core.autocrlf=false`.
- Not done: the HF upload (`repo` is still the placeholder: the user creates the repo and a Write token, puts
  `<user>/smile-models` in the manifest, runs `upload` and `sources --repo`), the USB/iCloud `export`, SMILE's own run here
  with a visitor (the user's check).

### Left

1. HF: repo, token, `py -3.12 -m modules.comfy.models upload`, then `sources --repo`; `export E:\smile-models` for the USB SSD
   and iCloud Drive (set to "Always keep on this device").
2. Section 4 (patches to none) and section 5 (`-Update`), later rounds.
3. The ICK PC, on site: `py -3.12 comfy\inventory.py` as the check of what it runs against the locks, `git pull`,
   `scripts\install.bat` (its `ComfyUI\models` is moved into `..\smile-models` by the junction step; never during the show),
   delete `comfy\models\` (27 GiB) by hand, drop the `comfy/models/` line from `.gitignore`.
4. The `online` branch: merge `main` into it and run it on the dev PC.

## Verification

- `.venv\Scripts\python -m pytest` passes (new `test_models.py`, the updated `test_workflow.py`).
- `py -3.12 -m modules.comfy.models check` on this PC: every manifest file `ok`, the extras listed.
- `sources` confirms (or refutes, setting null) every public URL in the table above.
- `scripts\install.bat` run twice: the second run skips every step and ends in "Install complete",
  the smoke test included (2026-10-26).
- `install_comfy.ps1 -Update` on the test PC once, with the lock files as the proof (2026-10-26).
- `export` into a temp folder, then `install --mirror <that folder> --models <empty temp dir>` fills the
  empty dir and `check --models <it>` passes (this doubles as the offline-install test, ~18.5 GB copy).
- The user starts ComfyUI (`scripts\start_comfy.bat`) and SMILE: the warm-up passes, the panel's
  checkpoint shows `realvisxlV40_v40Bakedvae` with DMD2; a run of `smile_test.json` in the page works.
  (No ComfyUI test loops by Claude: the user verifies in the running apps.)

## Risks and notes

- The HF token lives in the user profile, never in the repo; the manifest holds only the repo id.
- A private mirror of third-party models is for the installation's own use (RealVisXL is OpenRAIL++,
  Qwen Apache-2.0, InstantID Apache-2.0); nothing is redistributed.
- `.part` files: an interrupted install leaves one next to the target; the next run resumes it.
- `git ls-files` tracks `data/settings/Smile.json` while the disk has `smile.json` (`core.ignorecase`
  hides it on Windows); `git mv -f Smile.json smile.json` fixes the case. Side note, not part of this change.
- Packs are git clones at a commit (2026-10-26); a pack whose repository disappears is copied from a
  good install, as before. A pack without a public git repository stays a registry archive, pinned by
  version in `comfy.lock`.
- Poser2 (2026-10-26): everything ComfyUI stays in `comfy/` and `scripts/install_comfy.ps1`, both
  self-contained with paths relative to the repo root, so they move into Poser2 as a unit when SMILE
  does; the model tool is the one that Poser2's own models can use too.
- The manifest must stay complete, because the packs stop healing a half-present folder: QwenVL skips
  its download when the folder has any `.safetensors` (weights without tokenizer files never heal),
  liveportraitkj downloads only when `models/liveportrait` is absent altogether.
