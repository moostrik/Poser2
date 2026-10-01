# ComfyUI generation server for smile

Everything the smile app needs to rebuild its ComfyUI server on a new PC. Verified against the dev PC
install of 2026-09-28 (ComfyUI 4ef23c34, RTX 4090, driver 580.97, Python 3.12).

Pipeline: photo -> face crop (LivePortrait) -> describe (QwenVL) -> portrait (RealVisXL Lightning
+ InstantID + T2I depth) -> grin animation (Grin Frames) -> 512x640 frames + video.
Workflow: `workflows/AIIAv3 portrait grin v2.json` (the verified reference, used in step 3).
`workflows/AIIAv3 portrait grin v3.json` holds the portrait experiments: three LoRAs from
`models\loras` (clothing_slider, looking_at_viewer, dark), grin bypassed. The smile app reads
neither: it runs `data/comfy/smile_workflow.json` (API format, with the same three LoRAs)
and fills in the values from its panel. That one takes its depth from ComfyUI's own Depth Anything 3
nodes (`models\geometry_estimation\depth_anything_3_mono_large.safetensors`, kept loaded by
ComfyUI); the two workflows here still use the MiDaS node of `comfyui_controlnet_aux`. Once they are
saved with the Depth Anything 3 nodes too, that pack and `models\hf-cache` can leave the bundle.

To test the app's workflow in the ComfyUI page, generate it: `.venv\Scripts\python -m
modules.comfy.testflow` (in the repo; `-p` preset, `-s a|b` style) writes
`data\comfy\smile_test.json`; drag that file into the page. It is what the app sends, with the preset's
values, plus a switch for the input (the app's last capture `smile_capture.png`, or any photo
through the face crop), previews of the depth maps, description, prompt and overlay, and the video
recorder. The `AIIAv3` workflows here are the previous design.

The smile app starts the server with `scripts\start_comfy.bat`: the known harmless warnings
filtered, and a log per start in `output\logs\comfy_*.log`. With every model in place a run makes no
internet calls; a pack downloads a model only when it is missing (QwenVL on first use of another
model, for instance).

This folder carries everything including the models (~23 GB): put it on a USB drive, copy it into
the smile repo as `comfy/`, run the script. The server installs next to it, at `<repo>\ComfyUI`
(override with `-Target`). Add both to the repo's .gitignore - neither belongs in git:

```
comfy/models/
/ComfyUI/
```

## 1. Python 3.12 (once, safe next to an existing Python)

The script needs Python 3.12 via the `py` launcher; an existing 3.10 keeps working, since venvs
hardwire their own interpreter and 3.12 installs into its own folder.

- Install 3.12.x (64-bit) from python.org, choose **Customize installation** and **uncheck
  "Add python.exe to PATH"** (keep the py launcher checked; optionally uncheck "Associate files").
- Verify: `python --version` still shows the old default, `py --list` now shows `-3.12`.

## 2. Run the setup script (needs internet)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

What it does:

- clones ComfyUI at the pinned commit (on a local branch `smile`)
- downloads the 8 node packs from the Comfy Registry at the exact pinned versions (table below)
- applies the three patches in `patches\` (pack fixes; ComfyUI core stays unmodified)
- installs the bundled `models\` (existing files are left alone; `models\hf-cache\` goes to
  `%USERPROFILE%\.cache\huggingface\hub\` - that is MiDaS for the two workflows here)
- installs `aiia_grin` (our nodes: Grin Frames, with LivePortrait stitching switchable, and
  KSampler (full preview), which previews every sampler step with the real VAE; the smile app
  swaps it in for the KSampler) and the workflow
- builds the venv: torch cu130, ComfyUI requirements, comfyui-manager 4.2.2, every pack's
  requirements, then numba 0.67.0, numpy 2.5.3 and a forced onnxruntime-gpu 1.30.0 reinstall
  (undeclared/conflicting pack deps, pinned to the dev-verified state)

The bundled models were byte-verified against the dev PC. Two fresh downloads there once arrived
full-size but zero-filled, so prefer these copies over re-downloading; if a model ever behaves
oddly, compare its SHA256 with the dev PC copy. Keep `models\diffusion_models` empty.

## 3. Verify

```powershell
cd <repo>\ComfyUI; .venv\Scripts\activate
python main.py --enable-manager --preview-method auto
```

- Log shows `Device: cuda:0 NVIDIA GeForce RTX ...`.
- Open "AIIAv3 portrait grin v2", queue a photo: log shows a 2385MB ControlNet load, the portrait
  resembles the photo (check the Overlay preview), the whole face animates, video is 512x640.
- The first queue takes ~80 s (QwenVL 8-bit load, LivePortrait pipelines, MiDaS, landmark
  warmup). Everything stays loaded; after that a new photo takes ~23 s end to end (crop ~2.5 s,
  120 grin frames + video included, ~19 s with QwenVL on FP16).
- QwenVL runs `Qwen3-VL-4B-Instruct` at `None (FP16)` - on a 24 GB card that is ~4 s faster per
  photo than 8-bit (bitsandbytes int8 trades speed for VRAM). It auto-falls-back to 8-bit under
  VRAM pressure; the 8-bit dtype patch covers that path. The -FP8 variants need Triton and fail
  on Windows. That holds for the two workflows here. The smile app's workflow uses
  `Qwen3-VL-2B-Instruct` at `None (FP16)` (`models\LLM\Qwen-VL\`), because the card is full:
  measured per description, 2B FP16 1.0 s, 4B FP16 1.6 s, 2B 8-bit 3.6 s, 4B 8-bit 4.9 s, and 2B
  FP16 takes ~4.4 GB less of the card than 4B FP16. The 2B texts are a bit less precise.
- Previews inside the subgraphs (such as Face crop's Crop preview) show on the subgraph node
  only with the Nodes 2.0 setting (`Comfy.VueNodes.Enabled`). The root previews (Overlay, All
  frames) always work.

## Log lines

Safe to ignore:

- FutureWarnings (`torch.jit.load`, `ToTensor()`, `estimate`, pynvml), `triton not found`,
  torchsde `Should have tb<=t1`: `scripts\start_comfy.bat` filters these (`PYTHONWARNINGS`), a
  start by hand shows them.
- `0 models unloaded.` at every sampler step: ComfyUI fills the card on purpose (dynamic VRAM,
  ~23.6 of 24.5 GB at the peak) and reports that nothing could be freed. `--vram-headroom 2`
  lowers the peak to ~21.4 GB at the same speed; it does not remove the line.
- At startup: `[ComfyUI-Manager] PyTorch is not installed` (it wants torchaudio), the
  `PatchTritonVAE requires triton` traceback (KJNodes, unused node), `No module named
  'llama_cpp'` (QwenVL GGUF variant, unused), matrix-nio and OpenGL_accelerate notices,
  `Ignoring jetson-stats` during install, and the two `[DEPRECATION WARNING] legacy API` lines.
- `Skip loading CUDA and cuDNN DLLs since torch is imported` is a GOOD sign: onnxruntime reuses
  torch's CUDA, face analysis is on GPU.
- `ConnectionResetError ... _call_connection_lost` after "Prompt executed": a browser tab closed
  or reloaded, harmless.
- HF Hub GET/HEAD requests for `Intel/dpt-hybrid-midas` (only with the MiDaS node of the two
  workflows here): metadata checks; the model itself comes from the bundled cache.

Not ignorable: `CUDAExecutionProvider is not in available provider names` means face analysis
fell back to CPU (crops take ~25 s instead of ~1 s). Fix: rerun the forced reinstall from
setup.ps1: `.venv\Scripts\pip.exe install --force-reinstall --no-deps onnxruntime-gpu==1.30.0`.

## Pinned pack versions

| Registry id | Version | Patched |
| --- | --- | --- |
| comfyui_instantid | 1.0.0 | |
| comfyui_controlnet_aux | 1.1.5 | |
| comfyui-liveportraitkj | 1.1.0 | import fix |
| comfyui-advancedliveportrait | 1.0.0 | paste-mask fix + debug print removed |
| ComfyUI-QwenVL | 2.2.0 | 8-bit dtype fix |
| comfyui-kjnodes | 1.5.0 | |
| comfyui-videohelpersuite | 1.7.9 | |
| ComfyUI-Crystools | 1.27.4 | optional, GPU monitor |

If the registry ever drops a version, copy that pack's folder from a good install instead.
Updating a pack discards its patch: re-run `git apply patches\<name>.patch` from the ComfyUI root,
or hold the version. PuLID, GGUF, FLUX, Qwen-Image-Edit and facexlib from the dev PC are not
needed.
