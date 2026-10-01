# Sets up the ComfyUI generation server for aiia2.
# Needs internet (ComfyUI clone, torch wheels, Comfy Registry archives).
# Expects next to this script: patches\, aiia_grin\, workflows\.
# Models are copied separately, see README.md.

param(
    # default: a ComfyUI folder next to this script's folder, i.e. <aiia2 repo>\ComfyUI
    [string]$Target = (Join-Path (Split-Path $PSScriptRoot -Parent) "ComfyUI"),
    [string]$ComfyCommit = "4ef23c34d950eecc37040a21ee1741a49d2e44b1"
)
$ErrorActionPreference = "Stop"
$src = $PSScriptRoot

# node packs, pinned to the versions verified on the dev PC (Comfy Registry id, version)
$packs = @(
    @('comfyui_instantid',            '1.0.0'),
    @('comfyui_controlnet_aux',       '1.1.5'),
    @('comfyui-liveportraitkj',       '1.1.0'),
    @('comfyui-advancedliveportrait', '1.0.0'),
    @('ComfyUI-QwenVL',               '2.2.0'),
    @('comfyui-kjnodes',              '1.5.0'),
    @('comfyui-videohelpersuite',     '1.7.9'),
    @('ComfyUI-Crystools',            '1.27.4')
)

# --- prerequisites ---------------------------------------------------------
$driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader) -split '\.' | Select-Object -First 1
if ([int]$driver -lt 580) { throw "NVIDIA driver $driver too old; cu130 needs 580+" }
py -3.12 --version ; if (-not $?) { throw "Python 3.12 not found (py -3.12)" }

# --- ComfyUI core, pinned --------------------------------------------------
if (-not (Test-Path "$Target\.git")) {
    git clone https://github.com/comfyanonymous/ComfyUI $Target
}
git -C $Target checkout -B smile $ComfyCommit
Set-Location $Target

# --- node packs from the Comfy Registry ------------------------------------
foreach ($pk in $packs) {
    $id, $ver = $pk
    $dst = "custom_nodes\$id"
    if (Test-Path $dst) { Write-Host "skip $id (exists)"; continue }
    $meta = Invoke-RestMethod "https://api.comfy.org/nodes/$id/versions/$ver"
    $archive = "$env:TEMP\$id-$ver" + $(if ($meta.downloadUrl -match '\.tar\.gz$') { '.tar.gz' } else { '.zip' })
    Invoke-WebRequest $meta.downloadUrl -OutFile $archive
    New-Item -ItemType Directory -Force $dst | Out-Null
    if ($archive.EndsWith('.zip')) { Expand-Archive $archive $dst -Force } else { tar -xf $archive -C $dst }
    Remove-Item $archive
    Write-Host "installed $id $ver"
}

# --- fixes (three pack fixes; ComfyUI core stays unmodified) -----------------
foreach ($p in Get-ChildItem "$src\patches\*.patch") {
    git apply $p.FullName
    Write-Host "applied $($p.Name)"
}

# --- models (bundled next to this script; ~23 GB) ---------------------------
if (Test-Path "$src\models") {
    robocopy "$src\models" models /E /XC /XN /XO /XD hf-cache /NFL /NDL /NJH | Out-Null
    $hub = "$env:USERPROFILE\.cache\huggingface\hub"
    robocopy "$src\models\hf-cache" $hub /E /XC /XN /XO /NFL /NDL /NJH | Out-Null
    Write-Host "models installed"
} else {
    Write-Host "no models\ folder found - copy the models by hand, see README.md"
}

# --- our node + workflow ----------------------------------------------------
Copy-Item "$src\aiia_grin" custom_nodes\ -Recurse -Force
New-Item -ItemType Directory -Force user\default\workflows | Out-Null
Copy-Item "$src\workflows\*" user\default\workflows\

# --- venv, torch cu130, requirements ----------------------------------------
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pip.exe install comfyui-manager==4.2.2
foreach ($req in Get-ChildItem custom_nodes\*\requirements.txt) {
    .venv\Scripts\pip.exe install -r $req.FullName
}
# undeclared/conflicting deps, pinned to the dev-verified state:
# numba is imported by liveportraitkj but missing from its requirements (it came in
# via facexlib/PuLID on the dev PC); liveportraitkj also pins numpy<=1.26.4 while the
# verified install runs numpy 2.5.3, so restore that last.
.venv\Scripts\pip.exe install numba==0.67.0 numpy==2.5.3
# the CPU onnxruntime package can overwrite the GPU one's files depending on pack install
# order; reinstall the GPU build last so insightface/landmarks run on CUDA
.venv\Scripts\pip.exe install --force-reinstall --no-deps onnxruntime-gpu==1.30.0

Write-Host "`nDone. Now copy the model folders (README.md), then start with:"
Write-Host "  cd $Target; .venv\Scripts\activate; python main.py --enable-manager --preview-method auto"
