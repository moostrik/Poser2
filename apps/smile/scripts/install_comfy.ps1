<#
Installs, or checks, the ComfyUI server of SMILE at <repo>\ComfyUI from comfy\comfy.lock:
ComfyUI core at its pinned commit, the node packs at theirs, the patches of comfy\patches, our node
pack comfy\aiia_grin as a junction, the venv from comfy\requirements.lock, and ComfyUI\models as a junction
to the models folder beside the repo. Safe to run again: everything that is in place is skipped, so a
second run is the check. Exit 1 on the first failure, with its reason. Called by scripts\install.bat,
which installs the models afterwards (the model tool, modules\comfy\models.py).

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_comfy.ps1 [-Target DIR] [-Models DIR]

Everything is cloned with core.autocrlf off: the packs' files stay as upstream has them, which the patches
need (a patch made for LF does not apply to a CRLF checkout).
#>
param(
    [string]$Target = (Join-Path (Split-Path $PSScriptRoot -Parent) 'ComfyUI'),
    [string]$Models = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'smile-models')
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$src = Join-Path $repo 'comfy'

function Step([string]$name) { Write-Host "`n== $name" -ForegroundColor Cyan }
function Note([string]$text) { Write-Host "   $text" }
function Fail([string]$text) { throw $text }
function RunGit {
    # git with its exit code checked; output returned
    $out = & git.exe @args 2>&1
    if ($LASTEXITCODE -ne 0) { Fail "git $($args -join ' ') failed:`n$out" }
    return $out
}
function Quiet {
    # a native command that may fail: True when its exit code is 0, its output dropped (under
    # ErrorActionPreference Stop a redirected stderr line would throw, so it is lowered here)
    $exe = $args[0]
    $rest = @($args | Select-Object -Skip 1)       # always an array: a lone string would splat into its characters
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $exe @rest 2>&1 | Out-Null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $eap
    return ($code -eq 0)
}
function IsLink([string]$path) {
    $item = Get-Item $path -Force -ErrorAction SilentlyContinue
    return ($null -ne $item) -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
}
function PackVersion([string]$path) {
    $toml = Join-Path $path 'pyproject.toml'
    if (-not (Test-Path $toml)) { return '' }
    $m = Select-String -Path $toml -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value } else { return '' }
}

try {
    $lock = Get-Content (Join-Path $src 'comfy.lock') -Raw -Encoding UTF8 | ConvertFrom-Json

    # --- prerequisites -----------------------------------------------------------------------------
    Step 'Prerequisites'
    if (-not (Quiet nvidia-smi --query-gpu=driver_version '--format=csv,noheader')) { Fail 'nvidia-smi not found: no NVIDIA driver' }
    $smi = & nvidia-smi --query-gpu=driver_version '--format=csv,noheader'
    $driver = [int](("$smi" -split '\.')[0])
    if ($driver -lt 580) { Fail "NVIDIA driver $smi is too old, the CUDA 13.0 torch needs 580 or newer" }
    Note "NVIDIA driver $smi"
    if (-not (Quiet py -3.12 --version)) { Fail 'Python 3.12 not found (py -3.12): install it with the py launcher' }
    Note "$(& py -3.12 --version)"
    if (-not (Quiet git --version)) { Fail 'git not found' }

    # --- ComfyUI core, pinned ------------------------------------------------------------------------
    $commit = $lock.comfyui.commit
    Step "ComfyUI core at $($commit.Substring(0, 8))"
    if (-not (Test-Path (Join-Path $Target '.git'))) {
        Note "cloning $($lock.comfyui.git) into $Target"
        RunGit -c core.autocrlf=false clone --quiet $lock.comfyui.git $Target | Out-Null
        RunGit -C $Target config core.autocrlf false | Out-Null
    }
    $head = (RunGit -C $Target rev-parse HEAD) -join ''
    if ($head -ne $commit) {
        if (-not (Quiet git -C $Target cat-file -e "$commit^{commit}")) { RunGit -C $Target fetch --quiet origin $commit | Out-Null }
        RunGit -C $Target checkout --quiet -B smile $commit | Out-Null
        Note "checked out $($commit.Substring(0, 8)) on branch smile"
    } else {
        Note 'at the pinned commit'
    }
    # models/ is left out: the junction step below moves its placeholders into the models folder
    $dirty = & git -C $Target status --porcelain --untracked-files=no -- . ':!models'
    if ($dirty) { Fail "ComfyUI core has local changes (it must stay unmodified):`n$($dirty -join "`n")" }

    # --- node packs, pinned ---------------------------------------------------------------------------
    Step 'Node packs'
    $nodes = Join-Path $Target 'custom_nodes'
    New-Item -ItemType Directory -Force $nodes | Out-Null
    foreach ($pack in $lock.packs) {
        $dst = Join-Path $nodes $pack.id
        if ((Test-Path $dst) -and -not (Get-ChildItem $dst -Force | Select-Object -First 1)) { Remove-Item $dst }    # an empty folder (a failed download) counts as absent
        if ($pack.commit) {
            if (Test-Path (Join-Path $dst '.git')) {
                $at = (& git -C $dst rev-parse HEAD) -join ''
                if ($at -ne $pack.commit) {
                    $changes = & git -C $dst status --porcelain --untracked-files=no
                    if ($changes) { Fail "$($pack.id) is at $($at.Substring(0, 8)) with local changes, the pin is $($pack.commit.Substring(0, 8)): sort it out by hand" }
                    if (-not (Quiet git -C $dst cat-file -e "$($pack.commit)^{commit}")) { RunGit -C $dst fetch --quiet origin $pack.commit | Out-Null }
                    RunGit -C $dst checkout --quiet --detach $pack.commit | Out-Null
                    Note "$($pack.id): moved to $($pack.commit.Substring(0, 8))"
                } else {
                    Note "$($pack.id) $($pack.version): at its pin"
                }
            } elseif (Test-Path $dst) {
                # a folder without git: the registry archive of the old setup. Its version must be the pin.
                $v = PackVersion $dst
                if ($v -ne $pack.version) { Fail "$($pack.id) at $dst is version '$v', the pin is $($pack.version): remove the folder and run again" }
                Note "$($pack.id) $($pack.version): archive install, kept"
            } else {
                Note "$($pack.id): cloning $($pack.git)"
                RunGit -c core.autocrlf=false clone --quiet $pack.git $dst | Out-Null
                RunGit -C $dst config core.autocrlf false | Out-Null
                RunGit -C $dst checkout --quiet --detach $pack.commit | Out-Null
                Note "$($pack.id) $($pack.version): at $($pack.commit.Substring(0, 8))"
            }
        } else {
            # no git pin: the Comfy Registry archive of the version
            if (Test-Path $dst) {
                $v = PackVersion $dst
                if ($v -ne $pack.version) { Fail "$($pack.id) at $dst is version '$v', the pin is $($pack.version): remove the folder and run again" }
                Note "$($pack.id) $($pack.version): in place"
            } else {
                $meta = Invoke-RestMethod "https://api.comfy.org/nodes/$($pack.id)/versions/$($pack.version)"
                $archive = Join-Path $env:TEMP "$($pack.id)-$($pack.version).archive"
                Invoke-WebRequest $meta.downloadUrl -OutFile $archive
                New-Item -ItemType Directory -Force $dst | Out-Null
                & (Join-Path $env:SystemRoot 'System32\tar.exe') -xf $archive -C $dst     # Windows' bsdtar reads both the tar.gz and the zip the registry serves (Git's tar on the PATH would not)
                if ($LASTEXITCODE -ne 0) { Fail "could not extract $archive" }
                Remove-Item $archive
                Note "$($pack.id) $($pack.version): registry archive installed"
            }
        }
    }
    $known = @($lock.packs | ForEach-Object { $_.id }) + @('aiia_grin')
    foreach ($dir in Get-ChildItem $nodes -Directory | Where-Object { $_.Name -ne '__pycache__' -and $known -notcontains $_.Name }) {
        Write-Host "   WARNING: $($dir.Name) is not in comfy.lock (an extra pack; remove it if SMILE does not need it)" -ForegroundColor Yellow
    }

    # --- patches (pack fixes; a stopgap each, see data\InstallMigration.md section 4) -----------------
    Step 'Patches'
    foreach ($patch in Get-ChildItem (Join-Path $src 'patches\*.patch')) {
        # the pack a patch belongs to is in its first path: --- a/custom_nodes/<pack>/...
        $first = (Select-String -Path $patch.FullName -Pattern '^--- a/custom_nodes/([^/]+)/' | Select-Object -First 1)
        if (-not $first) { Fail "$($patch.Name): no custom_nodes path in it" }
        $packDir = Join-Path $nodes $first.Matches[0].Groups[1].Value
        if (-not (Test-Path $packDir)) { Fail "$($patch.Name): its pack $packDir is not installed" }
        # --ignore-whitespace: the archives of some packs have CRLF files, the patches are LF
        if (Quiet git -C $packDir apply -p3 --ignore-whitespace --check $patch.FullName) {
            RunGit -C $packDir apply -p3 --ignore-whitespace $patch.FullName | Out-Null
            Note "$($patch.Name): applied"
            continue
        }
        if (Quiet git -C $packDir apply -p3 --ignore-whitespace --reverse --check $patch.FullName) { Note "$($patch.Name): already applied"; continue }
        Fail "$($patch.Name) neither applies nor is applied: the pack's files differ from what the patch expects"
    }

    # --- our node pack: a junction, so an edit in the repo needs only a ComfyUI restart ----------------
    Step 'aiia_grin'
    $grin = Join-Path $nodes 'aiia_grin'
    if ((Test-Path $grin) -and -not (IsLink $grin)) {
        Remove-Item -Recurse -Force $grin
        Note 'the copied folder of the old setup removed'
    }
    if (-not (Test-Path $grin)) {
        New-Item -ItemType Junction -Path $grin -Target (Join-Path $src 'aiia_grin') | Out-Null
        Note "junction made to $src\aiia_grin"
    } else {
        Note 'junction in place'
    }

    # --- venv from the lock ----------------------------------------------------------------------------
    Step 'Python venv'
    $venv = Join-Path $Target '.venv'
    $python = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $python)) {
        & py -3.12 -m venv $venv
        if ($LASTEXITCODE -ne 0) { Fail 'could not create the venv' }
        Note "venv created at $venv"
    }
    $lockFile = Join-Path $src 'requirements.lock'
    $lockHash = (Get-FileHash $lockFile -Algorithm SHA256).Hash
    $stamp = Join-Path $venv 'smile-install.stamp'
    if ((Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $lockHash)) {
        Note 'requirements.lock unchanged since the last install, pip skipped'
    } else {
        & $python -m pip install --quiet --upgrade pip
        # --no-deps: the lock is a complete freeze; resolving it would fail on metadata the working venv
        # itself contradicts (facenet-pytorch asks numpy<2, the install runs numpy 2.5.3)
        & $python -m pip install --no-deps -r $lockFile
        if ($LASTEXITCODE -ne 0) { Fail 'pip install -r comfy\requirements.lock failed' }
        Set-Content -Path $stamp -Value $lockHash -Encoding ascii
        Note 'requirements.lock installed'
    }
    $ortCheck = "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers()"
    if (-not (Quiet $python -c $ortCheck)) {
        $ort = (Select-String -Path $lockFile -Pattern '^onnxruntime-gpu==(\S+)' | Select-Object -First 1).Matches[0].Groups[1].Value
        Note "onnxruntime has no CUDA provider: reinstalling onnxruntime-gpu==$ort"
        & $python -m pip install --quiet --force-reinstall --no-deps "onnxruntime-gpu==$ort"
        if (-not (Quiet $python -c $ortCheck)) { Fail 'onnxruntime still has no CUDA provider (face analysis would run on the CPU)' }
    }
    Note 'onnxruntime CUDA provider ok'

    # --- the models folder: ComfyUI\models is a junction to it ------------------------------------------
    Step "Models folder $Models"
    New-Item -ItemType Directory -Force $Models | Out-Null
    $modelsDir = Join-Path $Target 'models'
    if (IsLink $modelsDir) {
        $to = (Get-Item $modelsDir -Force).Target
        if ("$to" -and ((Resolve-Path "$to").Path -ne (Resolve-Path $Models).Path)) { Fail "ComfyUI\models is a junction to $to, not to $Models" }
        Note 'junction in place'
    } else {
        if (Test-Path $modelsDir) {
            # a real folder (a fresh clone's placeholders, or an old install's models): moved into the models
            # folder, nothing overwritten; what stays behind is a name clash
            & robocopy $modelsDir $Models /E /MOVE /XC /XN /XO /NFL /NDL /NJH /NJS /NP | Out-Null
            if ($LASTEXITCODE -ge 8) { Fail "robocopy of $modelsDir into $Models failed ($LASTEXITCODE)" }
            # what stayed behind exists on both sides: the same bytes (ComfyUI's put_*_here placeholders) is no clash
            foreach ($file in Get-ChildItem $modelsDir -Recurse -File -Force -ErrorAction SilentlyContinue) {
                $twin = Join-Path $Models $file.FullName.Substring($modelsDir.Length + 1)
                if ((Test-Path $twin) -and ((Get-Item $twin).Length -eq $file.Length) -and
                    (($file.Length -eq 0) -or ((Get-FileHash $twin -Algorithm SHA256).Hash -eq (Get-FileHash $file.FullName -Algorithm SHA256).Hash))) {
                    Remove-Item $file.FullName -Force
                }
            }
            $left = @(Get-ChildItem $modelsDir -Recurse -File -Force -ErrorAction SilentlyContinue)
            if ($left.Count -gt 0) {
                Fail "these files exist in both $modelsDir and ${Models}:`n$(($left | ForEach-Object { $_.FullName }) -join "`n")`nresolve them by hand, then run again"
            }
            Remove-Item -Recurse -Force $modelsDir -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Junction -Path $modelsDir -Target $Models | Out-Null
        Note "junction made: $modelsDir -> $Models"
    }

    Write-Host "`nComfyUI is installed at $Target. Next: the models (scripts\install.bat does it: py -3.12 -m modules.comfy.models install)." -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
