<#
Installs, or checks, the ComfyUI server of SMILE at <repo>\ComfyUI from comfy\comfy.lock:
ComfyUI core at its pinned commit, the node packs at theirs from the archives in comfy\packs (the
Comfy Registry's files, kept in the repo), any patch in comfy\patches (none today), our node pack
comfy\aiia_grin as a junction, and the venv from comfy\requirements.lock. The models live in
ComfyUI\models as ComfyUI has it; scripts\install.bat fills that folder afterwards with the model tool
(modules\comfy\models.py). Safe to run again: everything in place is skipped, so a second run is the
check. Exit 1 on the first failure, with its reason.

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_comfy.ps1 [-Target DIR]

Updating (first on the dev PC, never on the ICK PC during the show):

    install_comfy.ps1 -Update [-ComfyCommit SHA]   the install as it is gets its picture taken (modules.comfy.smoke
                                                   --save output\smoke\before: a fixed-seed portrait and grin frame),
                                                   is renamed <Target>-previous, and a new one is built with ComfyUI
                                                   at origin HEAD (or the commit given), every pack at the registry's
                                                   latest version (its archive downloaded into comfy\packs) and the venv
                                                   from their requirements.txt files (not the lock), the models copied
                                                   over from the previous install, then smoke-tested with its picture
                                                   in output\smoke\after
    install_comfy.ps1 -Accept                      the pictures compared and good: comfy.lock and requirements.lock
                                                   are written from the new install and <Target>-previous is deleted;
                                                   commit the locks and the new archives in comfy\packs
    install_comfy.ps1 -Rollback                    the new install is deleted and <Target>-previous comes back

An update may change what a node expects (an input added, a node renamed): the smoke test then refuses the
workflow and names the node; the fix belongs in modules\comfy\workflow.py or a re-export of the workflow,
never in ComfyUI or a pack.

Everything is cloned with core.autocrlf off: the packs' files stay as upstream has them, which a patch
would need (a patch made for LF does not apply to a CRLF checkout).
#>
param(
    [string]$Target = (Join-Path (Split-Path $PSScriptRoot -Parent) 'ComfyUI'),
    [switch]$Update,
    [string]$ComfyCommit = '',
    [switch]$Accept,
    [switch]$Rollback
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
function RemoveInstall([string]$dir) {
    # the aiia_grin junction goes first, as a link, so the repo folder behind it is not touched;
    # rmdir does not follow junctions either
    $link = Join-Path $dir 'custom_nodes\aiia_grin'
    if (IsLink $link) { (Get-Item $link -Force).Delete() }
    & cmd /c rmdir /s /q "$dir"
    if (Test-Path $dir) { Fail "could not remove $dir" }
}
function WriteText([string]$path, [string]$text) {
    [IO.File]::WriteAllText($path, $text.Replace("`r`n", "`n"), (New-Object Text.UTF8Encoding $false))
}
function WriteLocks($lockObj, [string]$target, [string]$python) {
    # the locks from the install as it stands: the core commit, the packs' versions, the venv's freeze
    $nodes = Join-Path $target 'custom_nodes'
    $packs = @(foreach ($pack in $lockObj.packs) {
        [ordered]@{ id = $pack.id; version = (PackVersion (Join-Path $nodes $pack.id)); git = $pack.git; commit = $null; note = $pack.note }
    })
    $new = [ordered]@{
        _comment = $lockObj._comment
        comfyui = [ordered]@{ git = $lockObj.comfyui.git; commit = ((& git -C $target rev-parse HEAD) -join ''); date = (Get-Date -Format yyyy-MM-dd) }
        torch_index = $lockObj.torch_index
        packs = $packs
    }
    $json = ($new | ConvertTo-Json -Depth 5).Replace('\u0027', "'") + "`n"
    WriteText (Join-Path $src 'comfy.lock') $json
    $freeze = @(& $python -m pip freeze --all) | Where-Object { $_ -notmatch '^onnxruntime==' }
    $header = @(
        "# The ComfyUI venv, pinned: pip freeze --all of the install accepted on $(Get-Date -Format yyyy-MM-dd) (scripts\install_comfy.ps1 -Accept).",
        "# install_comfy.ps1 installs this file with --no-deps (a freeze is complete); the packs' requirements.txt are not run.",
        "# torch/torchvision are CUDA builds from the PyTorch index below; the plain onnxruntime is left out (the GPU build must win).",
        "--extra-index-url $($lockObj.torch_index)")
    WriteText (Join-Path $src 'requirements.lock') (($header + $freeze) -join "`n") + "`n"
    $hash = (Get-FileHash (Join-Path $src 'requirements.lock') -Algorithm SHA256).Hash
    Set-Content -Path (Join-Path $target '.venv\smile-install.stamp') -Value $hash -Encoding ascii
    Note "comfy.lock: ComfyUI $($new.comfyui.commit.Substring(0, 8)), packs $(($packs | ForEach-Object { "$($_.id) $($_.version)" }) -join ', ')"
    Note "requirements.lock: $($freeze.Count) lines"
}

$previous = "$Target-previous"
$appPython = Join-Path $repo '.venv\Scripts\python.exe'
$smokeOut = Join-Path $repo 'output\smoke'

try {
    $lock = Get-Content (Join-Path $src 'comfy.lock') -Raw -Encoding UTF8 | ConvertFrom-Json

    if ($Rollback) {
        Step 'Rollback'
        if (-not (Test-Path $previous)) { Fail "nothing to roll back to: $previous is not there" }
        if (Test-Path $Target) { RemoveInstall $Target }
        Rename-Item $previous (Split-Path $Target -Leaf)
        Write-Host "`n$previous is $Target again; the locks were never changed." -ForegroundColor Green
        exit 0
    }
    if ($Accept) {
        Step 'Accept the update'
        if (-not (Test-Path $previous)) { Fail "no update to accept: $previous is not there" }
        WriteLocks $lock $Target (Join-Path $Target '.venv\Scripts\python.exe')
        RemoveInstall $previous
        Write-Host "`nThe update is the install: comfy.lock and requirements.lock describe it, $previous is gone. Commit the locks and comfy\packs (git rm the archives of versions no longer in the lock)." -ForegroundColor Green
        exit 0
    }
    if ($Update) {
        Step 'Update: new pins'
        if (Test-Path $previous) { Fail "$previous exists: -Accept or -Rollback the last update first" }
        if (-not (Test-Path $appPython)) { Fail "the app venv is missing ($appPython): run scripts\install.bat first" }
        if (Test-Path $Target) {
            Note "the picture of the install as it is -> $smokeOut\before"
            & $appPython -m modules.comfy.smoke --comfy $Target --save (Join-Path $smokeOut 'before')
            if ($LASTEXITCODE -ne 0) { Fail 'the install as it is does not pass its own smoke test; sort that out before updating' }
            # nothing may run from the folder while it is renamed (a ComfyUI, or a child it left)
            foreach ($proc in Get-Process | Where-Object { $_.Path -and $_.Path -like "$Target\*" }) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
            $tries = 0
            while ($true) {
                try { Rename-Item $Target (Split-Path $previous -Leaf) -ErrorAction Stop; break }
                catch { if (++$tries -ge 10) { throw }; Start-Sleep 3 }
            }
            Note "$Target renamed to $previous"
        }
        $lock.comfyui.commit = if ($ComfyCommit) { $ComfyCommit } else { ((& git ls-remote $lock.comfyui.git HEAD) -split '\s+')[0] }
        Note "ComfyUI: $($lock.comfyui.commit.Substring(0, 8)) (origin HEAD)"
        foreach ($pack in $lock.packs) {
            $meta = Invoke-RestMethod "https://api.comfy.org/nodes/$($pack.id)"
            $was = $pack.version
            $pack.version = $meta.latest_version.version
            $pack.commit = $null
            Note "$($pack.id): $was -> $($pack.version)"
        }
    }

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
    $dirty = & git -C $Target status --porcelain --untracked-files=no
    if ($dirty) { Fail "ComfyUI core has local changes (it must stay unmodified):`n$($dirty -join "`n")" }

    # --- node packs, pinned ---------------------------------------------------------------------------
    Step 'Node packs'
    $nodes = Join-Path $Target 'custom_nodes'
    $packs = Join-Path $src 'packs'
    New-Item -ItemType Directory -Force $nodes, $packs | Out-Null
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
                # the archive lives in the repo (comfy\packs, git LFS); a version not there yet (an update) comes
                # from the Comfy Registry and stays in comfy\packs, to be committed with the locks
                $archive = Get-ChildItem (Join-Path $packs "$($pack.id)-$($pack.version).*") -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($archive -and $archive.Length -lt 1024) { Fail "$($archive.Name) is a git LFS pointer, not the archive: run git lfs pull" }
                if (-not $archive) {
                    $meta = Invoke-RestMethod "https://api.comfy.org/nodes/$($pack.id)/versions/$($pack.version)"
                    $tmp = Join-Path $packs "$($pack.id)-$($pack.version).download"
                    Invoke-WebRequest $meta.downloadUrl -OutFile $tmp
                    # the registry names every file node.tar.gz, some are zips: name it by its content
                    $magic = [IO.File]::ReadAllBytes($tmp)[0..1]
                    $ext = if ($magic[0] -eq 0x50 -and $magic[1] -eq 0x4B) { '.zip' } else { '.tar.gz' }
                    $archive = Rename-Item $tmp "$($pack.id)-$($pack.version)$ext" -PassThru
                    Note "$($pack.id) $($pack.version): archive downloaded from the registry into comfy\packs"
                }
                New-Item -ItemType Directory -Force $dst | Out-Null
                & (Join-Path $env:SystemRoot 'System32\tar.exe') -xf $archive.FullName -C $dst     # Windows' bsdtar reads both formats (Git's tar on the PATH would not)
                if ($LASTEXITCODE -ne 0) { Fail "could not extract $($archive.FullName)" }
                Note "$($pack.id) $($pack.version): installed from $($archive.Name)"
            }
        }
    }
    $known = @($lock.packs | ForEach-Object { $_.id }) + @('aiia_grin')
    foreach ($dir in Get-ChildItem $nodes -Directory | Where-Object { $_.Name -ne '__pycache__' -and $known -notcontains $_.Name }) {
        Write-Host "   WARNING: $($dir.Name) is not in comfy.lock (an extra pack; remove it if SMILE does not need it)" -ForegroundColor Yellow
    }

    # --- patches (pack fixes; a stopgap each, see data\InstallMigration.md section 4) -----------------
    Step 'Patches'
    $patches = @(Get-ChildItem (Join-Path $src 'patches\*.patch') -ErrorAction SilentlyContinue)     # the folder may not exist: no pack is patched
    if ($patches.Count -eq 0) { Note 'none: no pack is patched' }
    foreach ($patch in $patches) {
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
    if ($Update) {
        # unlocked: what ComfyUI and the packs ask for today, the lock is written at -Accept from the result
        & $python -m pip install --quiet --upgrade pip
        & $python -m pip install torch torchvision --index-url $lock.torch_index
        if ($LASTEXITCODE -ne 0) { Fail 'pip install torch failed' }
        & $python -m pip install -r (Join-Path $Target 'requirements.txt')
        if ($LASTEXITCODE -ne 0) { Fail "pip install -r ComfyUI's requirements.txt failed" }
        & $python -m pip install comfyui-manager
        foreach ($req in Get-ChildItem (Join-Path $nodes '*\requirements.txt')) {
            & $python -m pip install -r $req.FullName
            if ($LASTEXITCODE -ne 0) { Fail "pip install -r $($req.FullName) failed" }
        }
        # a pack's metadata may pull numpy back below 2 (facenet-pytorch); the install runs on numpy 2
        & $python -m pip install --quiet --upgrade numpy numba
        & $python -m pip install --quiet --force-reinstall --no-deps onnxruntime-gpu
        Note 'venv built from the requirements of today (unlocked)'
    } elseif ((Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $lockHash)) {
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

    # --- the models of an update: copied over from the previous install ----------------------------------
    if ($Update -and (Test-Path (Join-Path $previous 'models'))) {
        Step 'Models from the previous install'
        & robocopy (Join-Path $previous 'models') (Join-Path $Target 'models') /E /XC /XN /XO /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { Fail "robocopy of the models from $previous failed ($LASTEXITCODE)" }
        Note "copied into $Target\models (nothing overwritten)"
    }

    if ($Update) {
        Step 'Update: the proof'
        Note "the smoke test of the new install, its picture -> $smokeOut\after"
        & $appPython -m modules.comfy.smoke --comfy $Target --save (Join-Path $smokeOut 'after')
        if ($LASTEXITCODE -ne 0) { Fail "the new install fails its smoke test: look at it, then install_comfy.ps1 -Rollback (or fix and run -Update again after -Rollback)" }
        Write-Host "`nThe update is built at $Target and passes the smoke test. Compare $smokeOut\before with $smokeOut\after (portrait.png, grin_last.png)," -ForegroundColor Green
        Write-Host "then: install_comfy.ps1 -Accept (writes the locks, removes $previous) or install_comfy.ps1 -Rollback." -ForegroundColor Green
        exit 0
    }
    Write-Host "`nComfyUI is installed at $Target. Next: the models into $Target\models (scripts\install.bat does it: py -3.12 -m modules.comfy.models install)." -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
