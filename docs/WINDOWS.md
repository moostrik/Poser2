# Windows show machine

How to freeze a Windows 11 machine for an unattended installation run: no updates, no notifications, no forced reboots, minimal background activity. Each step is a PowerShell block pasted into one elevated window (Start > `powershell` > Run as administrator, as the logged-in account). The final section restores normal operation after the run.

Most per-user (HKCU) changes only show in the Settings UI after a sign-out or reboot; the double reboot in [Verify](#verify) applies them all.

## Health check

Run these read-only checks first. Fix what they surface before freezing; a machine is frozen in a known-good state, not patched into one — installing fresh updates shortly before a run risks driver regressions worse than a few missed patch weeks.

```powershell
# Pending reboot / half-staged update (must be clean before pausing)
if (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired') { 'Reboot pending!' } else { 'Clean' }

# Unexpected shutdowns and BSODs, last 14 days (bugcheck 0 = power loss, not a crash)
Get-WinEvent -FilterHashtable @{LogName='System'; Id=41,6008} -MaxEvents 10 |
    Select-Object TimeCreated, Id, @{n='Msg';e={$_.Message.Split("`n")[0]}}

# Disk health and free space
Get-PhysicalDisk | Select-Object FriendlyName, HealthStatus
Get-Volume -DriveLetter C | Select-Object @{n='FreeGB';e={[math]::Round($_.SizeRemaining/1GB)}}
```

If a reboot is pending, let that update finish and reboot once before continuing; pausing does not cancel a staged restart.

## Freeze updates

Pauses all updates for 35 days (the maximum) and turns off every side channel. Put the expiry date in a calendar: an expired pause downloads updates and eventually forces a restart mid-run.

```powershell
$ux = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
$start  = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$expiry = (Get-Date).ToUniversalTime().AddDays(35).ToString('yyyy-MM-ddTHH:mm:ssZ')

# Pause all updates (same values the Settings UI writes)
Set-ItemProperty $ux PauseUpdatesStartTime        $start  -Type String
Set-ItemProperty $ux PauseUpdatesExpiryTime       $expiry -Type String
Set-ItemProperty $ux PauseFeatureUpdatesStartTime $start  -Type String
Set-ItemProperty $ux PauseFeatureUpdatesEndTime   $expiry -Type String
Set-ItemProperty $ux PauseQualityUpdatesStartTime $start  -Type String
Set-ItemProperty $ux PauseQualityUpdatesEndTime   $expiry -Type String

# "Get the latest updates as soon as they're available" off
Set-ItemProperty $ux IsContinuousInnovationOptedIn 0 -Type DWord

# "Receive updates for other Microsoft products" off
$sm = New-Object -ComObject Microsoft.Update.ServiceManager
if ($sm.Services | Where-Object ServiceID -eq '7971f918-a847-4430-9279-4a52d1efe18d') {
    $sm.RemoveService('7971f918-a847-4430-9279-4a52d1efe18d')
}

# Microsoft Store automatic app updates off (policy; restore removes this value)
if (-not (Test-Path 'HKLM:\SOFTWARE\Policies\Microsoft\WindowsStore')) { New-Item 'HKLM:\SOFTWARE\Policies\Microsoft\WindowsStore' -Force | Out-Null }
Set-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\WindowsStore' AutoDownload 2 -Type DWord

# Active hours covering the show's daily window (second guard against restarts)
Set-ItemProperty $ux ActiveHoursStart 9 -Type DWord
Set-ItemProperty $ux ActiveHoursEnd   3 -Type DWord
Set-ItemProperty $ux SmartActiveHoursState 0 -Type DWord

"Updates paused until $expiry"
```

Settings > Windows Update then shows "Updates paused until <date>".

## Silence notifications

```powershell
# Master notifications toggle off
Set-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\PushNotifications' ToastEnabled 0 -Type DWord

# Suggestion content off
$cdm = 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\ContentDeliveryManager'
Set-ItemProperty $cdm 'SubscribedContent-310093Enabled' 0 -Type DWord   # Windows welcome experience
Set-ItemProperty $cdm 'SubscribedContent-338389Enabled' 0 -Type DWord   # tips and suggestions
if (-not (Test-Path 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\UserProfileEngagement')) { New-Item 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\UserProfileEngagement' -Force | Out-Null }
Set-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\UserProfileEngagement' ScoobeSystemSettingEnabled 0 -Type DWord   # "finish setting up" nags

# Lock screen: static picture, no Spotlight downloads or overlay
Set-ItemProperty $cdm RotatingLockScreenEnabled        0 -Type DWord
Set-ItemProperty $cdm RotatingLockScreenOverlayEnabled 0 -Type DWord
Set-ItemProperty $cdm 'SubscribedContent-338387Enabled' 0 -Type DWord
```

## Trim startup

List what starts at login, then disable per entry. The `StartupApproved` flag is what Task Manager's Startup page writes, so entries stay listed there and can be re-enabled with one click.

```powershell
Get-CimInstance Win32_StartupCommand | Select-Object Name, Location, Command
```

Keep: remote access (Tailscale, Splashtop), the Windows Security tray, and cooling control software if it drives fan or pump curves. Disable: everything whose job is checking for updates (vendor download assistants, notifiers), launchers (Steam), and browser auto-starts. The entry name passed to `Set-ItemProperty` must match the `Run` value name exactly; HKCU entries get the flag under `$cu`, HKLM entries under `$lm`.

```powershell
$off = [byte[]](3,0,0,0,0,0,0,0,0,0,0,0)   # StartupApproved "disabled"
$cu  = 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run'
$lm  = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run'

# Worked example (2026 run); substitute the machine's own entry names
Set-ItemProperty $cu 'Steam' $off -Type Binary
Set-ItemProperty $cu 'MicrosoftEdgeAutoLaunch_FA57CEF577B1C07579203372CB758482' $off -Type Binary
Set-ItemProperty $lm 'Focusrite Notifier' $off -Type Binary
Set-ItemProperty $lm 'Logitech Download Assistant' $off -Type Binary
Set-ItemProperty $lm 'Logi Download Assistant' $off -Type Binary

# Edge: no startup boost, no background mode (policy; restore removes the values)
if (-not (Test-Path 'HKLM:\SOFTWARE\Policies\Microsoft\Edge')) { New-Item 'HKLM:\SOFTWARE\Policies\Microsoft\Edge' -Force | Out-Null }
Set-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Edge' StartupBoostEnabled 0 -Type DWord
Set-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Edge' BackgroundModeEnabled 0 -Type DWord

# No auto-restart of apps at sign-in
Set-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon' RestartApps 0 -Type DWord
```

The widgets taskbar value (`TaskbarDa`) is write-protected on current Windows 11; removing the widgets packages in the next section stops the feed service regardless.

## Remove software

Inventory, then remove per app:

```powershell
winget list          # desktop apps
Get-AppxPackage | Where-Object { -not $_.IsFramework -and -not $_.NonRemovable } | Select-Object Name   # Store apps
```

Never remove: anything named driver, runtime, or redistributable; Python versions the show uses; GPU software; camera and audio vendor packages; remote-access tools. A missing runtime crashes the show; a kept app costs nothing once its startup entry is off. When in doubt, keep it.

Always suspect: vendor update services, game overlays, Store consumer apps, and anything whose service already fails at boot (check the System log).

```powershell
winget uninstall "<exact DisplayName from winget list>"
Get-AppxPackage <PackageName> | Remove-AppxPackage
```

Store packages removed in the 2026 run, as a baseline set:

```powershell
'Microsoft.GamingApp','Microsoft.XboxGamingOverlay','Microsoft.Xbox.TCUI',
'Microsoft.XboxIdentityProvider','Microsoft.XboxSpeechToTextOverlay',
'MicrosoftWindows.Client.WebExperience','Microsoft.WidgetsPlatformRuntime',
'Microsoft.YourPhone','MicrosoftWindows.CrossDevice',
'Microsoft.MicrosoftSolitaireCollection','Clipchamp.Clipchamp','Microsoft.GetHelp',
'Microsoft.WindowsFeedbackHub','Microsoft.Todos','Microsoft.BingSearch',
'Microsoft.ZuneMusic','Microsoft.PowerAutomateDesktop' |
    ForEach-Object { Get-AppxPackage $_ | Remove-AppxPackage }
```

Adobe refuses its own uninstallers while Creative Cloud processes run, and the Creative Cloud app refuses to uninstall while products remain. Order: kill the processes, uninstall products, uninstall Creative Cloud last. When that still fails, Adobe's Creative Cloud Cleaner Tool (login-free, from helpx.adobe.com) removes everything; sweep afterwards for an orphaned `AdobeUpdateService`, the `AdobeNotificationClient` Appx package, and leftover `Adobe` folders under both Program Files trees. Licenses sit in the account, not the machine; reinstalling later restores them.

After any uninstall round, check the next boot's System log for services that still try to start (the uninstaller may leave the service registered):

```powershell
Get-Service | Where-Object Status -eq 'Stopped' | Where-Object StartType -eq 'Automatic'
sc.exe delete "<orphaned service name>"
```

## Clean caches

Clean early, so the show's first full run rebuilds shader and model caches long before opening.

```powershell
# User and system temp (locked files skip harmlessly), DirectX shader cache
Remove-Item "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item 'C:\Windows\Temp\*' -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:LOCALAPPDATA\D3DSCache\*" -Recurse -Force -ErrorAction SilentlyContinue

# Update download cache and component store (DISM takes several minutes)
Delete-DeliveryOptimizationCache -Force
Dism.exe /Online /Cleanup-Image /StartComponentCleanup

# Storage Sense off, so no automatic cleanup runs mid-show
Set-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\StorageSense\Parameters\StoragePolicy' '01' 0 -Type DWord
```

## Power

USB selective suspend is the single most likely cause of external camera dropouts after hours of running; the setting is hidden from `powercfg /query` summaries but exists on every scheme under the GUID below.

```powershell
powercfg /change monitor-timeout-ac 0      # screen never off
powercfg /change standby-timeout-ac 0      # sleep never

# USB selective suspend off
powercfg /setacvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0

# PCI Express link state power management off
powercfg /setacvalueindex SCHEME_CURRENT 501a4d13-42af-4429-9fd1-a8218c268e20 ee12f906-d277-404b-b6da-e5fa1a576df5 0

powercfg /setactive SCHEME_CURRENT

# Hibernate and fast startup off: every shutdown is a true cold boot
powercfg /h off
```

## Verify

Reboot twice. The first boot does one-time work: pending file deletions from uninstalls, Store package cleanup, policy application. The second boot is what every show-day boot looks like; judge that one.

After the second boot:

```powershell
# Nothing trimmed came back
Get-CimInstance Win32_StartupCommand | Select-Object Name, Location

# USB selective suspend reads 0x0
powercfg /query SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 | Select-String 'Current AC'

# Boot errors (compare against the known-benign list for the machine)
$boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2; StartTime=$boot} |
    Select-Object TimeCreated, Id, ProviderName, @{n='Msg';e={$_.Message.Split("`n")[0]}}
```

Then run the show at full load for a long soak, days before opening: temperatures, camera stability, frame timing. The soak doubles as the cache warm-up and as the check that any earlier one-off crash stays a one-off.

During the run: a short reboot before opening hours once a week.

## Resume after the run

```powershell
$ux = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'

# Resume updates
'PauseUpdatesStartTime','PauseUpdatesExpiryTime','PauseFeatureUpdatesStartTime','PauseFeatureUpdatesEndTime','PauseQualityUpdatesStartTime','PauseQualityUpdatesEndTime' |
    ForEach-Object { Remove-ItemProperty $ux $_ -ErrorAction SilentlyContinue }
(New-Object -ComObject Microsoft.Update.ServiceManager).AddService2('7971f918-a847-4430-9279-4a52d1efe18d',7,'') | Out-Null

# Store auto-updates back on
Remove-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\WindowsStore' AutoDownload -ErrorAction SilentlyContinue

# Notifications and Storage Sense back on
Set-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\PushNotifications' ToastEnabled 1 -Type DWord
Set-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\StorageSense\Parameters\StoragePolicy' '01' 1 -Type DWord

# Edge policies off
Remove-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Edge' StartupBoostEnabled,BackgroundModeEnabled -ErrorAction SilentlyContinue

# Re-enable startup entries (Task Manager > Startup apps works too)
$on = [byte[]](2,0,0,0,0,0,0,0,0,0,0,0)
Set-ItemProperty 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run' 'Steam' $on -Type Binary
```

Optional, only to restore the previous behavior: `powercfg /h on`, and `Set-ItemProperty $ux IsContinuousInnovationOptedIn 1`. Notification suppression and removed Store apps can stay as they are; removed apps reinstall from the Store or winget.

## Machine health between runs

Do these after resuming updates, not during a run; firmware and drivers stay frozen while a show is live.

| Task                                | When                      | Why |
| ----------------------------------- | ------------------------- | --- |
| Install the Windows update backlog  | Right after the run       | A long pause accumulates patches; install and reboot while there is slack to catch a regression |
| Update GPU driver                   | After the run             | Frozen driver ages a month+ per run; also first suspect if a BSOD recurred during the run |
| Check BIOS against the board page   | After the run             | The i7-13700K is in the Raptor Lake voltage-degradation family; run a BIOS with the 2024 microcode fixes (0x12B or later) and the Intel Default power profile, not an unlimited board profile |
| Dust out filters, radiator, GPU     | After every run           | A mini-ITX case under weeks of sustained load clogs fast; dust is the main thermal drift in this build |
| Watch temperatures under load       | After cleaning            | NZXT CAM is not installed; use HWiNFO or the BIOS fan page. Rising CPU temps at equal load mean dust, a tired AIO pump, or paste |
| SSD health and firmware             | Yearly                    | `Get-PhysicalDisk` HealthStatus plus the 990 PRO firmware in Samsung Magician; keep the drive under ~80% full |
| AIO liquid cooler                   | Replace at ~5–7 years     | Pumps wear and loops permeate; a failing pump shows as creeping CPU temps long before failure |
| Memory                              | Only if instability appears | Memtest86 overnight before blaming anything else; at 2026 DRAM prices a false RAM diagnosis is expensive. The 2×48 GB DDR5-6000 kit is near-unreplaceable at sane cost until supply recovers (~2027) |
| Keep the machine trimmed            | Ongoing                   | Startup items and update services reaccumulate with every tool install; re-run the inventory in [Trim startup](#trim-startup) before the next freeze |

## This machine, 2026 run

| Item                    | State                                                            |
| ----------------------- | ---------------------------------------------------------------- |
| Machine                 | IAAI, Windows 11 Pro 25H2                                        |
| Updates paused until    | 2026-11-10 — resume or re-pause before then                      |
| Startup disabled        | Steam, Edge auto-launch, Focusrite Notifier, Logitech ×2         |
| Startup kept            | Tailscale, Splashtop, Windows Security tray                      |
| Removed                 | Adobe (all), GeForce Experience, FrameView, old Nsights, ROG Live Service, Xbox/Game Bar, Widgets, Phone Link, Store consumer apps |
| Kept deliberately       | Splashtop (remote control), Spotify, Horizon Forbidden West, Store Python 3.10 |
| Cooling                 | NZXT 240 mm AIO on the CPU, GPU in its own case compartment **(site fact)**; both stay cool under load, CPU holds max boost. Hardware fan/pump profile; NZXT CAM is not installed |
| BIOS                    | ASUS 3001 (2025-04) — includes the Raptor Lake anti-degradation microcode; after the run, confirm the power profile is Intel Default, not an unlimited board profile |
| CPU replacement path    | LGA1700: any 12th–14th gen drops in on this BIOS. A spare LGA1700 chip is the machine's cheapest insurance; RAM and GPU are the near-irreplaceable parts |
| PSU                     | Cooler Master 850 W SFX Gold **(site fact)** — NVIDIA's recommended minimum for the RTX 4090. Fine with the i7 at stock; an i9 swap requires Intel Default power limits, since i9-at-unlimited plus 4090 transients exceeds the margin |
| Benign boot errors      | Intel Platform License Manager timeout (7009), FDResPub (7023) — pre-existing |
